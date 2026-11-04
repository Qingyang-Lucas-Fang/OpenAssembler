"""Run the ptxas-derived fixed-pipe latency sweep entirely under docs/research/.

This invokes the existing SASS encoder/emulator and GPU runner, but patches only
their output roots to docs/research/a100-latency/corpus/. It measures each
producer/consumer SASS dependency at producer stalls 1..8 (predicate guards
1..15) against the emulator on a real A100.

    modal run docs/research/a100_latency_sweep.py --trials 2 --budget 300
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import modal


_SCRIPT = Path(__file__).resolve()
ROOT = _SCRIPT.parents[2] if len(_SCRIPT.parents) > 2 else Path("/root")
OUT = Path(__file__).resolve().parent / "a100-latency"
ARCH = "sm_80"
APP = modal.App("openassembler-a100-latency-research")
IMAGE = modal.Image.debian_slim(python_version="3.11").add_local_python_source("openassembler", copy=False)


@APP.function(image=IMAGE, gpu="A100-40GB", timeout=360, max_containers=1, retries=0)
def run_batch(jobs_blob: bytes, job_timeout: float, budget: float) -> dict:
    import os
    import tempfile

    from openassembler.gpu import batch

    folder = tempfile.mkdtemp()
    jobs_path, results_path = os.path.join(folder, "jobs.pkl"), os.path.join(folder, "results.pkl")
    with open(jobs_path, "wb") as stream:
        stream.write(jobs_blob)
    device = batch.device_info()
    t0 = time.monotonic()
    results = batch.run(jobs_path, results_path, job_timeout, budget)
    return {"device": device, "gpu_seconds": round(time.monotonic() - t0, 1), **results}


@APP.local_entrypoint()
def main(trials: int = 2, budget: float = 300, limit: int = 0,
         job_timeout: float = 10, seed: int = 77) -> None:
    sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
    import gpu_common as G
    import lat_sweep as L

    corpus_root = OUT / "corpus"
    L.ROOT = OUT
    G.ROOT = OUT
    L.build(ARCH)

    sweep_dir = f"corpus/{ARCH}/latsweep"
    jobs = G.build_jobs(ARCH, f"dir:{sweep_dir}", "", trials, seed, limit)
    print(f"{len(jobs)} cubins × {trials} trial(s), A100-40GB; budget {budget:.0f}s")
    result = run_batch.with_options(timeout=int(budget + 60)).remote(pickle.dumps(jobs), job_timeout, budget)
    result_path = corpus_root / ARCH / "gpu_a100_40gb_latsweep_corrected.json"
    status = G.compare(ARCH, f"dir:{sweep_dir}", jobs, result["results"], result["device"], result_path)
    G.print_summary(status, result_path)

    rows = json.loads(result_path.read_text())["probes"]
    per_pair = {}
    for name in L.SWEEP:
        kname = L.kname(name)
        stalls = L.stalls(name)
        tested = {stall: rows.get(f"s{stall}@{kname}", {}).get("status") == "match" for stall in stalls}
        threshold = next((stall for stall in stalls if all(tested[s] for s in stalls if s >= stall)), None)
        per_pair[name] = {"minimum_safe_stall": threshold,
                          "status_by_stall": {str(s): rows.get(f"s{s}@{kname}", {}).get("status") for s in stalls}}
    summary = {"device": result["device"], "gpu_seconds": result["gpu_seconds"],
               "jobs": len(jobs), "trials": trials, "results": per_pair}
    out_path = corpus_root / ARCH / "latency-summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    for name, row in per_pair.items():
        print(f"{name:20} minimum safe stall = {row['minimum_safe_stall']}")
    print(f"evidence: {out_path}")
