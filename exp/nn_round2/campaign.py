"""A100 campaign transport. Experiments stay outside the production compiler.

PYTHONPATH=. modal run exp/nn_round2/campaign.py --phase operators
PYTHONPATH=. modal run exp/nn_round2/campaign.py --phase tensor --payload PATH
PYTHONPATH=. modal run exp/nn_round2/campaign.py --phase whole
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import modal

_FILE = Path(__file__).resolve()
ROOT = _FILE.parents[2] if len(_FILE.parents) > 2 else Path("/root")
HERE = ROOT / "exp/nn_round2"
app = modal.App("openassembler-nn-arithmetic-round2")
image = (modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
         .add_local_dir(str(ROOT / "openassembler"), "/root/openassembler", copy=True,
                        ignore=["**/__pycache__", "**/*.pyc"])
         .add_local_file(str(HERE / "operator_worker.py"), "/root/exp/nn_round2/operator_worker.py", copy=True)
         .add_local_file(str(HERE / "tensor_suite.py"), "/root/exp/nn_round2/tensor_suite.py", copy=True)
         .add_local_file(str(HERE / "whole_worker.py"), "/root/exp/nn_round2/whole_worker.py", copy=True)
         .add_local_file(str(HERE / "counter_worker.py"), "/root/exp/nn_round2/counter_worker.py", copy=True))


@app.function(image=image, gpu="A100-40GB", timeout=1800, max_containers=1, retries=0)
def execute(phase, payload, options):
    import sys
    sys.path.insert(0, "/root")
    if phase == "operators":
        from exp.nn_round2.operator_worker import run
        return run(payload, **options)
    if phase == "tensor":
        from exp.nn_round2.tensor_suite import execute as run
        return run(payload, **options)
    if phase == "whole":
        from exp.nn_round2.whole_worker import run
        return run(payload, **options)
    if phase == "counters":
        from exp.nn_round2.counter_worker import run
        return run(payload, **options)
    raise ValueError(phase)


@app.local_entrypoint()
def main(phase: str = "operators", payload: str = "", samples: int = 11,
         rounds: int = 3, out: str = "", case: str = "all", cache_experiment: bool = False,
         training_graph_experiment: bool = False):
    sys.path.insert(0, str(ROOT))
    if phase == "operators":
        data = json.loads(Path(payload or HERE / "operator_build/payload.json").read_text())
        options = {"samples": samples, "graph_repeats": 16}
    elif phase == "tensor":
        from exp.nn_round2.tensor_suite import load_payload
        if not payload:
            raise ValueError("tensor requires a completed build manifest via --payload")
        data = load_payload(Path(payload))
        options = {"rounds": rounds, "samples": samples, "graph_replays": 5}
    elif phase == "whole":
        data = json.loads(Path(payload or HERE / "whole_build/payload.json").read_text())
        if case != "all":
            data["cases"] = [c for c in data["cases"] if c["name"] == case]
            if not data["cases"]:
                raise ValueError(case)
        options = {"samples": samples, "cache_experiment": cache_experiment,
                   "training_graph_experiment": training_graph_experiment}
    elif phase == "counters":
        data = json.loads(Path(payload).read_text())
        options = {}
    else:
        raise ValueError(phase)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = Path(out) if out else HERE / "runs" / f"{phase}-{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"schema": 1, "phase": phase, "started_utc": datetime.now(timezone.utc).isoformat(),
              "harness_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in HERE.glob("*.py")},
              "payload_sha256": hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest(),
              "options": options, "status": "running"}
    path.write_text(json.dumps(record, indent=2) + "\n")
    try:
        result = execute.remote(phase, data, options)
        record.update(status="completed", result=result)
        if phase == "tensor":
            record["builds"] = data["builds"]
            record["source_identities"] = data["source_identities"]
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        path.write_text(json.dumps(record, indent=2) + "\n")
        print(f"evidence: {path}", flush=True)
