"""Run probe cubins on a Modal GPU and compare with our emulator + the PTX spec.

    modal run tools/gpu_modal.py --dry-run                  # count jobs + cost estimate, no GPU
    modal run tools/gpu_modal.py --only add.s32 --trials 2  # smoke test (~30 s of A100)
    modal run tools/gpu_modal.py                            # all ptxas probes on A100-40GB
    modal run tools/gpu_modal.py --which ours               # cubins built by OUR compiler

Billing safety (Modal bills a container from start to exit):
  * `modal run` is an ephemeral app: it stops when this local process exits.
    If the local process is killed hard, run `modal app list` and
    `modal app stop <id>` to be sure nothing is left running.
  * Three layers of timeouts: per-job watchdog in openassembler.gpu.batch (kills a
    hung kernel's worker), a global --budget inside the container, and
    Modal's function `timeout` (hard kill) = BUDGET_S + 60 s.
  * One container does the whole batch: one cold start, then jobs back to back.

The container gets only the stdlib-only `openassembler.gpu` package and the jobs
pickle; the emulator / PTX-spec comparison happens locally.
"""

from __future__ import annotations

import time

import modal

GPU_DEFAULT = "A100-40GB"          # cheapest A100 ($0.000583/s ~ $2.10/h, Modal pricing 2026-10)
BUDGET_S = 600                     # in-container hard stop for a batch
PRICE_PER_S = {"A100-40GB": 0.000583, "A100": 0.000583, "A100-80GB": 0.000694, "A10": 0.000306,
               "L4": 0.000222, "T4": 0.000164, "L40S": 0.000542, "H100": 0.001097, "H100!": 0.001097}

app = modal.App("openassembler-gpu-probes")
image = modal.Image.debian_slim(python_version="3.11").add_local_python_source("openassembler", copy=False)


@app.function(image=image, gpu=GPU_DEFAULT, timeout=BUDGET_S + 60, max_containers=1, retries=0)
def run_batch(jobs_blob: bytes, job_timeout: float, budget: float) -> dict:
    import os
    import tempfile

    from openassembler.gpu import batch

    d = tempfile.mkdtemp()
    jp, rp = os.path.join(d, "jobs.pkl"), os.path.join(d, "res.pkl")
    with open(jp, "wb") as f:
        f.write(jobs_blob)
    info = batch.device_info()
    t0 = time.monotonic()
    r = batch.run(jp, rp, job_timeout, min(budget, BUDGET_S))
    return {"device": info, "gpu_seconds": round(time.monotonic() - t0, 1), **r}


@app.local_entrypoint()
def main(gpu: str = GPU_DEFAULT, which: str = "ptxas", trials: int = 4, only: str = "", seed: int = 1,
         arch: str = "sm_80", limit: int = 0, job_timeout: float = 20, budget: float = 300,
         dry_run: bool = False, suffix: str = "", timing: bool = False, grid: int = 4096,
         reps: int = 30, iters: int = 4096, rounds: int = 3) -> None:
    import pickle
    import sys
    from pathlib import Path

    sys.path[:0] = [str(Path(__file__).resolve().parent)]
    import gpu_common as G

    jobs = G.build_jobs(arch, which, only, trials, seed, limit)
    if timing:
        # timing jobs: same cubin, k = loop trip count; correctness jobs stay in the batch
        # Several rounds, each in a different random order: a fixed order biased results
        # (two byte-identical cubins timed 0.125 vs 0.098 ms, because the GPU was still
        # ramping up during the first jobs). report_times keeps the best median per variant.
        import random as _r
        tj = []
        for rnd in range(rounds):
            order = list(jobs)
            _r.Random(seed + rnd).shuffle(order)
            for j in order:
                inputs = j["trials"][0][0]
                tj.append({**j, "name": j["name"] + f"#time{rnd}", "mode": "time", "grid": grid,
                           "reps": reps, "warmup": 20, "trials": [(inputs, iters, j["trials"][0][2])]})
        jobs = jobs + tj
    est = 15 + 0.15 * len(jobs) * max(1, trials) / 4   # cold start + ~0.15 s per probe per 4 trials
    price = PRICE_PER_S.get(gpu, 0.001)
    print(f"{len(jobs)} {which} probes x {trials} trials on {gpu}; est. {est:.0f}s GPU "
          f"= ${est * price:.3f} (budget cap {budget:.0f}s = ${budget * price:.2f})")
    if dry_run or not jobs:
        return
    t0 = time.time()
    try:
        res = run_batch.with_options(gpu=gpu).remote(pickle.dumps(jobs), job_timeout, budget)
    except modal.exception.FunctionTimeoutError:
        print("Modal hard timeout hit: nothing returned. Lower --limit or raise --budget.")
        return
    dev = res["device"]
    print(f"{dev['name']} cc {dev['cc']} driver {dev['driver']}: {res['gpu_seconds']}s in container, "
          f"{time.time() - t0:.0f}s wall, {res['meta']['workers']} worker(s) "
          f"~ ${(res['gpu_seconds'] + 10) * price:.3f}")
    tag = gpu.lower().replace("!", "").replace("-", "_")
    kind = "_ours" if which == "ours" else ("_" + Path(which[4:]).name if which.startswith("dir:") else "")
    out = G.ROOT / "corpus" / arch / f"gpu_{tag}{kind}{suffix}.json"
    if out.exists() and not suffix:
        # corpus/ holds the committed evidence of past runs; a reproduction run must not
        # overwrite it. Write next to it under a timestamp instead (diff the two to compare).
        out = out.with_name(f"{out.stem}_{time.strftime('%Y%m%d-%H%M%S')}.json")
        print(f"note: {out.name}: not overwriting the recorded run")
    if timing:
        corr = [j for j in jobs if j.get("mode") != "time"]
        idx = {i: r for i, r in res["results"].items()}
        cres = {k: idx[i] for k, i in enumerate(i for i, j in enumerate(jobs) if j.get("mode") != "time")}
        st = G.compare(arch, which, corr, cres, dev, out)
        G.print_summary(st, out)
        G.report_times(jobs, res["results"], dev, out.with_name(out.stem + "_times.json"))
        return
    st = G.compare(arch, which, jobs, res["results"], dev, out)
    G.print_summary(st, out)
