"""Generic Modal app for `openassembler run --backend modal` (and `cnr`): runs a jobs
pickle (openassembler.gpu.cuda_driver.run_plan_job format, built by
openassembler.runner) on one Modal GPU container and writes the results pickle.

    modal run openassembler/gpu/modal_app.py --jobs jobs.pkl --out res.pkl \
        [--gpu A100-40GB] [--budget 300] [--job-timeout 60]

openassembler invokes this through the `modal` CLI as a subprocess, so the modal
package only has to be installed in whatever Python env provides `modal` (it is
not a dependency of openassembler). Setup once: pip install modal && modal setup.

Billing safety (Modal bills a container from start to exit), same layers as
tools/gpu_modal.py:
  * `modal run` is an ephemeral app: it stops when this local process exits. If the
    local process is killed hard, `modal app list` / `modal app stop <id>`.
  * per-job watchdog in openassembler.gpu.batch (kills a hung kernel's worker), the
    global --budget inside the container, and Modal's own function timeout
    (hard kill) = budget + 60 s; max_containers=1, retries=0.
  * one container runs the whole batch: one cold start, then jobs back to back.
The container gets only openassembler/__init__.py + openassembler/gpu/ (stdlib-only)
and the jobs pickle; the comparison happens locally (openassembler.runner).
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import modal

GPU_DEFAULT = "A100-40GB"
BUDGET_CAP_S = 1800                 # never let one batch run longer than this, whatever --budget says
_PKG = Path(__file__).resolve().parent            # openassembler/gpu
_REMOTE = "/root/oa"                               # container: /root/oa/openassembler/{__init__.py,gpu/}

app = modal.App("openassembler-run")
image = (modal.Image.debian_slim(python_version="3.11")
         .add_local_file(str(_PKG.parent / "__init__.py"), f"{_REMOTE}/openassembler/__init__.py", copy=False)
         .add_local_dir(str(_PKG), f"{_REMOTE}/openassembler/gpu", copy=False,
                        ignore=["__pycache__", "*.pyc", "modal_app.py"]))


@app.function(image=image, gpu=GPU_DEFAULT, timeout=BUDGET_CAP_S + 60, max_containers=1, retries=0)
def run_batch(jobs_blob: bytes, job_timeout: float, budget: float) -> dict:
    import os
    import sys
    import tempfile

    sys.path.insert(0, _REMOTE)
    from openassembler.gpu import batch

    d = tempfile.mkdtemp()
    jp, rp = os.path.join(d, "jobs.pkl"), os.path.join(d, "res.pkl")
    with open(jp, "wb") as f:
        f.write(jobs_blob)
    info = batch.device_info()
    t0 = time.monotonic()
    r = batch.run(jp, rp, job_timeout, min(budget, BUDGET_CAP_S))
    return {"device": info, "gpu_seconds": round(time.monotonic() - t0, 1), **r}


@app.local_entrypoint()
def main(jobs: str, out: str, gpu: str = GPU_DEFAULT, budget: float = 300, job_timeout: float = 60) -> None:
    blob = Path(jobs).read_bytes()
    n = len(pickle.loads(blob))
    budget = min(budget, BUDGET_CAP_S)
    print(f"{n} jobs on Modal {gpu} (budget {budget:.0f}s)")
    t0 = time.time()
    fn = run_batch.with_options(gpu=gpu, timeout=int(budget) + 60)
    try:
        res = fn.remote(blob, job_timeout, budget)
    except modal.exception.FunctionTimeoutError:
        raise SystemExit("Modal hard timeout hit: nothing returned. Raise --budget or run fewer kernels.")
    dev = res["device"]
    print(f"{dev['name']} cc {dev['cc']} driver {dev['driver']}: {res['gpu_seconds']}s in container, "
          f"{time.time() - t0:.0f}s wall, {res['meta']['workers']} worker(s)")
    Path(out).write_bytes(pickle.dumps(res))
