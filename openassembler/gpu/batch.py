"""Transport-independent GPU batch runner (stdlib only).

    python -m openassembler.gpu.batch jobs.pkl results.pkl [--job-timeout 20] [--budget 600]

Runs on the GPU host. Jobs are processed in chunks by a worker *subprocess*; if
a kernel faults (poisoned context) or hangs, the worker is killed, that job is
recorded as an error, and a fresh worker resumes with the next job. A global
`--budget` (seconds) stops the whole batch early so a remote host never runs
away with billing; unfinished jobs are reported as "skipped-budget".

The same entry point is used by every transport: Modal (tools/gpu_modal.py),
SSH (tools/gpu_run.py --ssh host), or a local GPU (tools/gpu_run.py --local).
"""

from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import sys
import time


def worker(jobs_path: str, start: int, out_path: str) -> None:
    """Runs jobs[start:] in this process, appending results to out_path as it goes."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from openassembler.gpu.cuda_driver import CudaError, Driver, run_plan_job, run_probe_job, time_probe_job

    with open(jobs_path, "rb") as f:
        jobs = pickle.load(f)
    drv = Driver()
    drv.open()
    for i in range(start, len(jobs)):
        job = jobs[i]
        # progress marker: the parent knows which job was running if we die here
        with open(out_path + ".cur", "w") as f:
            f.write(str(i))
        try:
            mode = job.get("mode")
            res = (time_probe_job(drv, job) if mode == "time" else run_plan_job(drv, job) if mode == "plan"
                   else run_probe_job(drv, job))
        except CudaError as e:
            res = {"error": str(e), "error_name": e.name}
            if e.what.startswith(("kernel execution", "cuLaunchKernel")):
                # a fault poisons the context: write the result and let the parent restart us
                _append(out_path, i, res)
                os._exit(3)
        except Exception as e:  # noqa: BLE001
            # Anything else (a malformed job, cuBLAS missing, ...): record it and go on.
            # Without this the worker died with no result for job i, and the parent
            # restarted it at job i again until the budget ran out.
            res = {"error": f"{type(e).__name__}: {e}", "error_name": "HOST_EXCEPTION"}
        _append(out_path, i, res)
    drv.close()


def _append(out_path: str, i: int, res: dict) -> None:
    with open(out_path, "ab") as f:
        pickle.dump((i, res), f)


def _read(out_path: str) -> dict:
    out = {}
    if not os.path.exists(out_path):
        return out
    with open(out_path, "rb") as f:
        while True:
            try:
                i, r = pickle.load(f)
            except EOFError:
                break
            out[i] = r
    return out


def run(jobs_path: str, results_path: str, job_timeout: float, budget: float) -> dict:
    with open(jobs_path, "rb") as f:
        n = len(pickle.load(f))
    part = results_path + ".part"
    for p in (part, part + ".cur"):
        if os.path.exists(p):
            os.remove(p)
    t0 = time.monotonic()
    start, restarts = 0, 0
    while start < n:
        if time.monotonic() - t0 > budget:
            break
        proc = subprocess.Popen([sys.executable, "-m", "openassembler.gpu.batch", "--worker", jobs_path, str(start), part],
                                cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        last_done, last_t = len(_read(part)), time.monotonic()
        while proc.poll() is None:
            time.sleep(0.2)
            done = len(_read(part))
            if done != last_done:
                last_done, last_t = done, time.monotonic()
            if time.monotonic() - last_t > job_timeout or time.monotonic() - t0 > budget:
                proc.kill()
                proc.wait()
                cur = int(open(part + ".cur").read()) if os.path.exists(part + ".cur") else last_done
                if cur not in _read(part):
                    why = "timeout" if time.monotonic() - t0 <= budget else "budget"
                    _append(part, cur, {"error": f"killed: {why} (> {job_timeout}s for one job)",
                                        "error_name": f"HOST_{why.upper()}"})
                break
        res = _read(part)
        start = max(res) + 1 if res else start + 1
        restarts += 1
    res = _read(part)
    for i in range(n):
        res.setdefault(i, {"error": "not run: global budget exhausted", "error_name": "SKIPPED_BUDGET"})
    meta = {"workers": restarts, "seconds": round(time.monotonic() - t0, 1)}
    with open(results_path, "wb") as f:
        pickle.dump({"results": res, "meta": meta}, f)
    return {"results": res, "meta": meta}


def device_info() -> dict:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from openassembler.gpu.cuda_driver import Driver

    d = Driver().info()
    return d.__dict__


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(sys.argv[2], int(sys.argv[3]), sys.argv[4])
        sys.exit(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("results")
    ap.add_argument("--job-timeout", type=float, default=20)
    ap.add_argument("--budget", type=float, default=600)
    a = ap.parse_args()
    r = run(a.jobs, a.results, a.job_timeout, a.budget)
    print(f"{len(r['results'])} jobs, {r['meta']['workers']} worker(s), {r['meta']['seconds']}s")
