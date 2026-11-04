"""Run probe cubins on a GPU you reach over SSH (or on this machine), then compare.

    python tools/gpu_run.py --ssh user@host [--only add] [--trials 4]
    python tools/gpu_run.py --local            # this machine has an NVIDIA GPU + driver

Same jobs, same remote code (stdlib-only openassembler.gpu), and same comparison as
tools/gpu_modal.py. Only the transport differs:
  1. build jobs locally -> jobs.pkl
  2. copy openassembler/__init__.py + openassembler/gpu/ + jobs.pkl to host:/tmp/openassembler_gpu_<pid>
  3. ssh: `timeout <budget+60> python3 -m openassembler.gpu.batch jobs.pkl res.pkl`
     (GNU timeout = hard kill on the remote side even if ssh drops)
  4. copy res.pkl back, delete the remote directory, compare locally
Remote needs: Linux, NVIDIA driver (libcuda.so.1), python3 >= 3.8. No CUDA toolkit.
"""

from __future__ import annotations

import argparse
import os
import pickle
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_common as G  # noqa: E402


def sh(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ssh", metavar="HOST")
    g.add_argument("--local", action="store_true")
    ap.add_argument("--which", default="ptxas", choices=["ptxas", "ours"])
    ap.add_argument("--arch", default="sm_80")
    ap.add_argument("--only", default="")
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--job-timeout", type=float, default=20)
    ap.add_argument("--budget", type=float, default=300)
    ap.add_argument("--tag", default="", help="output name, e.g. a100 -> gpu_a100.json")
    a = ap.parse_args()

    jobs = G.build_jobs(a.arch, a.which, a.only, a.trials, a.seed, a.limit)
    print(f"{len(jobs)} {a.which} probes x {a.trials} trials")
    tmp = Path(tempfile.mkdtemp())
    (tmp / "jobs.pkl").write_bytes(pickle.dumps(jobs))
    budget_cmd = ["timeout", str(int(a.budget + 60))]

    if a.local:
        env = dict(os.environ, PYTHONPATH=str(G.ROOT))
        subprocess.run([*budget_cmd, sys.executable, "-m", "openassembler.gpu.batch", str(tmp / "jobs.pkl"),
                        str(tmp / "res.pkl"), "--job-timeout", str(a.job_timeout), "--budget", str(a.budget)],
                       check=True, env=env, cwd=G.ROOT)
        info = subprocess.run([sys.executable, "-c", "from openassembler.gpu.batch import device_info as d; print(d())"],
                              capture_output=True, text=True, env=env, cwd=G.ROOT).stdout.strip()
    else:
        rdir = f"/tmp/openassembler_gpu_{os.getpid()}"
        q = shlex.quote
        sh(["ssh", a.ssh, f"mkdir -p {q(rdir)}/openassembler"])
        sh(["scp", "-q", "-r", str(G.ROOT / "openassembler" / "__init__.py"), str(G.ROOT / "openassembler" / "gpu"),
            f"{a.ssh}:{rdir}/openassembler/"])
        sh(["scp", "-q", str(tmp / "jobs.pkl"), f"{a.ssh}:{rdir}/"])
        try:
            sh(["ssh", a.ssh, f"cd {q(rdir)} && {' '.join(budget_cmd)} python3 -m openassembler.gpu.batch jobs.pkl res.pkl "
                f"--job-timeout {a.job_timeout} --budget {a.budget}"])
            sh(["scp", "-q", f"{a.ssh}:{rdir}/res.pkl", str(tmp / "res.pkl")])
            info = subprocess.run(["ssh", a.ssh, f"cd {q(rdir)} && python3 -c "
                                   "'from openassembler.gpu.batch import device_info as d; print(d())'"],
                                  capture_output=True, text=True).stdout.strip()
        finally:
            subprocess.run(["ssh", a.ssh, f"rm -rf {q(rdir)}"])

    res = pickle.loads((tmp / "res.pkl").read_bytes())
    tag = a.tag or (a.ssh.split("@")[-1] if a.ssh else "local")
    out = G.ROOT / "corpus" / a.arch / f"gpu_{tag}{'_ours' if a.which == 'ours' else ''}.json"
    st = G.compare(a.arch, a.which, jobs, res["results"], {"info": info}, out)
    G.print_summary(st, out)


if __name__ == "__main__":
    main()
