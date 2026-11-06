"""Reproduce the actual CLI capture + verified cache-only replay, in isolation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
APP = HERE / "application"
BUILD = HERE / "build"


def main():
    APP.mkdir(parents=True, exist_ok=True)
    source = HERE.parent / "model.py"
    shutil.copyfile(source, APP / "model.py")
    config = '''[application]
frontend = "torch"
entrypoint = "model.py"
args = ["--mode", "train", "--batch", "2", "--seq", "32", "--width", "64", "--heads", "4", "--layers", "1", "--vocab", "256", "--steps", "3", "--warmup", "2"]
requirements = []
seed = 1729
torch = "2.7.1"
build_backend = "modal"
gpu = "A100-40GB"
budget = 600
job_timeout = 480
verification = "state"
rtol = 0.001
atol = 0.0001
state_rtol = 0.02
state_atol = 0.002
'''
    (APP / "openassembler.toml").write_text(config)
    commands = [
        ("compile", [sys.executable, "-m", "openassembler", "compile", str(APP),
                     "-o", "proof_model", "--verify", "--no-sass", "--out-dir", str(BUILD),
                     "--build-backend", "modal", "--build-budget", "600"]),
        ("replay", [sys.executable, "-m", "openassembler", "run", str(BUILD),
                    "--application", "proof_model", "--backend", "modal",
                    "--verify-application", "--budget", "240", "--job-timeout", "180"]),
    ]
    results = {"model_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "python": sys.version, "commands": []}
    for label, command in commands:
        start = time.monotonic()
        with (HERE / (label + ".log")).open("w") as log:
            proc = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=780 if label == "compile" else 360)
        row = {"label": label, "command": command, "returncode": proc.returncode,
               "local_wall_seconds": time.monotonic() - start}
        results["commands"].append(row)
        (HERE / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(row), flush=True)
        if proc.returncode:
            raise SystemExit(proc.returncode)
    manifest = json.loads((BUILD / "manifest.json").read_text())
    app = manifest["applications"][0]
    capture = json.loads((BUILD / app["report"]).read_text())
    replay = json.loads((BUILD / "run.json").read_text())
    results["capture"] = {key: capture.get(key) for key in
                          ("runtime", "compiled_by", "captured_by", "verification", "model_instances",
                           "optimizer_steps", "forward_calls", "new_gpu_codegen", "timing", "eager_timing")}
    results["replay"] = {key: replay.get(key) for key in
                         ("runtime", "compiled", "verification", "model_instances", "optimizer_steps",
                          "forward_calls", "new_gpu_codegen", "timing", "eager_timing")}
    results["kernel_count"] = len(app["kernels"])
    results["kernel_compiled_by"] = sorted({kernel["compiled_by"] for kernel in app["kernels"]})
    results["cache_zip_bytes"] = (BUILD / app["cache"]).stat().st_size
    results["source_zip_bytes"] = (BUILD / app["source_bundle"]).stat().st_size
    results["cubin_bytes"] = sum((BUILD / kernel["cubin"]).stat().st_size for kernel in app["kernels"])
    (HERE / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"kernel_count": results["kernel_count"],
                      "compiled_by": results["kernel_compiled_by"],
                      "new_gpu_codegen": replay["new_gpu_codegen"],
                      "verification": replay["verification"]}), flush=True)


if __name__ == "__main__":
    main()
