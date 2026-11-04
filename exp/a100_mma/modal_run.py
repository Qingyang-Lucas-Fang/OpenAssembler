"""Bounded A100 execution, with isolated worker processes and local evidence."""
from __future__ import annotations
import base64
import hashlib
import json
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
APP = modal.App("openassembler-a100-mma-compiler-exp")
IMAGE = modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
IMAGE = IMAGE.add_local_python_source("openassembler", copy=False)
IMAGE = IMAGE.add_local_file(HERE / "gpu_worker.py", "/root/exp/a100_mma/gpu_worker.py")
IMAGE = IMAGE.add_local_file(HERE / "triton_kernel.py", "/root/exp/a100_mma/triton_kernel.py")


@APP.function(image=IMAGE, gpu="A100-40GB", timeout=600, max_containers=1, retries=0)
def execute(cases):
    import os
    import subprocess
    import sys
    import time
    root = Path("/root/exp/a100_mma")
    results = []
    started = time.monotonic()
    for i, case in enumerate(cases):
        if time.monotonic() - started > 480:
            results.append({"error": "experiment budget exhausted", "name": case["name"]})
            break
        source, result = root / f"case-{i}.json", root / f"result-{i}.json"
        source.write_text(json.dumps(case))
        try:
            proc = subprocess.run([sys.executable, str(root / "gpu_worker.py"), str(source), str(result)],
                                  timeout=120, capture_output=True, text=True,
                                  env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
            record = json.loads(result.read_text()) if result.exists() else {"error": proc.stderr}
            record.update({"name": case["name"], "worker_rc": proc.returncode})
            record["input_provenance"] = case.get("input_provenance", {})
            print(case["name"], record.get("milliseconds", record.get("error", record.get("passed"))), flush=True)
        except subprocess.TimeoutExpired:
            record = {"name": case["name"], "error": "worker exceeded 120s watchdog"}
        results.append(record)
    return results


@APP.local_entrypoint()
def main(build: str = "warp64", m: int = 64, n: int = 64, k: int = 64, dtype: str = "f16",
         tile_m: int = 16, tile_n: int = 8, threads: int = 32, suite: str = "",
         triton_sweep: bool = False):
    configs = ([dict(build=build, m=m, n=n, k=k, dtype=dtype, tile_m=tile_m,
                     tile_n=tile_n, threads=threads, triton_sweep=triton_sweep)]
               if not suite else json.loads((HERE / suite).read_text()))
    cases = [build_case(config) for config in configs]
    results = execute.remote(cases)
    from datetime import datetime, timezone
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    run_directory=HERE/'runs'/stamp
    run_directory.mkdir(parents=True,exist_ok=True)
    for result in results:
        result['recorded_utc']=stamp
        for artifact in result.pop("artifacts", []):
            directory = HERE / "artifacts" / result["name"]
            directory.mkdir(parents=True, exist_ok=True)
            stem = directory / artifact["name"]
            stem.with_suffix(".ptx").write_text(artifact.pop("ptx"))
            stem.with_suffix(".cubin").write_bytes(base64.b64decode(artifact.pop("cubin")))
            stem.with_suffix(".json").write_text(json.dumps(artifact, indent=2) + "\n")
        target = HERE / "results" / f"{result['name']}.json"
        target.parent.mkdir(exist_ok=True)
        target.write_text(json.dumps([result], indent=2) + "\n")
        (run_directory/target.name).write_text(json.dumps([result],indent=2)+'\n')
        print(target)
    if any("error" in r for r in results):
        raise SystemExit(1)


def build_case(config):
    if config.get("kind") == "hmma":
        result = dict(config)
        path = (HERE / result.pop("cubin_path")).resolve()
        if not path.is_relative_to(HERE):
            raise ValueError("keep probe binaries inside the experiment")
        result["cubin"] = base64.b64encode(path.read_bytes()).decode()
        return result
    build = config["build"]
    directory = HERE / "build" / build
    if not directory.resolve().is_relative_to(HERE):
        raise ValueError("keep build paths inside the experiment")
    manifest = json.loads((directory / "manifest.json").read_text())
    entry = manifest["entries"][0]
    m, n, k = (config[key] for key in ("m", "n", "k"))
    case = {"name": config.get("name", build), "shape": [m, n, k], "dtype": config["dtype"],
            "grid": [m // config["tile_m"], n // config["tile_n"], 1], "threads": config["threads"],
            "triton_sweep": config.get("triton_sweep", False), "smem": config.get("smem", 0)}
    if m % config["tile_m"] or n % config["tile_n"]:
        raise ValueError("full-tile kernel requires divisible launch dimensions")
    provenance = {"build": build, "grid": case["grid"], "threads": case["threads"], "dynamic_shared_bytes": case["smem"]}
    for name, key in (("ours", "cubin"), ("ptxas", "reference")):
        binary = (directory / entry[key]).read_bytes()
        case[name] = base64.b64encode(binary).decode()
        provenance[f"{name}_sha256"] = hashlib.sha256(binary).hexdigest()
        archive=HERE/'binaries'/f'{provenance[f"{name}_sha256"]}.cubin'
        archive.parent.mkdir(exist_ok=True)
        if not archive.exists():archive.write_bytes(binary)
    provenance["ptx_sha256"] = hashlib.sha256((directory / entry["ptx"]).read_bytes()).hexdigest()
    if (directory / "experiment.json").exists():
        provenance["compilation"] = json.loads((directory / "experiment.json").read_text())
    case["input_provenance"] = provenance
    case['steady_warmup_rounds']=config.get('steady_warmup_rounds',0)
    if config.get('alternatives'):
        case['compiled_variants']=[build_case(extra) for extra in config['alternatives']]
        provenance['alternatives']=[x['input_provenance'] for x in case['compiled_variants']]
    return case
