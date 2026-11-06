"""Paired A100 whole-model workloads through OpenAssembler's torch frontend."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import hashlib

import modal


_SCRIPT = Path(__file__).resolve()
ROOT = _SCRIPT.parents[2] if len(_SCRIPT.parents) > 2 else Path.cwd()
HERE = ROOT / "exp/whole_stack" if (ROOT / "exp/whole_stack").is_dir() else _SCRIPT.parent
APP = modal.App("openassembler-a100-whole-stack-study")
IMAGE = (modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
         .add_local_dir(str(ROOT / "openassembler"), "/root/openassembler", copy=True,
                        ignore=["**/__pycache__", "**/*.pyc"])
         .add_local_dir(str(HERE), "/root/whole_stack", copy=True,
                        ignore=["**/__pycache__", "**/*.pyc", "runs/**"]))

CASES = [
    {"name": "train_tiny", "mode": "train", "batch": 2, "seq": 32,
     "width": 64, "heads": 4, "layers": 1, "vocab": 256, "steps": 12, "warmup": 3},
    {"name": "train_base", "mode": "train", "batch": 8, "seq": 128,
     "width": 128, "heads": 4, "layers": 2, "vocab": 512, "steps": 10, "warmup": 3},
    {"name": "train_long_context", "mode": "train", "batch": 2, "seq": 512,
     "width": 256, "heads": 8, "layers": 2, "vocab": 512, "steps": 8, "warmup": 2},
    {"name": "train_wide", "mode": "train", "batch": 8, "seq": 128,
     "width": 512, "heads": 8, "layers": 2, "vocab": 512, "steps": 8, "warmup": 2},
    {"name": "train_deep", "mode": "train", "batch": 4, "seq": 128,
     "width": 256, "heads": 8, "layers": 8, "vocab": 512, "steps": 8, "warmup": 2},
    {"name": "infer_single_prefill", "mode": "infer", "batch": 1, "seq": 128,
     "width": 256, "heads": 8, "layers": 4, "vocab": 512},
    {"name": "infer_long_prefill", "mode": "infer", "batch": 8, "seq": 512,
     "width": 256, "heads": 8, "layers": 4, "vocab": 512},
    {"name": "infer_batched_prefill", "mode": "infer", "batch": 16, "seq": 256,
     "width": 512, "heads": 8, "layers": 2, "vocab": 512},
    {"name": "train_wide_bf16", "mode": "train", "batch": 8, "seq": 128,
     "width": 512, "heads": 8, "layers": 2, "vocab": 512, "dtype": "bf16"},
    {"name": "infer_batched_prefill_bf16", "mode": "infer", "batch": 16, "seq": 256,
     "width": 512, "heads": 8, "layers": 2, "vocab": 512, "dtype": "bf16"},
]


def _args(case: dict, profile: bool = False) -> list[str]:
    result = ["--mode", case["mode"]]
    for key in ("batch", "seq", "width", "heads", "layers", "vocab", "steps", "warmup", "dtype"):
        if key in case:
            result.extend(("--" + key.replace("_", "-"), str(case[key])))
    if case["mode"] == "infer":
        result.extend(("--reps", "20", "--decode-tokens", "16", "--decode-reps", "3"))
    if profile:
        result.append("--profile")
    return result


@APP.function(image=IMAGE, gpu="A100-40GB", timeout=1800, max_containers=1, retries=0)
def execute(cases: list[dict], compiled_first: bool = False) -> dict:
    import contextlib
    import io
    import os
    import tempfile
    import torch

    sys.path.insert(0, "/root")
    from openassembler.torchfront.worker import _child, compare_states, runtime_info

    torch.cuda.init()
    runtime = runtime_info()
    root = Path("/root/whole_stack")
    scratch = Path(tempfile.mkdtemp(prefix="oa-whole-stack-"))
    cache = Path(tempfile.mkdtemp(prefix="oa-whole-stack-cache-"))
    os.environ.update(TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"),
                      TRITON_CACHE_DIR=str(cache / "triton"),
                      TORCHINDUCTOR_FX_GRAPH_CACHE="1",
                      TORCHINDUCTOR_AUTOGRAD_CACHE="1",
                      TORCHINDUCTOR_COMPILE_THREADS="1",
                      PYTHONHASHSEED="0", CUDA_MODULE_LOADING="LAZY")
    total_budget = 1710.0
    start_all = time.monotonic()
    results = []
    for case_index, case in enumerate(cases):
        remaining = total_budget - (time.monotonic() - start_all)
        if remaining < 30:
            raise TimeoutError(f"whole-stack suite exhausted budget before {case['name']}")
        is_infer = case["mode"] == "infer"
        cfg = {"entrypoint": "model.py", "args": _args(case, True),
               "seed": 1729, "mode": "compile",
               "rtol": 5e-3 if is_infer else 1e-3,
               "atol": 5e-4 if is_infer else 1e-4,
               "state_rtol": 2e-2, "state_atol": 2e-3, "requirements": []}
        if case.get("dtype") == "bf16":
            cfg.update(rtol=5e-2, atol=5e-3, state_rtol=5e-2, state_atol=5e-3)
        log = io.StringIO()
        reports, states, wall_times = {}, {}, {}
        method_order = [("eager", False), ("compiled", True)]
        if compiled_first:
            method_order.reverse()
        for label, is_compiled in method_order:
            method_start = time.monotonic()
            with contextlib.redirect_stdout(log):
                reports[label], states[label] = _child(
                    root, cfg, scratch, f"case{case_index}-{label}", is_compiled,
                    min(total_budget - (time.monotonic() - start_all) - 10, 480))
            wall_times[label] = time.monotonic() - method_start
        eager, compiled = reports["eager"], reports["compiled"]
        eager_state, compiled_state = states["eager"], states["compiled"]
        eager_wall, compiled_wall = wall_times["eager"], wall_times["compiled"]
        try:
            verification = compare_states(eager_state, compiled_state, cfg["rtol"], cfg["atol"],
                                          cfg["state_rtol"], cfg["state_atol"])
        except RuntimeError as exc:
            verification = {"status": "failed", "error": str(exc), "rtol": cfg["rtol"],
                            "atol": cfg["atol"], "state_rtol": cfg["state_rtol"],
                            "state_atol": cfg["state_atol"]}
        if is_infer:
            decode_equal = eager["timing"].get("decode_final_token_ids") == compiled["timing"].get("decode_final_token_ids")
            verification["decode_token_ids_equal"] = decode_equal
            if not decode_equal:
                verification["status"] = "failed"
                verification["decode_error"] = "greedy fixed-window decoded token sequences differ"
        compiled_graphs = compiled.get("graphs", [])
        graph_summary = [{"nodes": g.get("nodes"),
                          "operations": g.get("operations", [])}
                         for g in compiled_graphs]
        for report in (eager, compiled):
            report.pop("runtime", None)
            report.pop("graphs", None)
        compiled["graph_summary"] = graph_summary
        results.append({"case": case, "verification": verification,
                        "source_sha256": hashlib.sha256((root / "model.py").read_bytes()).hexdigest(),
                        "precision": "FP32 master parameters; BF16 autocast" if case.get("dtype") == "bf16" else "FP32 parameters/activations; TF32 disabled",
                        "protocol": "same A100; fresh seeded subprocesses; resident-data step latency; initialization/compilation/final snapshots excluded",
                        "process_order": [label for label, _ in method_order],
                        "eager": {"worker_wall_s": eager_wall, "report": eager,
                                  "timing": eager.get("timing")},
                        "openassembler_cli_compiled": {
                            "worker_wall_s": compiled_wall, "report": compiled,
                            "timing": compiled.get("timing")}})
        print(json.dumps({"case": case["name"], "verification": verification,
                          "eager_s": eager.get("seconds"), "compiled_s": compiled.get("seconds")}),
              flush=True)
    return {"runtime": runtime, "elapsed_s": time.monotonic() - start_all,
            "compiler_attribution": "OpenAssembler torch frontend worker uses AOTAutograd + TorchInductor/Triton; native OpenAssembler PTX/SASS is not used for whole-model graphs; full CLI bundle/replay is not measured",
            "baseline_attribution": "eager PyTorch 2.7.1 CUDA; ATen/cuBLAS/cuDNN kernels",
            "cases": results}


@APP.local_entrypoint()
def main(out: str = "exp/whole_stack/runs/whole-stack.json", dry_run: bool = False,
         case: str = "all", rounds: int = 2):
    selected = CASES if case == "all" else [item for item in CASES if item["name"] == case]
    if not selected:
        raise SystemExit(f"unknown case {case!r}; choose all or one of {', '.join(c['name'] for c in CASES)}")
    selected = [{**item, "steps": 20, "warmup": 8} if item["mode"] == "train" else item
                for item in selected]
    if not 1 <= rounds <= 3:
        raise ValueError("rounds must be 1..3")
    if dry_run:
        print(json.dumps(selected, indent=2))
        return
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    if path.name == "whole-stack.json":
        path = path.with_name(f"whole-stack-{stamp}.json")
    result = {"schema": 2, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "cases": [], "errors": [], "status": "running",
              "source_sha256": hashlib.sha256((HERE / "model.py").read_bytes()).hexdigest()}
    path.write_text(json.dumps(result, indent=2) + "\n")
    for round_id in range(rounds):
        for item in selected:
            print(f"Starting {item['name']} round {round_id + 1}", flush=True)
            try:
                one = execute.remote([item], compiled_first=bool(round_id % 2))
                result["runtime"] = one["runtime"]
                result["compiler_attribution"] = one["compiler_attribution"]
                result["baseline_attribution"] = one["baseline_attribution"]
                for row in one["cases"]:
                    row["round"] = round_id + 1
                result["cases"].extend(one["cases"])
            except Exception as exc:
                result["errors"].append({"case": item, "round": round_id + 1, "error": str(exc)})
                print(f"Failed {item['name']}: {exc}", flush=True)
            path.write_text(json.dumps(result, indent=2) + "\n")
    result["status"] = "completed" if not result["errors"] else "completed_with_errors"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"evidence: {path}")


if __name__ == "__main__":
    main()
