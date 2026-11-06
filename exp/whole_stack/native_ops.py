"""Native scalar OA versus same-PTX ptxas and matched PyTorch CUDA operators.

The native reduction kernels deliberately retain the current repository's
one-thread-per-row implementations. This measures what exists, not an optimized
replacement. All correctness gates and tolerances are declared before timing.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import modal

_SCRIPT = Path(__file__).resolve()
ROOT = _SCRIPT.parents[2] if len(_SCRIPT.parents) > 2 else Path.cwd()
HERE = _SCRIPT.parent
BUILD = HERE / "native_ops_build"
APP = modal.App("openassembler-native-nn-operators")
IMAGE = (modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
         .add_local_python_source("openassembler", copy=False))

TOLERANCES = {
    "relu": {"rtol": 0.0, "atol": 0.0},
    "gelu": {"rtol": 2e-5, "atol": 5e-6},
    "softmax": {"rtol": 5e-5, "atol": 1e-6},
    "layernorm": {"rtol": 1e-4, "atol": 2e-5},
    "adam_step": {"rtol": 2e-5, "atol": 2e-6},
}
CASES = [
    {"op": op, "shape": [n]} for op in ("relu", "gelu", "adam_step")
    for n in (1 << 20, 1 << 22)
] + [
    {"op": op, "shape": [rows, width]} for op in ("softmax", "layernorm")
    for rows, width in ((128, 128), (1024, 512))
]


def build_payload(rebuild=False):
    source = ROOT / "examples/nn_ops.py"
    BUILD.mkdir(parents=True, exist_ok=True)
    if rebuild or not (BUILD / "manifest.json").is_file():
        command = ["python3.13", "-m", "openassembler", "compile", str(source),
                   "--out-dir", str(BUILD), "-o", "native_ops"]
        with (BUILD / "compile.log").open("w") as log:
            proc = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=240)
        if proc.returncode:
            raise RuntimeError((BUILD / "compile.log").read_text())
    manifest = json.loads((BUILD / "manifest.json").read_text())
    entry = manifest["entries"][0]
    if any(kernel["compiled_by"] != "openassembler" for kernel in entry["kernels"]):
        raise RuntimeError("native operator build unexpectedly uses a fallback")
    binary, reference = (BUILD / entry["cubin"]).read_bytes(), (BUILD / entry["reference"]).read_bytes()
    return {"oa": base64.b64encode(binary).decode(),
            "ptxas": base64.b64encode(reference).decode(),
            "oa_sha256": hashlib.sha256(binary).hexdigest(),
            "ptxas_sha256": hashlib.sha256(reference).hexdigest(),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "manifest": manifest}


@APP.function(image=IMAGE, gpu="A100-40GB", timeout=600, max_containers=1, retries=0)
def execute(payload):
    import ctypes as C
    import random
    import statistics
    import time
    import traceback
    import torch
    from torch.nn import functional as F
    from openassembler.gpu.cuda_driver import Driver

    torch.set_num_threads(2)
    torch.manual_seed(20261106)
    torch.cuda.init()
    torch.backends.cuda.matmul.allow_tf32 = False
    anchor = torch.empty(1, device="cuda")
    driver = Driver()
    modules = {method: driver.load(base64.b64decode(payload[method])) for method in ("oa", "ptxas")}
    resources = {}
    functions = {}
    for method, module in modules.items():
        for op in TOLERANCES:
            functions[method, op] = driver.function(module, op)
            resources[f"{method}/{op}"] = driver.func_attrs(functions[method, op])

    def guarded(shape):
        count = 1
        for size in shape:
            count *= size
        storage = torch.full((count + 512,), -12345.0, device="cuda", dtype=torch.float32)
        output = storage[256:-256].view(shape)
        output.fill_(float("nan"))
        return storage, output

    def numerical(actual, expected, tolerance):
        diff = actual.double() - expected
        close = torch.isclose(actual.double(), expected, **tolerance)
        finite = bool(torch.isfinite(actual).all())
        return {"passed": finite and bool(close.all()), "finite": finite,
                "elements_checked": actual.numel(), "mismatched_elements": int((~close).sum()),
                "relative_l2": float(torch.linalg.vector_norm(diff) /
                                     torch.linalg.vector_norm(expected).clamp_min(1e-30)),
                "max_abs": float(diff.abs().max())}

    started = time.monotonic()
    results = []
    for case_index, case in enumerate(CASES):
        op, shape = case["op"], case["shape"]
        count = 1
        for size in shape:
            count *= size
        name = op + "_" + "x".join(map(str, shape))
        row = {"name": name, **case, "tolerance": TOLERANCES[op], "patterns": [],
               "status": "not_run"}
        try:
            x = torch.randn(shape, device="cuda", dtype=torch.float32)
            inputs = {"x": x}
            output_names = ["out"]
            if op == "adam_step":
                inputs = {"w": x, "g": torch.randn_like(x), "m": 0.1 * torch.randn_like(x),
                          "v": 0.05 + 0.01 * torch.rand_like(x)}
                output_names = ["out", "mo", "vo"]
            normal_inputs = {key: value.clone() for key, value in inputs.items()}
            storages, outputs = {}, {}
            for method in ("oa", "ptxas", "torch_cuda"):
                storages[method], outputs[method] = {}, {}
                for out_name in output_names:
                    storage, output = guarded(shape)
                    storages[method][out_name], outputs[method][out_name] = storage, output

            def reference():
                if op == "adam_step":
                    w, g, m, v = (inputs[key].double() for key in ("w", "g", "m", "v"))
                    mn = 0.9 * m + 0.1 * g
                    vn = 0.999 * v + 0.001 * g * g
                    return {"out": w - 0.001 * mn / (torch.sqrt(vn) + 1e-8), "mo": mn, "vo": vn}
                xf = x.double()
                if op == "relu":
                    result = torch.clamp_min(xf, 0.0)
                elif op == "gelu":
                    result = 0.5 * xf * (1.0 + torch.tanh(0.7978845608 * (xf + 0.044715 * xf ** 3)))
                elif op == "softmax":
                    result = torch.softmax(xf, dim=-1)
                elif op == "layernorm":
                    mean = xf.mean(dim=-1, keepdim=True)
                    var = (xf - mean).square().mean(dim=-1, keepdim=True)
                    result = (xf - mean) * torch.rsqrt(var + 1e-5)
                return {"out": result}

            def torch_launch():
                out = outputs["torch_cuda"]
                if op == "relu":
                    torch.clamp_min(x, 0.0, out=out["out"])
                elif op == "gelu":
                    torch.ops.aten.gelu.out(x, approximate="tanh", out=out["out"])
                elif op == "softmax":
                    torch.softmax(x, dim=-1, out=out["out"])
                elif op == "layernorm":
                    # Public layer_norm allocates its returned output. Include
                    # that allocation, but do not add an artificial output copy.
                    out["out"] = F.layer_norm(x, (shape[-1],), eps=1e-5)
                else:
                    mn = 0.9 * inputs["m"] + 0.1 * inputs["g"]
                    vn = 0.999 * inputs["v"] + 0.001 * inputs["g"] * inputs["g"]
                    out["mo"], out["vo"] = mn, vn
                    out["out"] = inputs["w"] - 0.001 * mn / (torch.sqrt(vn) + 1e-8)

            def native_launcher(method):
                args = []
                if op == "adam_step":
                    tensors = [inputs[key] for key in ("w", "g", "m", "v")] + [outputs[method][key] for key in output_names]
                else:
                    tensors = [x, outputs[method]["out"]]
                args.extend(C.c_uint64(tensor.data_ptr()) for tensor in tensors)
                args.extend(C.c_int(value) for value in (shape if len(shape) == 2 else [count]))
                params = (C.c_void_p * len(args))(*[C.cast(C.pointer(arg), C.c_void_p) for arg in args])
                threads = shape[0] if len(shape) == 2 else count
                grid = ((threads + 127) // 128, 1, 1)
                fn = functions[method, op]

                def launch():
                    driver.ck(driver.cu.cuLaunchKernel(
                        fn, *grid, 128, 1, 1, 0,
                        C.c_void_p(torch.cuda.current_stream().cuda_stream), params, None), name)
                return launch

            launchers = {"oa": native_launcher("oa"), "ptxas": native_launcher("ptxas"),
                         "torch_cuda": torch_launch}
            for pattern in ("normal", "zero"):
                if pattern == "zero":
                    for key, value in inputs.items():
                        value.fill_(0.05 if key == "v" else 0.0)
                expected = reference()
                for method, launch in launchers.items():
                    for value in outputs[method].values():
                        value.fill_(float("nan"))
                    launch()
                torch.cuda.synchronize()
                checks = {}
                for method in launchers:
                    checks[method] = {out_name: numerical(outputs[method][out_name], expected[out_name], TOLERANCES[op])
                                      for out_name in output_names}
                    guarded_method = method != "torch_cuda" or op not in ("layernorm", "adam_step")
                    checks[method]["canaries"] = all(
                        bool((storage[:256] == -12345.0).all() and (storage[-256:] == -12345.0).all())
                        for storage in storages[method].values()) if guarded_method else None
                    checks[method]["guarded_outputs"] = guarded_method
                bitwise = all(torch.equal(outputs["oa"][key].view(torch.int32), outputs["ptxas"][key].view(torch.int32))
                              for key in output_names)
                row["patterns"].append({"pattern": pattern, "checks": checks,
                                        "oa_ptxas_bitwise_equal": bitwise})
            # Time the identical nonzero inputs that passed the full check.
            for key, value in inputs.items():
                value.copy_(normal_inputs[key])
            row["timing_inputs"] = "identical to normal-pattern full-output check"
            eligible = []
            for method in launchers:
                passed = all(pattern["checks"][method]["canaries"] is not False and
                             all(pattern["checks"][method][key]["passed"] for key in output_names)
                             for pattern in row["patterns"])
                if passed:
                    eligible.append(method)
            row["eligible_for_timing"] = eligible
            for method in eligible:
                for _ in range(5):
                    launchers[method]()
            torch.cuda.synchronize()
            samples = {method: [] for method in eligible}
            orders = []
            rng = random.Random(20261106 + case_index)
            for sample in range(20):
                order = list(eligible)
                rng.shuffle(order)
                orders.append(order)
                for method in order:
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record(); launchers[method](); end.record(); end.synchronize()
                    samples[method].append(start.elapsed_time(end))
            row["timings"] = {method: {"samples_ms": values, "median_ms": statistics.median(values),
                                       "mean_ms": statistics.fmean(values), "min_ms": min(values), "max_ms": max(values)}
                              for method, values in samples.items()}
            row["timing_order"] = orders
            row["status"] = "ok" if len(eligible) == 3 else "correctness_failure"
            row["torch_cuda_extra_copy"] = False
            row["torch_output_allocation"] = op in ("layernorm", "adam_step")
            row["algorithm_boundary"] = ("one thread per row, serial loops, strided lane access" if len(shape) == 2
                                         else "one thread per element; current scalar native compiler")
        except Exception as exc:
            row["status"] = "error"
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["traceback"] = traceback.format_exc()
        results.append(row)
        print(json.dumps({"case": name, "status": row["status"],
                          "timings_ms": {method: item["median_ms"] for method, item in row.get("timings", {}).items()}}), flush=True)
    for module in modules.values():
        driver.unload(module)
    return {"schema": 1, "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda,
                                      "device": torch.cuda.get_device_name(), **driver.info().__dict__},
            "elapsed_gpu_worker_seconds": time.monotonic() - started,
            "tolerances_declared_before_run": TOLERANCES,
            "reference": "FP64 PyTorch formula applied to exact FP32 input values",
            "timing": "5 warmup then 20 shuffled single-launch CUDA-event samples; no CUDA graphs; allocations/copies in PyTorch baseline included",
            "resources": resources, "cases": results}


@APP.local_entrypoint()
def main(rebuild: bool = False, dry_run: bool = False):
    if dry_run:
        print(json.dumps({"cases": CASES, "tolerances": TOLERANCES}, indent=2))
        return
    payload = build_payload(rebuild)
    result = execute.remote(payload)
    result["provenance"] = {key: value for key, value in payload.items() if key not in ("oa", "ptxas")}
    result["harness_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    directory = HERE / "runs"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"native-ops-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"evidence: {path}")
