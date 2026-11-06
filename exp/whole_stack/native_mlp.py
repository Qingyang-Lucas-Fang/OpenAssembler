"""Hybrid MLP whole-step A100 benchmark with native OA forward/dX/dW GEMMs.

This experiment does not change production code. All three methods share the
same FP32 LayerNorm, ReLU, MSE loss and AdamW CUDA runtime. Only GEMM SASS differs:
OpenAssembler, same-generated-PTX ptxas, or cuBLAS GemmEx. Operand conversions and
transposed packing are included in whole-step and stage measurements.
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
BUILD_ROOT = HERE / "native_builds"
APP = modal.App("openassembler-native-mlp-whole-step")
IMAGE = (modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
         .add_local_python_source("openassembler", copy=False))
CASES = [
    {"name": "mlp_512_512_2048", "batch": 512, "width": 512, "hidden": 2048},
    {"name": "mlp_1024_1024_4096", "batch": 1024, "width": 1024, "hidden": 4096},
]


def needed_shapes(cases):
    shapes = set()
    for case in cases:
        m, d, h = (case[key] for key in ("batch", "width", "hidden"))
        for rows, inputs, outputs in ((m, d, h), (m, h, d)):
            shapes.update(((rows, outputs, inputs), (rows, inputs, outputs),
                           (outputs, inputs, rows)))
    return sorted(shapes)


def build_payload(cases, rebuild=False):
    payload = {}
    python = str(ROOT / ".venv/bin/python")
    for m, n, k in needed_shapes(cases):
        key = f"{m}_{n}_{k}"
        out = BUILD_ROOT / key
        if rebuild or not (out / "manifest.json").is_file():
            command = [python, "-m", "openassembler", "gemm", "--m", str(m), "--n", str(n),
                       "--k", str(k), "--dtype", "f16", "--out-dir", str(out)]
            subprocess.run(command, cwd=ROOT, check=True)
        manifest = json.loads((out / "manifest.json").read_text())
        entry = manifest["entries"][0]
        cubin = (out / entry["cubin"]).read_bytes()
        reference = (out / entry["reference"]).read_bytes()
        payload[key] = {"shape": [m, n, k], "oa": base64.b64encode(cubin).decode(),
                        "ptxas": base64.b64encode(reference).decode(),
                        "oa_sha256": hashlib.sha256(cubin).hexdigest(),
                        "ptxas_sha256": hashlib.sha256(reference).hexdigest(),
                        "manifest_sha256": hashlib.sha256((out / "manifest.json").read_bytes()).hexdigest()}
    return payload


@APP.function(image=IMAGE, gpu="A100-40GB", timeout=600, max_containers=1, retries=0)
def execute(cases, payload):
    import ctypes as C
    from contextlib import contextmanager
    import copy
    import glob
    import io
    import math
    import random
    import statistics
    import time
    import torch
    from torch import nn
    from torch.nn import functional as F
    from openassembler.gpu.cuda_driver import Driver

    torch.set_num_threads(2)
    torch.cuda.init()
    context_anchor = torch.empty(1, device="cuda")  # Materialize primary context before driver module loads.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    driver = Driver()  # Reuse PyTorch's primary CUDA context; never Driver.open().
    modules, kernels, resources = [], {}, {}
    for key, data in payload.items():
        for method in ("oa", "ptxas"):
            module = driver.load(base64.b64decode(data[method]))
            fn = driver.function(module, "gemm")
            driver.ck(driver.cu.cuFuncSetAttribute(fn, 8, 65536), "dynamic shared opt-in")
            modules.append(module)
            kernels[(method, key)] = fn
            resources[f"{method}/{key}"] = driver.func_attrs(fn)
    paths = glob.glob("/usr/local/lib/python3*/site-packages/nvidia/cublas/lib/libcublas.so*")
    cublas = C.CDLL(paths[0] if paths else "libcublas.so.12")
    create = cublas.cublasCreate_v2
    create.argtypes, create.restype = [C.POINTER(C.c_void_p)], C.c_int
    destroy = cublas.cublasDestroy_v2
    destroy.argtypes, destroy.restype = [C.c_void_p], C.c_int
    set_stream = cublas.cublasSetStream_v2
    set_stream.argtypes, set_stream.restype = [C.c_void_p, C.c_void_p], C.c_int
    gemm = cublas.cublasGemmEx
    gemm.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int,
                    C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_void_p,
                    C.c_int, C.c_int, C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int]
    gemm.restype = C.c_int
    handle = C.c_void_p()
    if create(C.byref(handle)):
        raise RuntimeError("cublasCreate failed")
    if set_stream(handle, C.c_void_p(torch.cuda.current_stream().cuda_stream)):
        raise RuntimeError("cublasSetStream failed")
    alpha, beta = C.c_float(1.0), C.c_float(0.0)

    class Trace:
        def __init__(self):
            self.enabled = False
            self.rows = []

        @contextmanager
        def stage(self, name):
            if not self.enabled:
                yield
                return
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                yield
            finally:
                end.record()
                self.rows.append((name, start, end))

        def summary(self):
            torch.cuda.synchronize()
            grouped = {}
            for name, start, end in self.rows:
                grouped.setdefault(name, []).append(start.elapsed_time(end))
            return {name: {"samples_ms": values, "median_ms": statistics.median(values),
                           "mean_ms": statistics.fmean(values), "calls": len(values)}
                    for name, values in grouped.items()}

    class Backend:
        def __init__(self, method):
            self.method = method
            self.trace = Trace()
            self.check_guards = False
            self.guards = []
            self.gemm_checks = []

        def matmul(self, a, b, label):
            # Every mathematical product is C = A @ B.T, contiguous FP16 inputs.
            m, k = a.shape
            n, bk = b.shape
            assert k == bk and a.dtype == b.dtype == torch.float16
            assert a.is_contiguous() and b.is_contiguous()
            if self.check_guards:
                storage = torch.full((m * n + 512,), -12345.0, device="cuda", dtype=torch.float32)
                out = storage[256:-256].view(m, n)
                out.fill_(float("nan"))
                self.guards.append(storage)
            else:
                out = torch.empty((m, n), device="cuda", dtype=torch.float32)
            with self.trace.stage(label):
                if self.method == "cublas":
                    rc = gemm(handle, 1, 0, n, m, k, C.byref(alpha),
                              C.c_void_p(b.data_ptr()), 2, k, C.c_void_p(a.data_ptr()), 2, k,
                              C.byref(beta), C.c_void_p(out.data_ptr()), 0, n, 68, 99)
                    if rc:
                        raise RuntimeError(f"cublasGemmEx failed: {rc}")
                else:
                    key = f"{m}_{n}_{k}"
                    fn = kernels[(self.method, key)]
                    args = [C.c_uint64(t.data_ptr()) for t in (a, b, out)]
                    params = (C.c_void_p * 3)(*[C.cast(C.pointer(arg), C.c_void_p) for arg in args])
                    driver.ck(driver.cu.cuLaunchKernel(fn, m // 128, n // 128, 1, 128, 1, 1,
                                                       65536, C.c_void_p(torch.cuda.current_stream().cuda_stream),
                                                       params, None), label)
            if self.check_guards:
                truth = a.float() @ b.float().T
                rel = float((out - truth).norm() / truth.norm().clamp_min(1e-30))
                passed = bool(torch.isfinite(out).all()) and rel < 5e-5
                self.gemm_checks.append({"label": label, "shape": [m, n, k],
                                         "relative_l2_vs_fp32": rel, "passed": passed,
                                         "elements_checked": m * n})
                if not passed:
                    raise AssertionError(self.gemm_checks[-1])
            return out

        def validate_guards(self):
            torch.cuda.synchronize()
            passed = all(bool((s[:256] == -12345.0).all() and (s[-256:] == -12345.0).all())
                         for s in self.guards)
            if not passed:
                raise AssertionError(f"{self.method} GEMM output canary changed")
            self.guards = []
            return passed

    class Linear(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, bias, backend, name):
            with backend.trace.stage(name + "/forward_pack_cast"):
                xl, wl = x.to(torch.float16).contiguous(), weight.to(torch.float16).contiguous()
            ctx.save_for_backward(xl, wl)
            ctx.backend, ctx.name = backend, name
            out = backend.matmul(xl, wl, name + "/forward_gemm")
            with backend.trace.stage(name + "/bias"):
                out = out + bias
            return out

        @staticmethod
        def backward(ctx, grad):
            xl, wl = ctx.saved_tensors
            backend, name = ctx.backend, ctx.name
            with backend.trace.stage(name + "/backward_pack_cast_transpose"):
                gl = grad.to(torch.float16).contiguous()
                wt = wl.T.contiguous()
                gt, xt = gl.T.contiguous(), xl.T.contiguous()
            dx = backend.matmul(gl, wt, name + "/dx_gemm")
            dw = backend.matmul(gt, xt, name + "/dw_gemm")
            with backend.trace.stage(name + "/bias_gradient"):
                db = grad.sum(dim=0)
            return dx, dw, db, None, None

    class Norm(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, bias, backend):
            ctx.backend = backend
            with backend.trace.stage("layernorm_forward"):
                out, mean, rstd = torch.native_layer_norm(x, [x.shape[-1]], weight, bias, 1e-5)
            ctx.save_for_backward(x, weight, bias, mean, rstd)
            return out

        @staticmethod
        def backward(ctx, grad):
            x, weight, bias, mean, rstd = ctx.saved_tensors
            with ctx.backend.trace.stage("layernorm_backward"):
                dx, dw, db = torch.ops.aten.native_layer_norm_backward(
                    grad, x, [x.shape[-1]], mean, rstd, weight, bias, [True, True, True])
            return dx, dw, db, None

    class Relu(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, backend):
            ctx.backend = backend
            ctx.save_for_backward(x)
            with backend.trace.stage("relu_forward"):
                return x.relu()

        @staticmethod
        def backward(ctx, grad):
            with ctx.backend.trace.stage("relu_backward"):
                return grad * (ctx.saved_tensors[0] > 0), None

    class MLP(nn.Module):
        def __init__(self, case, backend):
            super().__init__()
            d, h = case["width"], case["hidden"]
            self.norm_weight = nn.Parameter(torch.ones(d, device="cuda"))
            self.norm_bias = nn.Parameter(torch.zeros(d, device="cuda"))
            self.w1 = nn.Parameter(torch.randn(h, d, device="cuda") / math.sqrt(d))
            self.b1 = nn.Parameter(torch.zeros(h, device="cuda"))
            self.w2 = nn.Parameter(torch.randn(d, h, device="cuda") / math.sqrt(h))
            self.b2 = nn.Parameter(torch.zeros(d, device="cuda"))
            self.backend = backend

        def forward(self, x):
            norm = Norm.apply(x, self.norm_weight, self.norm_bias, self.backend)
            hidden = Linear.apply(norm, self.w1, self.b1, self.backend, "fc1")
            hidden = Relu.apply(hidden, self.backend)
            out = Linear.apply(hidden, self.w2, self.b2, self.backend, "fc2")
            with self.backend.trace.stage("residual_add"):
                return x + out

    def optim(model):
        return torch.optim.AdamW(model.parameters(), lr=3e-4, eps=1e-4, foreach=True)

    def train_step(model, optimizer, x, target):
        backend = model.backend
        with backend.trace.stage("step_total"):
            optimizer.zero_grad(set_to_none=True)
            x.grad = None
            with backend.trace.stage("forward_total"):
                out = model(x)
            with backend.trace.stage("loss_forward"):
                loss = F.mse_loss(out, target)
            with backend.trace.stage("backward_total"):
                loss.backward()
            with backend.trace.stage("adamw"):
                optimizer.step()
        return out, loss

    def cpu_state(model, optimizer, x, out, loss):
        state = {"output": out.detach().cpu(), "loss": loss.detach().cpu(), "dx": x.grad.cpu()}
        for name, param in model.named_parameters():
            state["parameter/" + name] = param.detach().cpu().clone()
            state["gradient/" + name] = param.grad.detach().cpu().clone()
            for key, value in optimizer.state[param].items():
                if isinstance(value, torch.Tensor):
                    state[f"adamw/{name}/{key}"] = value.detach().cpu().clone()
        return state

    def compare_state(reference, actual, bitwise=False):
        rows = []
        for key in reference:
            ref, got = reference[key].float(), actual[key].float()
            delta = got - ref
            rel = float(delta.norm() / ref.norm().clamp_min(1e-30))
            max_abs = float(delta.abs().max())
            exact = torch.equal(reference[key].reshape(-1).view(torch.uint8),
                                actual[key].reshape(-1).view(torch.uint8))
            # Thresholds set before A100 execution. Different Tensor Core sum
            # ordering can cross a half-conversion boundary in the backward.
            threshold = 5e-3 if key.startswith(("gradient/", "dx", "adamw/")) else 5e-4
            passed = bool(torch.isfinite(got).all()) and (exact if bitwise else (rel < threshold or max_abs < 2e-6))
            rows.append({"tensor": key, "elements": ref.numel(), "relative_l2": rel,
                         "max_abs": max_abs, "bitwise_equal": exact, "passed": passed})
        return {"passed": all(row["passed"] for row in rows),
                "require_bitwise": bitwise, "tensor_checks": rows}

    def event_measure(fn):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        wall = time.perf_counter()
        start.record(); fn(); end.record(); end.synchronize()
        return start.elapsed_time(end), (time.perf_counter() - wall) * 1000

    rows = []
    isolated_gemms = []
    started = time.monotonic()
    try:
        for case in cases:
            torch.manual_seed(901)
            backend0 = Backend("cublas")
            template = MLP(case, backend0)
            initial = copy.deepcopy(template.state_dict())
            x0 = torch.randn(case["batch"], case["width"], device="cuda")
            target = torch.randn_like(x0)
            validation_states, checks = {}, {}
            for method in ("cublas", "ptxas", "oa"):
                backend = Backend(method)
                backend.check_guards = True
                model = MLP(case, backend)
                model.load_state_dict(initial)
                optimizer = optim(model)
                x = x0.detach().clone().requires_grad_(True)
                for _ in range(3):
                    out, loss = train_step(model, optimizer, x, target)
                validation_states[method] = cpu_state(model, optimizer, x, out, loss)
                checks[method] = {"canaries": backend.validate_guards(),
                                  "gemm_checks": backend.gemm_checks}
            checks["oa_vs_ptxas"] = compare_state(validation_states["ptxas"], validation_states["oa"], bitwise=True)
            checks["oa_vs_cublas"] = compare_state(validation_states["cublas"], validation_states["oa"])
            checks["ptxas_vs_cublas"] = compare_state(validation_states["cublas"], validation_states["ptxas"])
            if not checks["oa_vs_ptxas"]["passed"]:
                raise AssertionError("OA and same-PTX ptxas full states differ")
            admitted_methods = ["oa", "ptxas"]
            if checks["oa_vs_cublas"]["passed"] and checks["ptxas_vs_cublas"]["passed"]:
                admitted_methods.append("cublas")
            methods = {}
            for method in admitted_methods:
                backend = Backend(method)
                model = MLP(case, backend)
                model.load_state_dict(initial)
                methods[method] = (model, optim(model), x0.detach().clone().requires_grad_(True))
            for _ in range(5):
                for model, optimizer, x in methods.values():
                    train_step(model, optimizer, x, target)
            torch.cuda.synchronize()
            timing, wall_timing = {key: [] for key in methods}, {key: [] for key in methods}
            paired_orders = []
            rng = random.Random(927)
            for _ in range(15):
                order = list(methods)
                rng.shuffle(order)
                paired_orders.append(order)
                for method in order:
                    model, optimizer, x = methods[method]
                    gpu_ms, wall_ms = event_measure(lambda: train_step(model, optimizer, x, target))
                    timing[method].append(gpu_ms)
                    wall_timing[method].append(wall_ms)
            inference, inference_wall = {key: [] for key in methods}, {key: [] for key in methods}
            with torch.no_grad():
                inference_outputs = {}
                for method, (model, _, x) in methods.items():
                    model.load_state_dict(initial)
                    inference_outputs[method] = {"output": model(x).cpu()}
                inference_checks = {method: compare_state(inference_outputs["ptxas"], state,
                                                          bitwise=method in ("oa", "ptxas"))
                                    for method, state in inference_outputs.items()}
                if not all(check["passed"] for check in inference_checks.values()):
                    raise AssertionError({"inference_checks": inference_checks})
                for _ in range(5):
                    for model, _, x in methods.values():
                        model(x)
                for _ in range(15):
                    order = list(methods)
                    rng.shuffle(order)
                    for method in order:
                        model, _, x = methods[method]
                        gpu_ms, wall_ms = event_measure(lambda: model(x))
                        inference[method].append(gpu_ms)
                        inference_wall[method].append(wall_ms)
            stage_timings = {}
            for method, (model, optimizer, x) in methods.items():
                trace = model.backend.trace
                trace.enabled = True
                for _ in range(5):
                    train_step(model, optimizer, x, target)
                trace.enabled = False
                stage_timings[method] = trace.summary()
            rows.append({"case": case, "validation": checks, "inference_validation": inference_checks,
                         "admitted_timing_methods": admitted_methods,
                         "paired_orders": paired_orders,
                         "training_samples_ms": timing, "training_wall_samples_ms": wall_timing,
                         "training_median_ms": {k: statistics.median(v) for k, v in timing.items()},
                         "inference_samples_ms": inference, "inference_wall_samples_ms": inference_wall,
                         "inference_median_ms": {k: statistics.median(v) for k, v in inference.items()},
                         "stage_timings_diagnostic": stage_timings})
            print(json.dumps({"case": case["name"], "training_ms": rows[-1]["training_median_ms"],
                              "inference_ms": rows[-1]["inference_median_ms"]}), flush=True)
        # Separate preallocated GEMM graph timing removes allocation, packing,
        # autograd and most Python dispatch from the kernel comparison. Explicit
        # stream binding is required for cuBLAS capture. Poisoning the output
        # before replay rejects an empty capture rather than timing it.
        for key, data in payload.items():
            m, n, k = data["shape"]
            a = torch.randn((m, k), device="cuda", dtype=torch.float16)
            b = torch.randn((n, k), device="cuda", dtype=torch.float16)
            truth = a.float() @ b.float().T
            graphs, stores, micro_checks = {}, {}, {}
            for method in ("oa", "ptxas", "cublas"):
                storage = torch.full((m * n + 512,), -12345.0, device="cuda")
                out = storage[256:-256].view(m, n)
                stores[method] = storage
                args = [C.c_uint64(t.data_ptr()) for t in (a, b, out)]
                params = (C.c_void_p * 3)(*[C.cast(C.pointer(arg), C.c_void_p) for arg in args])

                def micro_launch():
                    if method == "cublas":
                        rc = gemm(handle, 1, 0, n, m, k, C.byref(alpha),
                                  C.c_void_p(b.data_ptr()), 2, k, C.c_void_p(a.data_ptr()), 2, k,
                                  C.byref(beta), C.c_void_p(out.data_ptr()), 0, n, 68, 99)
                        if rc:
                            raise RuntimeError(f"cuBLAS micro GEMM failed: {rc}")
                    else:
                        driver.ck(driver.cu.cuLaunchKernel(kernels[(method, key)], m // 128, n // 128, 1,
                                                           128, 1, 1, 65536,
                                                           C.c_void_p(torch.cuda.current_stream().cuda_stream),
                                                           params, None), "micro GEMM")

                for _ in range(5):
                    micro_launch()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    if method == "cublas":
                        if set_stream(handle, C.c_void_p(torch.cuda.current_stream().cuda_stream)):
                            raise RuntimeError("cuBLAS capture stream binding failed")
                    for _ in range(10):
                        micro_launch()
                if set_stream(handle, C.c_void_p(torch.cuda.current_stream().cuda_stream)):
                    raise RuntimeError("cuBLAS default stream restore failed")
                out.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                relative = float((out - truth).norm() / truth.norm().clamp_min(1e-30))
                passed = (bool(torch.isfinite(out).all()) and relative < 5e-5 and
                          bool((storage[:256] == -12345.0).all() and (storage[-256:] == -12345.0).all()))
                micro_checks[method] = {"passed": passed, "elements_checked": m * n,
                                        "relative_l2_vs_fp32": relative,
                                        "poisoned_output_replay": True, "canaries": passed}
                if not passed:
                    raise AssertionError({"shape": key, "method": method, "check": micro_checks[method]})
                graphs[method] = graph
            micro_samples = {method: [] for method in graphs}
            for _ in range(8):
                for graph in graphs.values():
                    graph.replay()
            torch.cuda.synchronize()
            micro_rng = random.Random(193)
            for _ in range(9):
                order = list(graphs)
                micro_rng.shuffle(order)
                for method in order:
                    milliseconds, _ = event_measure(graphs[method].replay)
                    micro_samples[method].append(milliseconds / 10)
            isolated_gemms.append({"shape": [m, n, k], "checks": micro_checks,
                                   "samples_ms": micro_samples,
                                   "median_ms": {method: statistics.median(values)
                                                 for method, values in micro_samples.items()}})
            print(json.dumps({"isolated_gemm": key, "median_ms": isolated_gemms[-1]["median_ms"]}), flush=True)
    finally:
        destroy(handle)
        for module in modules:
            driver.unload(module)
        driver.close()
    return {"schema": 1, "device": torch.cuda.get_device_name(), "driver": driver.info().__dict__,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "contract": "FP32 master params/LayerNorm/ReLU/MSE/residual/AdamW; FP16 GEMM operands, FP32 accumulation/output; straight-through operand-conversion backward implemented by explicit autograd; casts/transposed packing included",
            "scope": "Hybrid equivalent residual MLP, not a whole-native transformer. Only forward/dX/dW GEMMs differ. No CUDA graph capture.",
            "timing": "15 shuffled paired single-step CUDA-event spans after 5 warmup steps; diagnostic stage events measured separately; synchronized host wall samples retained",
            "isolated_preallocated_gemm_graphs": isolated_gemms,
            "resources": resources, "elapsed_s": time.monotonic() - started, "cases": rows}


@APP.local_entrypoint()
def main(mode: str = "run", case: str = "all"):
    cases = CASES if case == "all" else [c for c in CASES if c["name"] == case]
    if not cases:
        raise SystemExit(f"unknown case {case}")
    payload = build_payload(cases, rebuild=mode == "build")
    if mode == "build":
        print(json.dumps({k: {s: v for s, v in d.items() if s not in ("oa", "ptxas")}
                          for k, d in payload.items()}, indent=2))
        return
    result = execute.remote(cases, payload)
    result["builds"] = {k: {s: v for s, v in d.items() if s not in ("oa", "ptxas")}
                        for k, d in payload.items()}
    result["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    result["harness_sha256"] = hashlib.sha256(_SCRIPT.read_bytes()).hexdigest()
    runs = HERE / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    output = runs / ("native-mlp-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + ".json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)
