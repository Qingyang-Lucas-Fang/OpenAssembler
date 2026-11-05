"""A100 full-output validation and graph timing for production CLI cubins."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import statistics

import modal

HERE = Path(__file__).resolve().parent
APP = modal.App("openassembler-a100-phase6-promotion-validation")
IMAGE = modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
IMAGE = IMAGE.add_local_python_source("openassembler", copy=False)


@APP.function(image=IMAGE, gpu="A100-40GB", timeout=1200, max_containers=1, retries=0)
def execute(cases):
    import ctypes as C
    import torch
    from openassembler.gpu.cuda_driver import Driver

    torch.backends.cuda.matmul.allow_tf32 = False
    results = []
    driver = Driver()
    torch.cuda.init()
    for case in cases:
        m, n, k = case["shape"]
        dtype = torch.float16 if case["dtype"] == "f16" else torch.bfloat16
        a = torch.empty((m, k), device="cuda", dtype=dtype)
        b = torch.empty((n, k), device="cuda", dtype=dtype)
        bias = torch.empty((n,), device="cuda", dtype=torch.float32)
        residual = torch.empty((m, n), device="cuda", dtype=torch.float32)
        pad = 256

        def output():
            storage = torch.full((m * n + 2 * pad,), -12345.0, device="cuda")
            return storage, storage[pad:-pad].view(m, n)

        ours_storage, ours = output()
        ref_storage, ref = output()
        temp_storage, temp = output()
        modules = []
        launches = {}
        resources = {}

        def load(binary, label, args, *, kernel="gemm", grid=None, block=128, smem=65536):
            module = driver.load(base64.b64decode(binary))
            modules.append(module)
            fn = driver.function(module, kernel)
            resources[label] = driver.func_attrs(fn)
            if smem:
                driver.ck(driver.cu.cuFuncSetAttribute(fn, 8, smem), "set opt-in shared memory")
            launch_grid = grid or (m // 128, n // 128, 1)
            pointers = [C.c_uint64(t.data_ptr()) for t in args]
            params = (C.c_void_p * len(pointers))(
                *[C.cast(C.pointer(pointer), C.c_void_p) for pointer in pointers]
            )

            def launch():
                stream = C.c_void_p(torch.cuda.current_stream().cuda_stream)
                driver.ck(driver.cu.cuLaunchKernel(
                    fn, *launch_grid, block, 1, 1, smem,
                    stream, params, None), label)

            return launch

        if case["epilogue"] == "none":
            ours_launch = load(case["ours"], "ours", [a, b, ours])
            ref_launch = load(case["reference"], "ptxas", [a, b, ref])
            control_launch = load(case["control"], "oa_m_fast_control", [a, b, temp])
        else:
            ours_launch = load(case["ours"], "ours", [a, b, ours, bias, residual])
            ref_launch = load(case["reference"], "ptxas", [a, b, ref, bias, residual])
            gemm_base = load(case["separate_oa_gemm"], "oa_base",
                             [a, b, temp])
            pointwise_grid = (m * n // 512, 1, 1)
            post = load(case["separate_oa_pointwise"], "oa_pointwise",
                        [temp, bias, residual, ref], kernel="postprocess",
                        grid=pointwise_grid, block=256, smem=0)
            def separate(gemm=gemm_base, pointwise=post):
                gemm(); pointwise()
            separate_launch = separate
            ref_gemm = load(case["separate_ptxas_gemm"], "ptxas_base",
                            [a, b, temp])
            ref_post = load(case["separate_ptxas_pointwise"], "ptxas_pointwise",
                            [temp, bias, residual, ref], kernel="postprocess",
                            grid=pointwise_grid, block=256, smem=0)
            def ptxas_separate(gemm=ref_gemm, pointwise=ref_post):
                gemm(); pointwise()

        checks = []
        for pattern, seed in (("zero", 13), ("integer", 17), ("normal", 23)):
            torch.manual_seed(seed)
            if pattern == "zero":
                a.zero_(); b.zero_(); bias.zero_(); residual.zero_()
            elif pattern == "integer":
                a.copy_(torch.randint(-2, 3, a.shape, device="cuda").to(dtype))
                b.copy_(torch.randint(-2, 3, b.shape, device="cuda").to(dtype))
                bias.copy_(torch.randint(-4, 5, bias.shape, device="cuda").float())
                residual.copy_(torch.randint(-3, 4, residual.shape, device="cuda").float())
            else:
                a.normal_(); b.normal_(); bias.normal_(); residual.normal_()
            ours_storage.fill_(-12345.0); ref_storage.fill_(-12345.0)
            ours.fill_(float("nan")); ref.fill_(float("nan"))
            ours_launch(); ref_launch(); torch.cuda.synchronize()
            if case["epilogue"] == "none":
                truth = a.float() @ b.float().T
            else:
                truth = (a.float() @ b.float().T + bias) + residual
                if case["epilogue"] == "bias_residual_relu":
                    truth = truth.clamp_min(0.0)
            err = ours - truth
            relative = float(torch.linalg.vector_norm(err) /
                             torch.linalg.vector_norm(truth).clamp_min(1e-30))
            ref_equal = torch.equal(ours.view(torch.int32), ref.view(torch.int32))
            ref_error = ours - ref
            ref_relative = float(torch.linalg.vector_norm(ref_error) /
                                 torch.linalg.vector_norm(ref).clamp_min(1e-30))
            ptxas_equal = ref_equal
            control_equal = None
            if case["epilogue"] == "none":
                temp_storage.fill_(-12345.0); temp.fill_(float("nan"))
                control_launch(); torch.cuda.synchronize()
                control_equal = torch.equal(ours.view(torch.int32), temp.view(torch.int32))
            else:
                separate_launch(); torch.cuda.synchronize()
                control_equal = torch.equal(ours.view(torch.int32), ref.view(torch.int32))
            exact = bool(torch.equal(ours, truth))
            canaries = bool((ours_storage[:pad] == -12345.0).all() and
                            (ours_storage[-pad:] == -12345.0).all() and
                            (ref_storage[:pad] == -12345.0).all() and
                            (ref_storage[-pad:] == -12345.0).all() and
                            (temp_storage[:pad] == -12345.0).all() and
                            (temp_storage[-pad:] == -12345.0).all())
            passed = canaries and bool(torch.isfinite(ours).all())
            if pattern in ("zero", "integer"):
                passed = passed and exact and ptxas_equal and control_equal
            else:
                passed = passed and relative <= 2e-5 and ref_relative <= 2e-5 and control_equal
            checks.append({"pattern": pattern, "passed": bool(passed), "relative_l2": relative,
                           "ptxas_bitwise_equal": ptxas_equal, "relative_l2_vs_ptxas": ref_relative,
                           "same_compiler_control_bitwise_equal": control_equal,
                           "integer_or_zero_exact": exact,
                           "canaries": canaries, "elements_checked": m * n})
            if not passed:
                raise AssertionError({"case": case["name"], "check": checks[-1]})

        torch.manual_seed(41); a.normal_(); b.normal_(); bias.normal_(); residual.normal_()
        launches["ours"] = ours_launch
        launches["ptxas"] = ref_launch
        if case["epilogue"] == "none":
            launches["oa_m_fast_control"] = control_launch
        else:
            launches["oa_gemm_plus_pointwise"] = separate_launch
            launches["ptxas_separate"] = ptxas_separate
        graphs = {}
        for label, launch in launches.items():
            for _ in range(3): launch()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(10): launch()
            graph.replay(); torch.cuda.synchronize(); graphs[label] = graph
        for _ in range(24):
            for graph in graphs.values(): graph.replay()
        torch.cuda.synchronize()
        samples = {label: [] for label in graphs}
        rng = random.Random(119)
        orders = []
        for _ in range(9):
            order = list(graphs); rng.shuffle(order); orders.append(order)
            for label in order:
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record(); graphs[label].replay(); end.record(); end.synchronize()
                samples[label].append(start.elapsed_time(end) / 10)
        medians = {label: statistics.median(times) for label, times in samples.items()}
        passed = all(item["passed"] for item in checks)
        results.append({"name": case["name"], "passed": passed, "device": driver.info().__dict__,
                        "shape": case["shape"], "dtype": case["dtype"], "cta_order": case["cta_order"],
                        "epilogue": case["epilogue"], "checks": checks, "resources": resources,
                        "milliseconds": medians, "samples_ms": samples, "paired_orders": orders})
        for module in modules: driver.unload(module)
    driver.close()
    return results


def prepare():
    cases = []
    configs = [
        ("tall_morton_f16", "tall_morton_f16", (16384, 4096, 4096), "f16", "morton8", "none"),
        ("tall_morton_bf16", "tall_morton_bf16", (16384, 4096, 4096), "bf16", "morton8", "none"),
        ("square_fused_linear_f16", "square_fused_linear_f16", (4096, 4096, 4096), "f16", "m_fast", "bias_residual"),
        ("square_fused_linear_bf16", "square_fused_linear_bf16", (4096, 4096, 4096), "bf16", "m_fast", "bias_residual"),
        ("square_fused_relu_f16", "square_fused_relu_f16", (4096, 4096, 4096), "f16", "m_fast", "bias_residual_relu"),
        ("square_fused_relu_bf16", "square_fused_relu_bf16", (4096, 4096, 4096), "bf16", "m_fast", "bias_residual_relu"),
    ]
    for name, directory, shape, dtype, order, epilogue in configs:
        root = HERE / directory
        manifest = json.loads((root / "manifest.json").read_text())
        entry = manifest["entries"][0]
        kernel = entry["kernels"][0]
        if kernel.get("gemm", {}).get("cta_order") != order or kernel["gemm"].get("epilogue") != epilogue:
            raise ValueError(f"unexpected compiler contract in {directory}")
        if epilogue == "none":
            baseline_root = HERE / f"tall_mfast_{dtype}"
            baseline_manifest = json.loads((baseline_root / "manifest.json").read_text())
            baseline_entry = baseline_manifest["entries"][0]
            control = (baseline_root / baseline_entry["cubin"]).read_bytes()
            separate = {}
        else:
            base_root = HERE / "controls" / f"square_{dtype}_gemm"
            post_root = HERE / "controls" / f"square_post_{epilogue}"
            control = (base_root / "openassembler.cubin").read_bytes()
            separate = {
                "separate_oa_gemm": base64.b64encode(control).decode(),
                "separate_ptxas_gemm": base64.b64encode((base_root / "ptxas.cubin").read_bytes()).decode(),
                "separate_oa_pointwise": base64.b64encode((post_root / "openassembler.cubin").read_bytes()).decode(),
                "separate_ptxas_pointwise": base64.b64encode((post_root / "ptxas.cubin").read_bytes()).decode(),
            }
        cases.append({"name": name, "shape": shape, "dtype": dtype, "cta_order": order,
                      "epilogue": epilogue,
                      "ours": base64.b64encode((root / entry["cubin"]).read_bytes()).decode(),
                      "reference": base64.b64encode((root / entry["reference"]).read_bytes()).decode(),
                      "control": base64.b64encode(control).decode(), **separate})
    return cases


@APP.local_entrypoint()
def main():
    results = execute.remote(prepare())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    target = HERE / f"validation_{stamp}.json"
    target.write_text(json.dumps(results, indent=2) + "\n")
    print(target)
    if any(not result.get("passed") for result in results):
        raise SystemExit(1)
