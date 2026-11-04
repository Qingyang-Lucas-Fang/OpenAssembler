"""One fresh process per binary pair; faults cannot poison later experiments."""
from __future__ import annotations
import base64
import ctypes as C
import glob
import json
import random
import statistics
import sys
import traceback
from pathlib import Path


def run_hmma(case):
    import struct
    from openassembler.gpu.cuda_driver import Driver, run_probe_job
    drv = Driver(); drv.open()
    rng = random.Random(41)
    bits = lambda v: struct.unpack("<I", struct.pack("<f", float(v)))[0]
    trials, expected = [], []
    for _ in range(8):
        values = [[rng.randrange(-64, 65) for _ in range(4)] for _ in range(32)]
        trials.append(([[bits(v) for v in row] for row in values], 0, 0))
        expected.append([[bits(case.get("input_factor", 1)*v + 16 * case["steps"]) for v in row[:2]] for row in values])
    try:
        result = run_probe_job(drv, {"kname": "gemm", "cubin": base64.b64decode(case["cubin"]), "trials": trials})
        mismatch = sum(x != y for got, exp in zip(result["out"], expected)
                       for grows, erows in zip(got, exp) for x, y in zip(grows, erows))
        return {"device": dataclasses_asdict(drv.info()), "encoded_gap": case["gap"],
                "steps": case["steps"], "trials": 8, "lanes": 32, "values_checked": 512,
                "mismatches": mismatch, "oob_bytes": result["oob_bytes"],
                "passed": mismatch == 0 and not any(result["oob_bytes"]),
                "attrs": result["attrs"], "actual": result["out"], "expected": expected,
                "family": case.get("family", "accumulator-chain"),
                "scope": "single warp; packed FP16 ones; integer f32 C; no clock measurement"}
    finally:
        drv.close()


def run(case):
    import torch
    from openassembler.gpu.cuda_driver import Driver
    torch.cuda.init()
    torch.set_num_threads(2)
    drv = Driver()  # Use PyTorch's current primary context, not a second context.
    m, n, k = case["shape"]
    dtype = getattr(torch, {"f16": "float16", "bf16": "bfloat16"}[case["dtype"]])
    dtype_id = 2 if case["dtype"] == "f16" else 14
    device_info = dataclasses_asdict(drv.info())
    torch.manual_seed(17)
    a = torch.randn((m, k), device="cuda", dtype=dtype)
    b = torch.randn((n, k), device="cuda", dtype=dtype)
    pad = 256
    storage = torch.full((m * n + 2 * pad,), -12345., device="cuda", dtype=torch.float32)
    out = storage[pad:-pad].view(m, n)
    ptrs = [C.c_uint64(t.data_ptr()) for t in (a, b, out)]
    params = (C.c_void_p * 3)(*[C.cast(C.pointer(p), C.c_void_p) for p in ptrs])
    grid = tuple(case["grid"])
    block = (case["threads"], 1, 1)
    modules = []
    functions, launches, resources = {}, {}, {}
    for name in ("ours", "ptxas"):
        mod = drv.load(base64.b64decode(case[name]))
        modules.append(mod)
        fn = drv.function(mod, "gemm")
        functions[name] = fn
        resources[name] = drv.func_attrs(fn)
        if case.get("smem", 0):
            drv.cu.cuFuncSetAttribute.argtypes = [C.c_void_p, C.c_int, C.c_int]
            drv.ck(drv.cu.cuFuncSetAttribute(fn, 8, case["smem"]), "opt-in dynamic shared memory")
        def launch(fn=fn):
            stream = C.c_void_p(torch.cuda.current_stream().cuda_stream)
            drv.ck(drv.cu.cuLaunchKernel(fn, *grid, *block, case.get("smem", 0), stream, params, None), "launch")
        launches[name] = launch
    for extra in case.get('compiled_variants',[]):
        mod=drv.load(base64.b64decode(extra['ours']));modules.append(mod)
        fn=drv.function(mod,'gemm');resources[extra['name']]=drv.func_attrs(fn)
        shared=extra.get('smem',0)
        if shared:
            drv.cu.cuFuncSetAttribute.argtypes=[C.c_void_p,C.c_int,C.c_int]
            drv.ck(drv.cu.cuFuncSetAttribute(fn,8,shared),'variant dynamic shared')
        def launch_extra(fn=fn,extra=extra,shared=shared):
            stream=C.c_void_p(torch.cuda.current_stream().cuda_stream)
            drv.ck(drv.cu.cuLaunchKernel(fn,*extra['grid'],extra['threads'],1,1,shared,stream,params,None),'variant launch')
        launches[extra['name']]=launch_extra

    paths = ["libcublas.so.12"] + glob.glob("/opt/conda/lib/python*/site-packages/nvidia/cublas/lib/libcublas.so*")
    blas = None
    for path in paths:
        try:
            blas = C.CDLL(path)
            break
        except OSError:
            pass
    if blas is None:
        raise RuntimeError("cuBLAS library not found")
    vp, i = C.c_void_p, C.c_int
    blas.cublasCreate_v2.argtypes = [C.POINTER(vp)]
    blas.cublasDestroy_v2.argtypes = [vp]
    blas.cublasSetStream_v2.argtypes = [vp, vp]
    blas.cublasGemmEx.argtypes = [vp, i, i, i, i, i, vp, vp, i, i, vp, i, i, vp, vp, i, i, i, i]
    handle = vp()
    def ck(status):
        if status:
            raise RuntimeError(f"cuBLAS status {status}")
    ck(blas.cublasCreate_v2(C.byref(handle)))
    one, zero = C.c_float(1), C.c_float(0)
    def cublas():
        ck(blas.cublasSetStream_v2(handle, vp(torch.cuda.current_stream().cuda_stream)))
        # C^T = B * A^T in column-major cuBLAS notation. Same f32 output and
        # f32 accumulation as both compiled kernels. No output cast in timing.
        ck(blas.cublasGemmEx(handle, 1, 0, n, m, k, C.byref(one), b.data_ptr(), dtype_id, k,
                            a.data_ptr(), dtype_id, k, C.byref(zero), out.data_ptr(), 0, n, 68, 99))
    launches["cublas"] = cublas
    artifacts = []
    if case.get("triton_sweep"):
        from triton_kernel import comparison_kernels
        extra, artifacts = comparison_kernels(a, b, out, m, n, k)
        launches.update(extra)
    checks = []
    # CPU float64 oracle evaluates exactly the quantized low-precision inputs.
    # Integer and zero cases additionally have an exactly representable answer.
    for pattern in ("zero", "integer", "normal", "scaled"):
        for seed in (17, 29):
            torch.manual_seed(seed)
            if pattern == "zero":
                a.zero_(); b.normal_()
            elif pattern == "integer":
                a.copy_(torch.randint(-3, 4, (m, k), device="cuda").to(dtype))
                b.copy_(torch.randint(-3, 4, (n, k), device="cuda").to(dtype))
            else:
                a.normal_(); b.normal_()
                if pattern == "scaled":
                    a.mul_(0.03125); b.mul_(32.)
            # Sample rows/columns for large matrices; full oracle for smoke sizes.
            full_oracle = m * n <= 65536
            nr, nc = (m, n) if full_oracle else (min(m, 32), min(n, 32))
            rows = torch.arange(nr, device="cuda") * max(1, m // nr)
            cols = torch.arange(nc, device="cuda") * max(1, n // nc)
            aa, bb = a[rows].cpu().double(), b[cols].cpu().double()
            truth = aa @ bb.T
            snapshots = {}
            full_reference = None
            for name in ["cublas"] + [name for name in launches if name != "cublas"]:
                launch = launches[name]
                storage.fill_(-12345.); out.fill_(float("nan"))
                launch(); torch.cuda.synchronize()
                if not bool((storage[:pad] == -12345).all() and (storage[-pad:] == -12345).all()):
                    raise AssertionError(f"{name} changed output canary")
                if not bool(torch.isfinite(out).all()):
                    raise AssertionError(f"{name} left nonfinite/unwritten outputs")
                if name == "cublas":
                    full_reference = out.clone()
                    full_rel = 0.0
                else:
                    full_diff = out - full_reference
                    full_rel = float(torch.linalg.vector_norm(full_diff) /
                                     torch.linalg.vector_norm(full_reference).clamp_min(1e-30))
                    if (pattern in ("zero", "integer") and not torch.equal(out, full_reference)) or full_rel > 2e-5:
                        raise AssertionError(f"{name} full-output comparison failed for {pattern}; relative_l2={full_rel}")
                sample = out[rows[:, None], cols[None, :]].cpu().double()
                snapshots[name] = sample
                error = sample - truth
                rel = float(torch.linalg.vector_norm(error) / torch.linalg.vector_norm(truth).clamp_min(1e-30))
                exact = bool(torch.equal(sample, truth))
                if pattern in ("zero", "integer") and not exact:
                    raise AssertionError(f"{name} failed exact {pattern} case: max_abs={float(error.abs().max())}; "
                                         f"actual={sample.flatten()[:16].tolist()}; expected={truth.flatten()[:16].tolist()}")
                if rel > 2e-5:
                    raise AssertionError(f"{name} relative L2 error {rel} exceeds 2e-5")
                checks.append({"pattern": pattern, "seed": seed, "method": name,
                               "sample_shape": list(truth.shape), "max_abs": float(error.abs().max()),
                               "relative_l2": rel, "exact": exact,
                               "full_output_relative_l2_vs_cublas": full_rel,
                               "full_output_elements_checked": m*n})
            if not torch.equal(snapshots["ours"], snapshots["ptxas"]):
                # Different legal accumulation orders need not be bit-identical.
                checks[-1]["ours_equals_ptxas"] = False
            else:
                checks[-1]["ours_equals_ptxas"] = True

    # Separate warm-up and graph timing, with randomized method order. Replaying
    # a graph excludes Python launch overhead while preserving kernel execution.
    graphs = {}
    nodes = case.get("graph_nodes", 20)
    for name, launch in launches.items():
        for _ in range(3):
            launch()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(nodes):
                launch()
        graph.replay(); torch.cuda.synchronize()
        graphs[name] = graph
    samples = {name: [] for name in graphs}
    rng = random.Random(23)
    for _ in range(case.get('steady_warmup_rounds',0)):
        for graph in graphs.values():graph.replay()
    torch.cuda.synchronize()
    for _ in range(7):
        order = list(graphs); rng.shuffle(order)
        for name in order:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(); graphs[name].replay(); end.record(); end.synchronize()
            samples[name].append(start.elapsed_time(end) / nodes)
    times = {name: statistics.median(v) for name, v in samples.items()}
    ck(blas.cublasDestroy_v2(handle))
    for mod in modules:
        drv.unload(mod)
    return {"device": device_info, "torch": torch.__version__, "shape_mnk": [m, n, k],
            "dtype": case["dtype"], "output_dtype": "f32", "accumulation": "f32",
            "layout": "A=MxK row-major; B=NxK row-major; C=A@B.T", "checks": checks,
            "resources": resources, "milliseconds": times, "samples_ms": samples,
            "graph_nodes": nodes, "tflops": {name: 2*m*n*k / (ms*1e9) for name, ms in times.items()},
            "artifacts": artifacts}


def dataclasses_asdict(value):
    import dataclasses
    return dataclasses.asdict(value)


if __name__ == "__main__":
    case_path, result_path = map(Path, sys.argv[1:3])
    try:
        case = json.loads(case_path.read_text())
        result = run_hmma(case) if case.get("kind") == "hmma" else run(case)
    except Exception:
        result = {"error": traceback.format_exc()}
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("checks", "samples_ms", "artifacts", "actual", "expected")}))
    raise SystemExit("error" in result)
