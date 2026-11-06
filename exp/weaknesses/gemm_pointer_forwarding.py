"""A100 experiment: forward transfer source pairs after register allocation.

All candidates use the normal CLI tensor encoder/container path. The candidate
forwards adjacent MOV/MOV/LDGSTS source pairs only after physical allocation,
updates the memory instruction's true register reads, then uses the unchanged
dependency scheduler. No issue gap, drain or synchronization rule is relaxed.

The measured control produced byte-identical streams to direct pointer homes
on all 18 tested dtype/shape cases. It is retained as a negative control, not a
second optimization or an independent physical-register layout improvement.
"""
from __future__ import annotations

import base64
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
from unittest.mock import patch

import modal

_SCRIPT = Path(__file__).resolve()
ROOT = _SCRIPT.parents[2] if len(_SCRIPT.parents) > 2 else Path.cwd()
HERE = _SCRIPT.parent
BUILD = HERE / "gemm_pointer_forward_builds"
APP = modal.App("openassembler-gemm-pointer-forwarding")
IMAGE = (modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
         .add_local_python_source("openassembler", copy=False))
SHAPES = [(128, 128, 128), (512, 512, 2048), (512, 2048, 512),
          (1024, 1024, 4096), (1024, 4096, 1024), (2048, 512, 512),
          (4096, 1024, 1024), (4096, 4096, 4096), (16384, 4096, 4096)]


def build_payload(rebuild=False, production=False):
    from openassembler.backend import tensor, ra
    from openassembler.cli import main as cli_main
    from openassembler.container.cubin import text_words

    import re
    original = ra.lower

    def forward_after_allocation(ops, phys):
        lowered = original(ops, phys)
        output = []
        for row in lowered:
            text, variable, reads, writes = row
            if text.startswith("LDGSTS.") and len(output) >= 2:
                low = re.fullmatch(r"MOV R(\d+), R(\d+)", output[-2][0])
                high = re.fullmatch(r"MOV R(\d+), R(\d+)", output[-1][0])
                address = re.search(r"\[R(\d+)\.64", text)
                if low and high and address:
                    dl,sl = map(int,low.groups())
                    dh,sh = map(int,high.groups())
                    if (dl+1 == dh and sl+1 == sh and sl%2 == 0
                            and int(address[1]) == dl and not {dl,dh}&{sl,sh}):
                        output.pop(); output.pop()
                        text = text[:address.start()] + text[address.start():].replace(f"[R{dl}.64",f"[R{sl}.64",1)
                        reads = [sl if r == dl else sh if r == dh else r for r in reads]
                        row = (text,variable,reads,writes)
            output.append(row)
        return output

    BUILD.mkdir(parents=True, exist_ok=True)
    (BUILD / ".gitignore").write_text("*\n!.gitignore\n")
    # A cached artifact is a frozen control, never evidence that today's
    # compiler generated it. Pin its full instruction/control/padding stream.
    # Production mode always rebuilds the direct candidate through the public
    # CLI and checks it against the independently hardware-validated stream.
    static = json.loads((HERE / "gemm_pointer_homes_static.json").read_text())
    frozen = {(row["case"], row["method"]): row["words_sha256"] for row in static}
    payload = {}
    for dtype in ("f16", "bf16"):
        for shape in SHAPES:
            key = "_".join(map(str, (*shape, dtype)))
            row = {"shape": list(shape), "dtype": dtype}
            for method in ("baseline", "direct_home"):
                production_direct = production and method == "direct_home"
                out = (HERE / "gemm_pointer_home_builds" / key / "baseline") if method == "baseline" else BUILD / key / method
                if rebuild or production_direct or not (out / "manifest.json").is_file():
                    context = patch.object(ra, "lower", forward_after_allocation) if method == "direct_home" and not production_direct else nullcontext()
                    with context:
                        rc = cli_main(["gemm", "--m", str(shape[0]), "--n", str(shape[1]),
                                       "--k", str(shape[2]), "--dtype", dtype,
                                       *(["--pointer-mode", "direct"] if production_direct else []),
                                       "--out-dir", str(out)])
                        if rc:
                            raise RuntimeError(f"CLI failed: {key}/{method}: {rc}")
                manifest = json.loads((out / "manifest.json").read_text())
                entry = manifest["entries"][0]
                cubin = (out / entry["cubin"]).read_bytes()
                row[method] = base64.b64encode(cubin).decode()
                row[method + "_sha256"] = hashlib.sha256(cubin).hexdigest()
                row[method + "_words_sha256"] = hashlib.sha256(b"".join(w.to_bytes(16, "little") for w in text_words(cubin, "gemm"))).hexdigest()
                if method == "baseline" and row[method + "_words_sha256"] != frozen[(key, method)]:
                    raise RuntimeError(f"frozen stream changed: {key}/{method}; restore the recorded baseline compiler before rebenchmarking")
                if method == "baseline":
                    reference = (out / entry["reference"]).read_bytes()
                    row["ptxas"] = base64.b64encode(reference).decode()
                    row["ptxas_sha256"] = hashlib.sha256(reference).hexdigest()
            payload[key] = row
    return payload


@APP.function(image=IMAGE, gpu="A100-40GB", timeout=1100, max_containers=1, retries=0)
def execute(payload, rounds=2):
    import ctypes as C
    import random
    import time
    import torch
    from openassembler.gpu.cuda_driver import Driver

    started = time.monotonic()
    torch.set_num_threads(2)
    torch.cuda.init()
    anchor = torch.empty(1, device="cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    driver = Driver()
    rng = random.Random(109)
    results = []
    for key, spec in payload.items():
        m, n, k = spec["shape"]
        dtype = torch.float16 if spec["dtype"] == "f16" else torch.bfloat16
        modules, functions = [], {}
        resources = {}
        for method in ("baseline", "direct_home", "ptxas"):
            module = driver.load(base64.b64decode(spec[method]))
            fn = driver.function(module, "gemm")
            driver.ck(driver.cu.cuFuncSetAttribute(fn, 8, 65536), "shared opt-in")
            modules.append(module)
            functions[method] = fn
            resources[method] = driver.func_attrs(fn)
        def launch(method, a, b, out):
            args = [C.c_uint64(t.data_ptr()) for t in (a, b, out)]
            params = (C.c_void_p * 3)(*[C.cast(C.pointer(arg), C.c_void_p) for arg in args])
            driver.ck(driver.cu.cuLaunchKernel(functions[method], m // 128, n // 128, 1,
                                               128, 1, 1, 65536,
                                               C.c_void_p(torch.cuda.current_stream().cuda_stream),
                                               params, None), method)
        checks = []
        torch.manual_seed(90210)
        for family in ("zero", "integer", "normal"):
            if family == "zero":
                a, b = torch.zeros((m, k), device="cuda", dtype=dtype), torch.randn((n, k), device="cuda", dtype=dtype)
            elif family == "integer":
                a = torch.randint(-2, 3, (m, k), device="cuda").to(dtype)
                b = torch.randint(-2, 3, (n, k), device="cuda").to(dtype)
            else:
                a, b = torch.randn((m, k), device="cuda", dtype=dtype), torch.randn((n, k), device="cuda", dtype=dtype)
            truth = a.float() @ b.float().T
            reference = None
            for method in ("ptxas", "baseline", "direct_home"):
                storage = torch.full((m * n + 512,), -12345.0, device="cuda")
                out = storage[256:-256].view(m, n)
                out.fill_(float("nan"))
                launch(method, a, b, out)
                torch.cuda.synchronize()
                relative = float((out-truth).norm() / truth.norm().clamp_min(1e-30))
                canary = bool((storage[:256] == -12345).all() and (storage[-256:] == -12345).all())
                if method == "ptxas":
                    reference = out.clone()
                bitwise = bool(torch.equal(out.view(torch.int32), reference.view(torch.int32)))
                passed = canary and bool(torch.isfinite(out).all()) and relative < 5e-5 and bitwise
                checks.append({"input": family, "method": method, "relative_l2": relative,
                               "same_ptx_bitwise": bitwise, "canaries": canary,
                               "elements": m*n, "passed": passed})
                if not passed:
                    raise AssertionError({"case": key, "check": checks[-1]})
        # Matched, preallocated graph timing, with output poisoning after capture
        # and full validation after replay to prove each graph performs work.
        outputs = {method: torch.empty((m, n), device="cuda") for method in functions}
        graphs = {}
        for method in functions:
            for _ in range(3):
                launch(method, a, b, outputs[method])
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(10):
                    launch(method, a, b, outputs[method])
            outputs[method].fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            if not torch.equal(outputs[method].view(torch.int32), reference.view(torch.int32)):
                raise AssertionError(f"empty or incorrect graph: {key}/{method}")
            graphs[method] = graph
        measurements = []
        for repeat in range(rounds):
            samples = {method: [] for method in functions}
            for _ in range(15):
                order = list(functions)
                rng.shuffle(order)
                for method in order:
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    graphs[method].replay()
                    end.record()
                    end.synchronize()
                    samples[method].append(start.elapsed_time(end) / 10)
            medians = {method: statistics.median(values) for method, values in samples.items()}
            measurements.append({"round": repeat, "samples_ms": samples, "median_ms": medians,
                                 "candidate_speedup": medians["baseline"] / medians["direct_home"]})
        result = {"key": key, "shape": spec["shape"], "dtype": spec["dtype"],
                  "checks": checks, "resources": resources, "timings": measurements}
        results.append(result)
        print(json.dumps({"case": key, "timings": [r["median_ms"] for r in measurements]}), flush=True)
        del graphs, outputs, a, b, truth, reference, out, storage
        torch.cuda.synchronize()
        for module in modules:
            driver.unload(module)
    return {"device": torch.cuda.get_device_name(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "driver": driver.info().__dict__,
            "elapsed_s": time.monotonic()-started, "results": results,
            "timing": "10 launches per CUDA graph; 15 shuffled paired event samples per round; replay output validated"}


@APP.local_entrypoint()
def main(mode: str = "run", subset: str = "all", rounds: int = 2):
    payload = build_payload(production=mode == "production")
    if subset != "all":
        payload = {key: row for key, row in payload.items() if subset in key}
    result = execute.remote(payload, rounds) if mode in ("run", "production") else {}
    result["candidate_generation"] = "public CLI --pointer-mode direct, rebuilt for this run" if mode == "production" else "post-allocation adjacent MOV/MOV/LDGSTS source forwarding; baseline physical layout retained"
    result["builds"] = {key: {k: v for k, v in row.items() if k not in ("baseline", "direct_home", "ptxas")}
                        for key, row in payload.items()}
    result["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    result["harness_sha256"] = hashlib.sha256(_SCRIPT.read_bytes()).hexdigest()
    output = HERE / ("gemm_pointer_forwarding-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + ".json")
    output.write_text(json.dumps(result, indent=2)+"\n")
    print(output)
