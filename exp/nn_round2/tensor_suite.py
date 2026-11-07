"""Isolated SM80 tensor-composition campaign; never imported by production.

Builds native SASS through the current tensor backend and the same-PTX ptxas
control. Experimental direct+Morton/direct+fusion combinations intentionally
use a separate experiment compiler, rather than weakening GemmPlan's admitted
production contract. HMMA issue/drain, copy visibility, address-WAR and matrix
dependencies are unchanged. A cuBLAS GemmEx control uses FP32 output and compute.

Local: python exp/nn_round2/tensor_suite.py build --profile full
Remote execution is coordinated by the campaign's parent runner, or use the
explicit ``run`` subcommand. No remote job starts at import/build time.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

_SCRIPT = Path(__file__).resolve()
ROOT = _SCRIPT.parents[2] if len(_SCRIPT.parents) > 2 else Path.cwd()
HERE = _SCRIPT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SHAPES = (
    ("smoke", (128, 128, 128), "small full-tile control"),
    ("mlp_small_forward", (512, 2048, 512), "first-layer forward"),
    ("mlp_small_dx", (512, 512, 2048), "first-layer input gradient"),
    ("mlp_small_dw", (2048, 512, 512), "first-layer weight gradient"),
    ("mlp_wide_forward", (1024, 4096, 1024), "wide-layer forward"),
    ("mlp_wide_dx", (1024, 1024, 4096), "wide-layer input gradient"),
    ("mlp_wide_dw", (4096, 1024, 1024), "wide-layer weight gradient"),
    ("square", (4096, 4096, 4096), "large compute-bound square control"),
    ("tall", (16384, 4096, 4096), "large cache/order-sensitive tall control"),
)
SASS_CRITICAL = ("HMMA", "LDSM", "LDGSTS", "LDGDEPBAR", "DEPBAR", "BAR", "STG")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def definitions(profile="full", dtypes=("f16", "bf16")):
    """Return a deterministic, explicit and bounded candidate matrix."""
    if profile not in ("smoke", "medium", "full"):
        raise ValueError("profile must be smoke, medium or full")
    if not dtypes or any(dtype not in ("f16", "bf16") for dtype in dtypes):
        raise ValueError("dtypes must be a nonempty subset of f16,bf16")
    shapes = SHAPES[:1] if profile == "smoke" else SHAPES[:7] if profile == "medium" else SHAPES
    rows = []
    for name, shape, role in shapes:
        for dtype in dtypes:
            candidates = [
                {"name": "oa_legacy", "order": "m_fast", "pointer": "legacy", "epilogue": "none"},
                {"name": "oa_direct", "order": "m_fast", "pointer": "direct", "epilogue": "none"},
            ]
            if shape[0] % 1024 == shape[1] % 1024 == 0:
                candidates += [
                    {"name": "oa_morton", "order": "morton8", "pointer": "legacy", "epilogue": "none"},
                    {"name": "exp_direct_morton", "order": "morton8", "pointer": "direct", "epilogue": "none"},
                ]
            # Forward activation/skip-connection arithmetic, plus smoke and
            # square controls. Never label dX/dW as having this epilogue.
            if "forward" in name or name in ("smoke", "square"):
                candidates += [
                    {"name": "oa_fused_relu", "order": "m_fast", "pointer": "legacy", "epilogue": "bias_residual_relu"},
                    {"name": "exp_direct_fused_relu", "order": "m_fast", "pointer": "direct", "epilogue": "bias_residual_relu"},
                ]
            rows.append({"name": name + "_" + dtype, "shape": list(shape), "dtype": dtype,
                         "role": role, "candidates": candidates})
    return rows


def source_and_config(case, candidate):
    from openassembler.backend.tensor import TensorConfig
    from openassembler.kernels.gemm import GemmPlan, generate
    # Source layout/epilogue is an admitted legacy source. Pointer selection is
    # separately explicit in this experimental build, including combinations
    # deliberately rejected by the production plan's validation.
    plan = GemmPlan(*case["shape"], case["dtype"], cta_order=candidate["order"],
                    epilogue=candidate["epilogue"])
    return generate(plan), TensorConfig(pointer_mode=candidate["pointer"]), asdict(plan)


def build_payload(profile="full", dtypes=("f16", "bf16"), build_dir=None):
    """Build immutable source/cubin/listing controls, returning a GPU payload.

    All writes are under exp/nn_round2. Cache hits require exact source,
    tensor-backend source, encoding-data and binary hashes, not a filename.
    The native code is disassembled and complete 128-bit words are matched.
    """
    from openassembler import oracle
    from openassembler.backend import tensor
    from openassembler.build import splice_kernels
    from openassembler.container.cubin import text_words
    from openassembler.ptx.parse import parse_module

    out_root = Path(build_dir) if build_dir else HERE / "tensor_builds"
    out_root = out_root.resolve()
    if not out_root.is_relative_to(HERE.resolve()):
        raise ValueError("tensor experiment outputs must remain under exp/nn_round2")
    out_root.mkdir(parents=True, exist_ok=True)
    identities = {}
    for folder in (ROOT / "openassembler", ROOT / "specs/sm_80"):
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix in (".py", ".json"):
                identities[str(path.relative_to(ROOT))] = digest(path.read_bytes())
    compiler_id = digest(json.dumps(identities, sort_keys=True).encode())
    payload = {"schema": 1, "profile": profile, "compiler_source_sha256": compiler_id,
               "source_identities": identities, "cases": [], "builds": []}
    toolkit = oracle.run_tool("ptxas", "--version").stdout.strip()
    payload["ptxas_version"] = toolkit
    for definition in definitions(profile, dtypes):
        case = {key: value for key, value in definition.items() if key != "candidates"}
        case["methods"], case["sources"] = [], {}
        for candidate in definition["candidates"]:
            source, config, plan = source_and_config(case, candidate)
            source_id = digest(source.encode())
            folder = out_root / case["name"] / candidate["name"]
            folder.mkdir(parents=True, exist_ok=True)
            source_path = folder / "kernel.ptx"
            source_path.write_text(source)
            kernel = parse_module(source).kernels["gemm"]
            code, labels, words, exits, nregs = tensor.generate(kernel, "sm_80", "opt", config=config)
            if not all(ctl.stall == 8 for text, ctl in code if text.startswith("HMMA")):
                raise AssertionError("experimental generation changed HMMA issue spacing")
            if not all(ctl.stall == 15 and ctl.yield_ for text, ctl in code if text == "NOP"):
                raise AssertionError("experimental generation changed epoch drain")
            reference_path, binary_path = folder / "ptxas.cubin", folder / "native.cubin"
            # Reuse identical source reference only within this exact build.
            if source_id not in case["sources"]:
                oracle.ptxas(source_path, reference_path, "sm_80", "-O3")
                reference = reference_path.read_bytes()
                case["sources"][source_id] = {"reference_b64": base64.b64encode(reference).decode(),
                                               "reference_sha256": digest(reference), "epilogue": candidate["epilogue"],
                                               "order": candidate["order"]}
            else:
                reference = base64.b64decode(case["sources"][source_id]["reference_b64"])
                reference_path.write_bytes(reference)
            binary = splice_kernels(reference, {"gemm": (words, nregs, exits)})
            binary_path.write_bytes(binary)
            listing = oracle.disasm_text(binary_path)
            (folder / "native.sass").write_text(listing)
            records = oracle.parse_listing(listing)
            if [record.word for record in records[:len(words)]] != words:
                raise AssertionError("nvdisasm disagrees with encoded instruction/control words")
            normalize = lambda value: re.sub(r"\s+", "", value.replace(".reuse", "").replace("+-", "-"))
            for (want, _), got in zip(code, records):
                if want.startswith(SASS_CRITICAL) and normalize(want) != normalize(got.text):
                    raise AssertionError(f"critical tensor operand mismatch: {want} -> {got.text}")
            if text_words(binary, "gemm") != words:
                raise AssertionError("spliced tensor text differs")
            counts = dict(Counter(text.split()[0] for text, _ in code))
            record = {"case": case["name"], "candidate": candidate, "source_sha256": source_id,
                      "native_sha256": digest(binary), "reference_sha256": digest(reference),
                      "words_sha256": digest(oracle.words_to_bytes(words)), "word_count": len(words),
                      "registers": nregs, "static_instruction_counts": counts,
                      "encoded_controls_validated_by_nvdisasm": True, "hmma_issue_gap": 8, "epoch_drain": 15,
                      "config": asdict(config), "source_plan": plan,
                      "experimental_combination": candidate["name"].startswith("exp_"),
                      "production_contract_unchanged": True, "compiler_source_sha256": compiler_id,
                      "path": str(folder.relative_to(ROOT))}
            (folder / "build.json").write_text(json.dumps(record, indent=2) + "\n")
            payload["builds"].append(record)
            case["methods"].append({"name": candidate["name"], "kind": "native_oa",
                                    "binary_b64": base64.b64encode(binary).decode(), "source_id": source_id,
                                    "epilogue": candidate["epilogue"], "experimental": record["experimental_combination"]})
        for number, (source_id, reference) in enumerate(case["sources"].items()):
            label = "ptxas_" + reference["order"] + ("_fused_relu" if reference["epilogue"] != "none" else "")
            case["methods"].append({"name": label, "kind": "same_ptx_ptxas", "binary_b64": reference["reference_b64"],
                                    "source_id": source_id, "epilogue": reference["epilogue"], "experimental": False})
        payload["cases"].append(case)
        print(f"tensor build {case['name']}: {len(case['methods'])} native/ptxas methods", flush=True)
    return payload


def load_payload(manifest_path):
    """Rehydrate a local build manifest, refusing drifted source/binary bytes."""
    saved = json.loads(Path(manifest_path).read_text())
    for name, expected in saved["source_identities"].items():
        if digest((ROOT / name).read_bytes()) != expected:
            raise RuntimeError(f"compiler source drifted after tensor build: {name}")
    cases = {case["name"]: {**case, "sources": {}, "methods": []} for case in saved["cases"]}
    for record in saved["builds"]:
        case = cases[record["case"]]
        candidate = record["candidate"]
        folder = (ROOT / record["path"]).resolve()
        if not folder.is_relative_to(HERE):
            raise ValueError("tensor build artifact escaped experiment workspace")
        source, native, reference = [(folder / name).read_bytes() for name in ("kernel.ptx", "native.cubin", "ptxas.cubin")]
        for data, key in ((source, "source_sha256"), (native, "native_sha256"), (reference, "reference_sha256")):
            if digest(data) != record[key]:
                raise RuntimeError(f"tensor artifact drifted: {folder}/{key}")
        source_id = record["source_sha256"]
        case["sources"][source_id] = {"reference_b64": base64.b64encode(reference).decode(),
                                      "reference_sha256": record["reference_sha256"],
                                      "epilogue": candidate["epilogue"], "order": candidate["order"]}
        case["methods"].append({"name": candidate["name"], "kind": "native_oa", "binary_b64": base64.b64encode(native).decode(),
                                "source_id": source_id, "epilogue": candidate["epilogue"],
                                "experimental": record["experimental_combination"]})
    for case in cases.values():
        for source_id, reference in case["sources"].items():
            label = "ptxas_" + reference["order"] + ("_fused_relu" if reference["epilogue"] != "none" else "")
            case["methods"].append({"name": label, "kind": "same_ptx_ptxas", "binary_b64": reference["reference_b64"],
                                    "source_id": source_id, "epilogue": reference["epilogue"], "experimental": False})
    return {**saved, "cases": list(cases.values())}


def execute(payload, rounds=2, samples=11, graph_replays=10):
    """Callable GPU worker. Full checks precede timing; failed rows stay failed."""
    import ctypes as C
    import glob
    import random
    import statistics
    import torch
    from openassembler.gpu.cuda_driver import Driver

    if rounds < 1 or samples < 3 or graph_replays < 1:
        raise ValueError("positive rounds/replays and at least three samples required")
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.cuda.init()
    context_anchor = torch.empty(1, device="cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    driver = Driver()
    if driver.info().cc != "8.0":
        raise RuntimeError("this tensor experiment requires an actual SM80 GPU")
    cublas_paths = glob.glob("/usr/local/lib/python3*/site-packages/nvidia/cublas/lib/libcublas.so*")
    cublas = C.CDLL(cublas_paths[0] if cublas_paths else "libcublas.so.12")
    handle = C.c_void_p()
    for name, types in (("cublasCreate_v2", [C.POINTER(C.c_void_p)]),
                        ("cublasDestroy_v2", [C.c_void_p]),
                        ("cublasSetStream_v2", [C.c_void_p, C.c_void_p]),
                        ("cublasSetMathMode", [C.c_void_p, C.c_int])):
        function = getattr(cublas, name)
        function.argtypes, function.restype = types, C.c_int
    def cublas_check(result, name):
        if result:
            raise RuntimeError(f"{name}: cuBLAS status {result}")
    cublas_check(cublas.cublasCreate_v2(C.byref(handle)), "cublasCreate")
    # CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION. Inputs stay FP16/BF16;
    # compute and output are explicitly FP32, with no narrowed output control.
    cublas_check(cublas.cublasSetMathMode(handle, 16), "cublasSetMathMode")
    gemm = cublas.cublasGemmEx
    gemm.argtypes, gemm.restype = [C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int,
        C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_void_p, C.c_int, C.c_int,
        C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int], C.c_int
    alpha, beta = C.c_float(1.0), C.c_float(0.0)
    results = []
    rng = random.Random(20261107)

    def validate(out, truth, storage, reference=None):
        finite = bool(torch.isfinite(out).all())
        difference = out.double() - truth
        relative = float(difference.norm() / truth.norm().clamp_min(1e-30))
        max_abs = float(difference.abs().max())
        canaries = bool((storage[:256] == -12345).all() and (storage[-256:] == -12345).all())
        per_element = bool(torch.allclose(out.double(), truth, rtol=3e-4, atol=3e-3))
        bitwise = bool(torch.equal(out.view(torch.int32), reference.view(torch.int32))) if reference is not None else None
        return {"finite": finite, "relative_l2": relative, "max_abs_error": max_abs,
                "per_element_gate": per_element, "canaries": canaries, "same_ptx_bitwise": bitwise,
                "elements": out.numel(), "passed": finite and relative < 5e-5 and canaries and per_element and bitwise is not False}

    for case in payload["cases"]:
        m, n, k = case["shape"]
        dtype = torch.float16 if case["dtype"] == "f16" else torch.bfloat16
        modules, functions, resources = [], {}, {}
        methods = {method["name"]: method for method in case["methods"]}
        epilogues = sorted({method["epilogue"] for method in methods.values()})
        for epilogue in epilogues:
            name = "cublas_gemmex" + ("_post_relu" if epilogue != "none" else "")
            methods[name] = {"name": name, "kind": "vendor_library", "source_id": None, "epilogue": epilogue}
        for name, method in methods.items():
            if method["kind"] == "vendor_library":
                continue
            module = driver.load(base64.b64decode(method["binary_b64"]))
            function = driver.function(module, "gemm")
            driver.ck(driver.cu.cuFuncSetAttribute(function, 8, 65536), "shared opt-in")
            modules.append(module)
            functions[name] = function
            resources[name] = driver.func_attrs(function)
        stores = {name: torch.full((m*n + 512,), -12345.0, device="cuda") for name in methods}
        outputs = {name: store[256:-256].view(m, n) for name, store in stores.items()}
        # Captured cuBLAS post-epilogue writes into its preallocated output, so
        # this matches a real unfused library pipeline without allocation time.
        def launch(name, a, b, bias, residual):
            method, out = methods[name], outputs[name]
            stream = C.c_void_p(torch.cuda.current_stream().cuda_stream)
            if method["kind"] == "vendor_library":
                cublas_check(cublas.cublasSetStream_v2(handle, stream), "cublasSetStream")
                input_type = 2 if dtype == torch.float16 else 14
                cublas_check(gemm(handle, 1, 0, n, m, k, C.byref(alpha), C.c_void_p(b.data_ptr()),
                                  input_type, k, C.c_void_p(a.data_ptr()), input_type, k,
                                  C.byref(beta), C.c_void_p(out.data_ptr()), 0, n, 68, 99), "cublasGemmEx")
                if method["epilogue"] != "none":
                    torch.add(out, bias, out=out)
                    torch.add(out, residual, out=out)
                    out.relu_()
                return
            arguments = [a, b, out] + ([bias, residual] if method["epilogue"] != "none" else [])
            values = [C.c_uint64(value.data_ptr()) for value in arguments]
            params = (C.c_void_p * len(values))(*[C.cast(C.pointer(value), C.c_void_p) for value in values])
            driver.ck(driver.cu.cuLaunchKernel(functions[name], m//128, n//128, 1, 128, 1, 1, 65536,
                                               stream, params, None), name)
        checks, references = [], {}
        admitted = set(methods)
        for pattern in ("zero", "integer", "normal"):
            torch.manual_seed(1729)
            if pattern == "zero":
                a = torch.zeros((m, k), device="cuda", dtype=dtype)
                b = torch.randn((n, k), device="cuda", dtype=dtype)
            elif pattern == "integer":
                a = torch.randint(-2, 3, (m, k), device="cuda").to(dtype)
                b = torch.randint(-2, 3, (n, k), device="cuda").to(dtype)
            else:
                a = torch.randn((m, k), device="cuda", dtype=dtype)
                b = torch.randn((n, k), device="cuda", dtype=dtype)
            bias = torch.randn((n,), device="cuda")
            residual = torch.randn((m, n), device="cuda")
            snapshots = [value.clone() for value in (a, b, bias, residual)]
            # Validation only: full FP64 reference is outside every timed
            # graph. Measured methods retain the identical FP32 contract.
            truth_base = a.double() @ b.double().T
            truths = {"none": truth_base, "bias_residual_relu": ((truth_base + bias.double()) + residual.double()).relu()}
            # ptxas references always execute first. Every OA method must agree
            # bitwise with the ptxas compilation of its exact source.
            order = sorted(methods, key=lambda name: methods[name]["kind"] != "same_ptx_ptxas")
            for name in order:
                method = methods[name]
                outputs[name].fill_(float("nan"))
                launch(name, a, b, bias, residual)
                torch.cuda.synchronize()
                if method["kind"] == "same_ptx_ptxas":
                    references[method["source_id"]] = outputs[name].clone()
                reference = references[method["source_id"]] if method["kind"] == "native_oa" else None
                check = {"pattern": pattern, "method": name, **validate(outputs[name], truths[method["epilogue"]], stores[name], reference)}
                check["inputs_unchanged"] = all(torch.equal(value, snapshot) for value, snapshot in zip((a, b, bias, residual), snapshots))
                check["passed"] = check["passed"] and check["inputs_unchanged"]
                checks.append(check)
                if not check["passed"]:
                    admitted.discard(name)
        graphs, graph_checks = {}, []
        for name in sorted(admitted):
            for _ in range(3):
                launch(name, a, b, bias, residual)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(graph_replays):
                    launch(name, a, b, bias, residual)
            outputs[name].fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            method = methods[name]
            reference = references[method["source_id"]] if method["kind"] == "native_oa" else None
            check = {"method": name, **validate(outputs[name], truths[method["epilogue"]], stores[name], reference)}
            check["inputs_unchanged"] = all(torch.equal(value, snapshot) for value, snapshot in zip((a, b, bias, residual), snapshots))
            check["passed"] = check["passed"] and check["inputs_unchanged"]
            graph_checks.append(check)
            if check["passed"]:
                graphs[name] = graph
        rounds_out = []
        for repeat in range(rounds):
            samples_out = {name: [] for name in graphs}
            paired = []
            for sample in range(samples):
                order = list(graphs)
                rng.shuffle(order)
                pair = {"sample": sample, "order": order, "times_ms": {}}
                for name in order:
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record(); graphs[name].replay(); end.record(); end.synchronize()
                    elapsed = start.elapsed_time(end) / graph_replays
                    samples_out[name].append(elapsed)
                    pair["times_ms"][name] = elapsed
                paired.append(pair)
            medians = {name: statistics.median(values) for name, values in samples_out.items()}
            rounds_out.append({"round": repeat, "samples_ms": samples_out, "paired_samples": paired, "median_ms": medians})
        # Tune on round zero, then independently report the same fixed selected
        # candidate on later rounds. Never report a best-of-all-rounds oracle.
        selection = {}
        if rounds_out:
            for epilogue in epilogues:
                names = [name for name in graphs if methods[name]["kind"] == "native_oa" and methods[name]["epilogue"] == epilogue]
                if names:
                    selected = min(names, key=rounds_out[0]["median_ms"].get)
                    selection[epilogue] = {"selected_on_round": 0, "method": selected,
                                           "holdout_rounds": [r["round"] for r in rounds_out[1:]],
                                           "holdout_ms": [r["median_ms"][selected] for r in rounds_out[1:]]}
        result = {"name": case["name"], "shape": case["shape"], "dtype": case["dtype"], "role": case["role"],
                  "methods": {name: {key: value for key, value in method.items() if key != "binary_b64"} for name, method in methods.items()},
                  "resources": resources, "checks": checks, "graph_checks": graph_checks,
                  "admitted_methods": sorted(graphs), "rounds": rounds_out, "experimental_selection": selection}
        results.append(result)
        print(json.dumps({"tensor_case": case["name"], "checks_passed": sum(check["passed"] for check in checks),
                          "checks": len(checks), "medians_ms": [r["median_ms"] for r in rounds_out]}), flush=True)
        del graphs, outputs, stores, a, b, bias, residual, truth_base, truths, references, snapshots
        torch.cuda.synchronize()
        for module in modules:
            driver.unload(module)
    cublas_check(cublas.cublasDestroy_v2(handle), "cublasDestroy")
    return {"schema": 1, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "device": driver.info().__dict__, "torch": torch.__version__, "cuda": torch.version.cuda,
            "elapsed_s": time.monotonic() - started, "results": results,
            "timing": {"rounds": rounds, "samples_per_round": samples, "launches_per_graph": graph_replays,
                       "ordering": "shuffled paired methods per sample", "tuning": "round zero selection; later rounds fixed holdout"},
            "contract": {"input": "f16/bf16 contiguous A[M,K],B[N,K]", "output": "fp32",
                         "accumulation": "fp32", "reference": "exact-source ptxas bitwise for every native method",
                         "numerical_truth": "complete FP64 GEMM and FP64 epilogue, outside timing; FP32-output tolerances",
                         "library": "GemmEx compute32F, FP32 output, reduced reduction disallowed; default tensor algo99",
                         "pointwise": "FP32 add bias, add residual, then relu; no arithmetic narrowing",
                         "scope": "isolated products; packing/transposes and whole-step evaluated separately"}}


def save_result(payload, result, path):
    result["builds"] = payload["builds"]
    result["compiler_source_sha256"] = payload["compiler_source_sha256"]
    result["source_identities"] = payload["source_identities"]
    result["harness_sha256"] = digest(Path(__file__).read_bytes())
    result["ptxas_version"] = payload["ptxas_version"]
    Path(path).write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("matrix", "build", "run"))
    parser.add_argument("--profile", choices=("smoke", "medium", "full"), default="full")
    parser.add_argument("--dtype", choices=("both", "f16", "bf16"), default="both")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--samples", type=int, default=11)
    args = parser.parse_args()
    dtypes = ("f16", "bf16") if args.dtype == "both" else (args.dtype,)
    if args.mode == "matrix":
        print(json.dumps(definitions(args.profile, dtypes), indent=2))
        return
    payload = build_payload(args.profile, dtypes)
    output = HERE / ("tensor_" + args.mode + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + ".json")
    if args.mode == "build":
        compact = {**payload, "cases": [{key: value for key, value in case.items() if key not in ("sources", "methods")}
                                        for case in payload["cases"]]}
        output.write_text(json.dumps(compact, indent=2) + "\n")
    else:
        import modal
        app = modal.App("openassembler-nn-round2-tensor")
        image = (modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
                 .add_local_python_source("openassembler", copy=False)
                 .add_local_file(Path(__file__), "/root/tensor_suite.py", copy=True))
        @app.function(image=image, gpu="A100-40GB", timeout=1800, max_containers=1, retries=0)
        def remote(data, rounds, samples):
            import sys
            sys.path.insert(0, "/root")
            from tensor_suite import execute as worker
            return worker(data, rounds=rounds, samples=samples)
        with app.run():
            result = remote.remote(payload, args.rounds, args.samples)
        save_result(payload, result, output)
    print(output)


if __name__ == "__main__":
    main()
