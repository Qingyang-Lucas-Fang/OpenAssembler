"""Modal experiment: fused rank-7 matrix multiplication on an A100.

Run from the repository root:
    modal run docs/research/a100_mma_tensorcore.py \
        --shapes 4096x4096x4096,4096x4096x16384,16384x4096x4096

The experiment compares cuBLAS, a one-kernel Triton GEMM, and one-kernel fused
Strassen / AlphaTensor 2x2x2 schemes. It emits PTX and nvdisasm SASS only below
docs/research/a100-mma-artifacts/ and a JSON report next to this script.
"""

from __future__ import annotations

import json
import random
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import modal


_SCRIPT = Path(__file__).resolve()
ROOT = _SCRIPT.parents[2] if len(_SCRIPT.parents) > 2 else Path("/root")
OUT = Path(__file__).resolve().parent
APP = modal.App("openassembler-a100-mma-research")
IMAGE = modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
IMAGE = IMAGE.add_local_python_source("openassembler", copy=False)


@APP.function(image=IMAGE, gpu="A100-40GB", timeout=660, max_containers=1, retries=0)
def run_experiment(shapes: list[tuple[int, int, int]], rounds: int, reps: int,
                   seed: int, schemes: dict) -> dict:
    import torch
    import triton
    import triton.language as tl

    globals()["tl"] = tl
    globals()["triton"] = triton

    @triton.jit
    def direct_gemm(A, B, C, M: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pm = tl.program_id(0)
        pn = tl.program_id(1)
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), tl.float32)
        for k0 in range(tl.cdiv(K, BK)):
            kk = k0 * BK + rk
            a = tl.load(A + rm[:, None] * K + kk[None, :],
                        (rm[:, None] < M) & (kk[None, :] < K), 0)
            b = tl.load(B + kk[:, None] * N + rn[None, :],
                        (kk[:, None] < K) & (rn[None, :] < N), 0)
            acc = tl.dot(a, b, acc)
        tl.store(C + rm[:, None] * N + rn[None, :], acc,
                 (rm[:, None] < M) & (rn[None, :] < N))

    @triton.jit
    def sum_a(A, rm, kk, M: tl.constexpr, K: tl.constexpr,
              HM: tl.constexpr, HK: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr,
              q0, q1, q2, q3):
        x = tl.full((BM, BK), 0.0, tl.float32)
        if q0 != 0:
            x += tl.load(A + rm[:, None] * K + kk[None, :],
                         (rm[:, None] < HM) & (kk[None, :] < HK), 0).to(tl.float32) * q0
        if q1 != 0:
            x += tl.load(A + rm[:, None] * K + (kk[None, :] + HK),
                         (rm[:, None] < HM) & (kk[None, :] < HK), 0).to(tl.float32) * q1
        if q2 != 0:
            x += tl.load(A + (rm[:, None] + HM) * K + kk[None, :],
                         (rm[:, None] < HM) & (kk[None, :] < HK), 0).to(tl.float32) * q2
        if q3 != 0:
            x += tl.load(A + (rm[:, None] + HM) * K + (kk[None, :] + HK),
                         (rm[:, None] < HM) & (kk[None, :] < HK), 0).to(tl.float32) * q3
        return x

    @triton.jit
    def sum_b(B, kk, rn, K: tl.constexpr, N: tl.constexpr,
              HK: tl.constexpr, HN: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr,
              q0, q1, q2, q3):
        x = tl.full((BK, BN), 0.0, tl.float32)
        if q0 != 0:
            x += tl.load(B + kk[:, None] * N + rn[None, :],
                         (kk[:, None] < HK) & (rn[None, :] < HN), 0).to(tl.float32) * q0
        if q1 != 0:
            x += tl.load(B + kk[:, None] * N + (rn[None, :] + HN),
                         (kk[:, None] < HK) & (rn[None, :] < HN), 0).to(tl.float32) * q1
        if q2 != 0:
            x += tl.load(B + (kk[:, None] + HK) * N + rn[None, :],
                         (kk[:, None] < HK) & (rn[None, :] < HN), 0).to(tl.float32) * q2
        if q3 != 0:
            x += tl.load(B + (kk[:, None] + HK) * N + (rn[None, :] + HN),
                         (kk[:, None] < HK) & (rn[None, :] < HN), 0).to(tl.float32) * q3
        return x

    globals()["sum_a"] = sum_a
    globals()["sum_b"] = sum_b

    @triton.jit
    def bilinear_gemm(A, B, C, M: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
                      BM: tl.constexpr,
                      BN: tl.constexpr, BK: tl.constexpr,
                      U, V, W):
        """One CTA computes corresponding tiles of all four C quadrants."""
        pm = tl.program_id(0)
        pn = tl.program_id(1)
        HM: tl.constexpr = M // 2
        HK: tl.constexpr = K // 2
        HN: tl.constexpr = N // 2
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        c00 = tl.zeros((BM, BN), tl.float32)
        c01 = tl.zeros((BM, BN), tl.float32)
        c10 = tl.zeros((BM, BN), tl.float32)
        c11 = tl.zeros((BM, BN), tl.float32)
        for r in tl.static_range(7):
            u0 = tl.load(U + r)
            u1 = tl.load(U + 7 + r)
            u2 = tl.load(U + 14 + r)
            u3 = tl.load(U + 21 + r)
            v0 = tl.load(V + r)
            v1 = tl.load(V + 7 + r)
            v2 = tl.load(V + 14 + r)
            v3 = tl.load(V + 21 + r)
            product = tl.zeros((BM, BN), tl.float32)
            for k0 in range(tl.cdiv(HK, BK)):
                kk = k0 * BK + rk
                av = sum_a(A, rm, kk, M, K, HM, HK, BM, BK,
                           u0, u1, u2, u3)
                bv = sum_b(B, kk, rn, K, N, HK, HN, BK, BN,
                           v0, v1, v2, v3)
                a = av.to(A.dtype.element_ty)
                b = bv.to(B.dtype.element_ty)
                product = tl.dot(a, b, product)
            w0 = tl.load(W + r)
            w1 = tl.load(W + 7 + r)
            w2 = tl.load(W + 14 + r)
            w3 = tl.load(W + 21 + r)
            if w0 != 0:
                c00 += product * w0
            if w1 != 0:
                c01 += product * w1
            if w2 != 0:
                c10 += product * w2
            if w3 != 0:
                c11 += product * w3
        tl.store(C + rm[:, None] * N + rn[None, :], c00,
                 (rm[:, None] < HM) & (rn[None, :] < HN))
        tl.store(C + rm[:, None] * N + (rn[None, :] + HN), c01,
                 (rm[:, None] < HM) & (rn[None, :] < HN))
        tl.store(C + (rm[:, None] + HM) * N + rn[None, :], c10,
                 (rm[:, None] < HM) & (rn[None, :] < HN))
        tl.store(C + (rm[:, None] + HM) * N + (rn[None, :] + HN), c11,
                 (rm[:, None] < HM) & (rn[None, :] < HN))

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.cuda.get_device_properties(0)
    device_info = {
        "name": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "multiprocessors": device.multi_processor_count,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
    }

    def kernel_artifact(compiled):
        asm = compiled.asm
        metadata = compiled.metadata
        sass = asm.get("sass")
        if sass is None:
            cubin = asm.get("cubin")
            if not isinstance(cubin, (bytes, bytearray)):
                raise RuntimeError(f"Triton did not expose a cubin; asm keys={list(asm)}")
            tool = shutil.which("nvdisasm") or "/usr/local/cuda/bin/nvdisasm"
            with tempfile.TemporaryDirectory() as td:
                cubin_path = Path(td) / "kernel.cubin"
                cubin_path.write_bytes(cubin)
                sass = subprocess.run([tool, "-hex", "-c", str(cubin_path)],
                                      capture_output=True, text=True, check=True).stdout
        ptx = asm.get("ptx", "")
        opcodes = {}
        in_kernel = False
        for line in sass.splitlines():
            if re.search(r"\.section\s+\.text\.", line):
                in_kernel = True
                continue
            if not in_kernel:
                continue
            match = re.search(r"\*/\s+(?:@!?[A-Z0-9]+\s+)?([A-Z][A-Z0-9_.]*)\b", line)
            if match:
                op = match.group(1).split(".")[0]
                opcodes[op] = opcodes.get(op, 0) + 1
                if op == "EXIT":
                    break
        reg_match = re.search(r"SHI_REGISTERS=(\d+)", sass)
        return {
            "n_regs": int(reg_match.group(1)) if reg_match else
                      getattr(compiled, "n_regs", getattr(metadata, "num_regs", None)),
            "n_spills": getattr(compiled, "n_spills", getattr(metadata, "spill_bytes", None)),
            "shared_bytes": getattr(metadata, "shared", None),
            "local_memory_ops": opcodes.get("LDL", 0) + opcodes.get("STL", 0),
            "ptx": ptx,
            "sass": sass,
            "opcode_counts": opcodes,
            "sass_instructions": sum(opcodes.values()),
        }

    def capture_graph(fn):
        fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        graph.replay()
        torch.cuda.synchronize()
        return graph

    def measure_graph(graph):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(reps):
            graph.replay()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / reps

    results = []
    artifacts = {}
    for m, k, n in shapes:
        if m % 64 or k % 64 or n % 128:
            raise ValueError("M/K/N must be divisible by 64/64/128 for the current tile configuration")
        for dtype_name, dtype in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
            a = torch.randn((m, k), device="cuda", dtype=dtype)
            b = torch.randn((k, n), device="cuda", dtype=dtype)
            ref = torch.mm(a, b).float()
            outputs = {}
            cases = []

            cublas_out = None
            def cublas_call():
                nonlocal cublas_out
                cublas_out = torch.mm(a, b)
            graph = capture_graph(cublas_call)
            outputs["cublas"] = cublas_out.float()
            cases.append({"algorithm": "cublas", "times_ms": [], "graph": graph})

            BM, BN, BK = 32, 64, 32
            direct_out = torch.empty((m, n), device="cuda", dtype=dtype)
            direct_grid = (triton.cdiv(m, BM), triton.cdiv(n, BN))
            compiled = direct_gemm.warmup(a, b, direct_out, m, k, n, BM, BN, BK,
                                          grid=direct_grid, num_warps=4, num_stages=3)
            direct_meta = kernel_artifact(compiled)
            def direct_call():
                direct_gemm[direct_grid](a, b, direct_out, m, k, n, BM, BN, BK,
                                         num_warps=4, num_stages=3)
            graph = capture_graph(direct_call)
            outputs["triton_direct"] = direct_out
            cases.append({"algorithm": "triton_direct", "times_ms": [], "graph": graph})
            shape_key = f"m{m}_k{k}_n{n}"
            direct_key = f"{shape_key}_{dtype_name}_direct"
            artifacts[direct_key] = direct_meta
            cases[-1].update({"sass_artifact": direct_key,
                              "n_regs": direct_meta["n_regs"],
                              "n_spills": direct_meta["n_spills"],
                              "shared_bytes": direct_meta["shared_bytes"],
                              "sass_instructions": direct_meta["sass_instructions"],
                              "opcode_counts": direct_meta["opcode_counts"]})

            for name, scheme in schemes.items():
                u, v, w = scheme["u"], scheme["v"], scheme["w"]
                u_dev = torch.tensor([x for row in u for x in row], device="cuda", dtype=torch.int32)
                v_dev = torch.tensor([x for row in v for x in row], device="cuda", dtype=torch.int32)
                w_dev = torch.tensor([x for row in w for x in row], device="cuda", dtype=torch.int32)
                out = torch.empty((m, n), device="cuda", dtype=dtype)
                grid = (triton.cdiv(m // 2, BM), triton.cdiv(n // 2, BN))
                compiled = bilinear_gemm.warmup(a, b, out, m, k, n, BM, BN, BK,
                                                u_dev, v_dev, w_dev,
                                                grid=grid, num_warps=4, num_stages=3)
                kernel_meta = kernel_artifact(compiled)
                def alpha_call(out=out, u_dev=u_dev, v_dev=v_dev, w_dev=w_dev, grid=grid):
                    bilinear_gemm[grid](a, b, out, m, k, n, BM, BN, BK,
                                        u_dev, v_dev, w_dev,
                                        num_warps=4, num_stages=3)
                graph = capture_graph(alpha_call)
                outputs[name] = out
                cases.append({"algorithm": name, "times_ms": [], "graph": graph,
                              "rank": scheme["rank"], "additions": scheme["additions"],
                              "source": scheme["source"]})
                artifact_key = f"{shape_key}_{dtype_name}_{name}"
                artifacts[artifact_key] = kernel_meta
                cases[-1].update({"sass_artifact": artifact_key,
                                  "n_regs": kernel_meta["n_regs"],
                                  "n_spills": kernel_meta["n_spills"],
                                  "shared_bytes": kernel_meta["shared_bytes"],
                                  "sass_instructions": kernel_meta["sass_instructions"],
                                  "opcode_counts": kernel_meta["opcode_counts"]})

            for round_id in range(rounds):
                order = list(cases)
                random.Random(seed + 1009 * m + 101 * k + n + round_id).shuffle(order)
                for case in order:
                    case["times_ms"].append(measure_graph(case["graph"]))
            torch.cuda.synchronize()
            for case in cases:
                case["graph"].replay()
            torch.cuda.synchronize()
            for case in cases:
                value = outputs[case["algorithm"]]
                delta = value - ref
                case["relative_l2_error"] = (torch.linalg.vector_norm(delta) /
                                               torch.linalg.vector_norm(ref)).item()
                case["max_abs_error"] = delta.abs().max().item()
                del case["graph"]
            results.append({"shape_mkn": [m, k, n], "dtype": dtype_name,
                            "tile": [BM, BN, BK], "cases": cases})
            del a, b, ref, outputs, cases, direct_out, cublas_out
            torch.cuda.empty_cache()

    return {"device": device_info, "seed": seed, "rounds": rounds, "reps": reps,
            "results": results, "artifacts": artifacts}


@APP.local_entrypoint()
def main(shapes: str = "4096x4096x4096,4096x4096x16384,16384x4096x4096",
         rounds: int = 3, reps: int = 20, seed: int = 7,
         budget: int = 600) -> None:
    import sys

    sys.path.insert(0, str(ROOT))
    from openassembler.fastmm.factors import all_schemes, verify

    shape_list = [tuple(map(int, x.split("x"))) for x in shapes.split(",") if x]
    if not shape_list or any(len(x) != 3 for x in shape_list) or not 1 <= rounds <= 10 or reps < 1:
        raise SystemExit("expected comma-separated MxKxN shapes, rounds 1..10, and reps >= 1")
    schemes = {}
    source_schemes = all_schemes()
    for name in ("strassen_2x2x2", "at_2x2x2"):
        s = source_schemes[name]
        if not verify(s) or s.rank != 7:
            raise SystemExit(f"exact rational verification failed for {name}")
        coeffs = []
        for matrix in (s.u, s.v, s.w):
            rows = []
            for row in matrix:
                values = []
                for value in row:
                    if value.denominator != 1:
                        raise SystemExit(f"non-integer coefficient in {name}")
                    values.append(int(value))
                rows.append(tuple(values))
            coeffs.append(tuple(rows))
        schemes[name] = {"u": coeffs[0], "v": coeffs[1], "w": coeffs[2],
                         "source": s.source, "rank": s.rank, "additions": s.additions()}
    fn = run_experiment.with_options(timeout=budget + 60)
    result = fn.remote(shape_list, rounds, reps, seed, schemes)
    clean = {k: v for k, v in result.items() if k != "artifacts"}
    report_path = OUT / "a100-mma-results.json"
    report_path.write_text(json.dumps(clean, indent=2) + "\n")
    artifact_dir = OUT / "a100-mma-artifacts"
    artifact_dir.mkdir(exist_ok=True)
    for name, artifact in result["artifacts"].items():
        (artifact_dir / f"{name}.ptx").write_text(artifact["ptx"])
        (artifact_dir / f"{name}.sass").write_text(artifact["sass"])
        artifact.pop("ptx", None)
        artifact.pop("sass", None)
    (artifact_dir / "metadata.json").write_text(json.dumps(result["artifacts"], indent=2) + "\n")
    print(f"device: {result['device']}")
    for row in result["results"]:
        print(f"\nM,K,N={row['shape_mkn']} {row['dtype']}")
        base = next(x for x in row["cases"] if x["algorithm"] == "cublas")
        base_ms = sorted(base["times_ms"])[len(base["times_ms"]) // 2]
        for case in row["cases"]:
            med = sorted(case["times_ms"])[len(case["times_ms"]) // 2]
            print(f"  {case['algorithm']:20} {med:8.3f} ms  {base_ms / med:6.3f}x cuBLAS  "
                  f"relL2={case['relative_l2_error']:.3e} maxAbs={case['max_abs_error']:.3e}")
    print(f"report: {report_path}")
    print(f"PTX/SASS: {artifact_dir}")
