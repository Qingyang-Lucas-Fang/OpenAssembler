"""Generate/compile the bounded correctness suite without production writes."""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
from backend import HERE, install
import backend
from kernels import cta_gemm, warp_gemm
from openassembler.cli import main


def compile_case(name, m, n, k, dtype, kind, optimize):
    path = HERE / "ptx" / f"{name}.ptx"
    path.parent.mkdir(exist_ok=True)
    path.write_text(warp_gemm(m, n, k, dtype) if kind == "warp" else
                    cta_gemm(m, n, k, dtype, async_copy=kind == "async"))
    output = HERE / "build" / name
    backend.OPTIMIZE_LOCALS = optimize
    backend.STAGE_RECORDS.clear()
    start = time.perf_counter()
    rc = main(["compile", str(path), "--out-dir", str(output), "--opt-level", "safe"])
    record = {"compiler": "OpenAssembler experimental MMA selector", "fallback": False,
              "reference_and_ELF_container": "NVIDIA ptxas", "cli_rc": rc,
              "cli_wall_ms": (time.perf_counter() - start)*1000,
              "kernels": backend.STAGE_RECORDS.copy()}
    if rc:
        raise RuntimeError(f"{name} compilation failed with rc={rc}")
    (output / "experiment.json").write_text(json.dumps(record, indent=2) + "\n")
    return dict(build=name, m=m, n=n, k=k, dtype=dtype, tile_m=16 if kind == "warp" else 64,
                tile_n=8 if kind == "warp" else 64, threads=32 if kind == "warp" else 128)


if __name__ == "__main__":
    install()
    configs = []
    for dtype in ("f16", "bf16"):
        configs.append(compile_case(f"warp_rect_{dtype}", 32, 24, 48, dtype, "warp", True))
        configs.append(compile_case(f"cta_rect_base_{dtype}", 128, 192, 256, dtype, "cta", False))
        configs.append(compile_case(f"cta_rect_ssa_{dtype}", 128, 192, 256, dtype, "cta", True))
        configs.append(compile_case(f"async_rect_ssa_{dtype}", 128, 192, 256, dtype, "async", True))
    (HERE / "smoke-suite.json").write_text(json.dumps(configs, indent=2) + "\n")
