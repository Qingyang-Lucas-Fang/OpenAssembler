"""Compile a small A100 experiment matrix and retain per-stage compiler costs."""
from __future__ import annotations
import json
import sys
import time

sys.dont_write_bytecode = True
from backend import HERE, install
import backend
from kernels import cta_gemm
from openassembler.cli import main


def compile_variant(name, m, n, k, dtype, tile, asynchronous, ssa, licm, interlock, triton=False):
    source = HERE / "ptx" / f"{name}.ptx"
    source.write_text(cta_gemm(m, n, k, dtype, asynchronous, tile))
    output = HERE / "build" / name
    backend.OPTIMIZE_LOCALS = ssa
    backend.HOIST_LOOP_INVARIANTS = licm
    backend.RELAX_TENSOR = interlock
    backend.MINED_SCALAR = interlock
    backend.STAGE_RECORDS.clear()
    start = time.perf_counter()
    rc = main(["compile", str(source), "--out-dir", str(output), "--opt-level", "safe"])
    if rc:
        raise RuntimeError(f"{name}: compile rc={rc}")
    (output / "experiment.json").write_text(json.dumps({
        "compiler": "OpenAssembler experimental MMA selector", "fallback": False,
        "reference_and_ELF_container": "NVIDIA ptxas", "cli_rc": rc,
        "cli_wall_ms": (time.perf_counter()-start)*1000,
        "kernels": backend.STAGE_RECORDS.copy()}, indent=2) + "\n")
    return dict(build=name, m=m, n=n, k=k, dtype=dtype, tile_m=tile, tile_n=tile,
                threads=(tile//32)**2*32, triton_sweep=triton)


if __name__ == "__main__":
    install()
    configs = [
        compile_variant("v2_4096_sync_base", 4096,4096,4096,"f16",64,False,False,False,False),
        compile_variant("v2_4096_async_safe", 4096,4096,4096,"f16",64,True,True,False,False),
        compile_variant("v2_4096_async_interlock", 4096,4096,4096,"f16",64,True,True,False,True),
        compile_variant("v2_4096_async_licm", 4096,4096,4096,"f16",64,True,True,True,True),
        compile_variant("v2_4096_async_tile128", 4096,4096,4096,"f16",128,True,True,True,True),
        compile_variant("v2_4096_bf16_tile128", 4096,4096,4096,"bf16",128,True,True,True,True,True),
        compile_variant("v2_mlp_f16_tile128", 4096,16384,4096,"f16",128,True,True,True,True,True),
        compile_variant("v2_mlp_bf16_tile128", 4096,16384,4096,"bf16",128,True,True,True,True,True),
    ]
    (HERE / "performance-suite.json").write_text(json.dumps(configs, indent=2) + "\n")
