"""openassembler.cppfront: CUDA C/C++ -> PTX via nvcc --ptx, then through our own
backend. Needs the oracle (nvcc via env/cuda.sh / colima) -- skipped offline,
same convention as test_backend.py's needs_corpus.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import pytest

from openassembler.backend import ra
from openassembler.emu.machine import Launch as SassLaunch
from openassembler.emu.machine import Machine, Memory
from openassembler.emu.program import load_listing
from openassembler.ptx.parse import parse_module

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from bench_build import listing  # noqa: E402


def _oracle_available() -> bool:
    from openassembler import oracle

    return oracle.available("nvcc")


needs_oracle = pytest.mark.skipif(not _oracle_available(), reason="needs the CUDA toolkit (nvcc): env/fetch_cuda.sh")

CUDA_SRC = """
extern "C" __global__ void vec_add(float* a, float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}
"""


@needs_oracle
def test_cuda_c_compiles_and_runs_bit_exact(tmp_path):
    from openassembler.cppfront import compile_to_ptx

    src = ROOT / "corpus" / ".tmp" / "test_cppfront_vec_add.cu"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(CUDA_SRC)
    try:
        ptx, kname = compile_to_ptx(src, entry=None, arch="sm_80")
        assert kname == "vec_add"
        mod = parse_module(ptx)
        code, stats = ra.compile_kernel(mod.kernels[kname])
        prog = load_listing(listing(code, stats["labels"]))

        n = 8
        a_vals = [float(i) for i in range(n)]
        b_vals = [float(i) * 10 for i in range(n)]
        A, B, OUT = 0x1000, 0x2000, 0x3000
        gmem = Memory()
        for i, v in enumerate(a_vals):
            gmem.write_bytes(A + 4 * i, struct.pack("<f", v))
        for i, v in enumerate(b_vals):
            gmem.write_bytes(B + 4 * i, struct.pack("<f", v))
        launch = SassLaunch(block=(n, 1, 1), params=struct.pack("<QQQi", A, B, OUT, n), param_base=0x160)
        Machine(prog, launch, gmem).run()
        got = [struct.unpack("<f", struct.pack("<I", gmem.read(OUT + 4 * i, 4)))[0] for i in range(n)]
        assert got == [a_vals[i] + b_vals[i] for i in range(n)]
    finally:
        src.unlink(missing_ok=True)
        src.with_suffix(".ptx").unlink(missing_ok=True)
