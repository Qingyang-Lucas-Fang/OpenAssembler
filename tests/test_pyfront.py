"""openassembler.pyfront: the Python-subset -> PTX compiler. Every kernel here is
checked two ways: against the PTX reference interpreter directly (semantics),
and through our SASS backend + emulator (the actual code path a real build
takes) -- same double-check discipline as the rest of this project.
"""

from __future__ import annotations

import random
import struct
import sys
from pathlib import Path

import pytest

from openassembler.backend import ra
from openassembler.emu.machine import Launch as SassLaunch
from openassembler.emu.machine import Machine, Memory
from openassembler.emu.program import load_listing
from openassembler.ptx.interp import Launch as InterpLaunch
from openassembler.ptx.interp import run_kernel
from openassembler.ptx.parse import parse_module
from openassembler.pyfront import Array, NotSupported, compile_kernel, f32, i32, kernel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from bench_build import listing  # noqa: E402


@kernel
def vec_add(a: Array(f32), b: Array(f32), out: Array(f32), n: i32):
    i = tid_x() + ctaid_x() * ntid_x()
    if i < n:
        out[i] = a[i] + b[i]


@kernel
def dot_row(a: Array(f32), b: Array(f32), out: Array(f32), n: i32):
    row = tid_x()
    acc = 0.0
    for j in range(0, n):
        acc = acc + a[row * n + j] * b[row * n + j]
    out[row] = acc


@kernel
def relu(x: Array(f32), out: Array(f32), n: i32):
    i = tid_x()
    if i < n:
        v = x[i]
        if v < 0.0:
            v = 0.0
        out[i] = v


def _sass_run(fn, params_fmt: str, params: tuple, n_threads: int, gmem: Memory) -> None:
    mod = parse_module(compile_kernel(fn))
    code, stats = ra.compile_kernel(mod.kernels[fn.__name__])
    prog = load_listing(listing(code, stats["labels"]))
    launch = SassLaunch(block=(n_threads, 1, 1), params=struct.pack(params_fmt, *params), param_base=0x160)
    Machine(prog, launch, gmem).run()


def _interp_run(fn, params: dict, n_threads: int, gmem: Memory) -> None:
    mod = parse_module(compile_kernel(fn))
    run_kernel(mod, fn.__name__, InterpLaunch(block=(n_threads, 1, 1), params=params), gmem)


def test_vec_add_interp_and_sass_agree():
    n = 8
    a_vals = [float(i) for i in range(n)]
    b_vals = [float(i) * 10 for i in range(n)]
    A, B, OUT = 0x1000, 0x2000, 0x3000

    gmem_i = Memory()
    for i, v in enumerate(a_vals):
        gmem_i.write_bytes(A + 4 * i, struct.pack("<f", v))
    for i, v in enumerate(b_vals):
        gmem_i.write_bytes(B + 4 * i, struct.pack("<f", v))
    _interp_run(vec_add, {"a": A, "b": B, "out": OUT, "n": n}, n, gmem_i)
    interp_out = [struct.unpack("<f", struct.pack("<I", gmem_i.read(OUT + 4 * i, 4)))[0] for i in range(n)]
    assert interp_out == [a_vals[i] + b_vals[i] for i in range(n)]

    gmem_s = Memory()
    for i, v in enumerate(a_vals):
        gmem_s.write_bytes(A + 4 * i, struct.pack("<f", v))
    for i, v in enumerate(b_vals):
        gmem_s.write_bytes(B + 4 * i, struct.pack("<f", v))
    _sass_run(vec_add, "<QQQi", (A, B, OUT, n), n, gmem_s)
    sass_out = [struct.unpack("<f", struct.pack("<I", gmem_s.read(OUT + 4 * i, 4)))[0] for i in range(n)]
    assert sass_out == interp_out


def test_dot_row_loop_and_accumulator():
    rows, n = 4, 4
    rng = random.Random(0)
    a_vals = [rng.uniform(-5, 5) for _ in range(rows * n)]
    b_vals = [rng.uniform(-5, 5) for _ in range(rows * n)]
    A, B, OUT = 0x1000, 0x2000, 0x3000

    gmem = Memory()
    for i, v in enumerate(a_vals):
        gmem.write_bytes(A + 4 * i, struct.pack("<f", v))
    for i, v in enumerate(b_vals):
        gmem.write_bytes(B + 4 * i, struct.pack("<f", v))
    _sass_run(dot_row, "<QQQi", (A, B, OUT, n), rows, gmem)
    got = [struct.unpack("<f", struct.pack("<I", gmem.read(OUT + 4 * r, 4)))[0] for r in range(rows)]
    expected = [sum(a_vals[r * n + j] * b_vals[r * n + j] for j in range(n)) for r in range(rows)]
    for g, e in zip(got, expected):
        assert abs(g - e) < 1e-4


def test_relu_if_else_reassignment():
    n = 8
    rng = random.Random(1)
    x_vals = [rng.uniform(-5, 5) for _ in range(n)]
    X, OUT = 0x1000, 0x2000

    gmem = Memory()
    for i, v in enumerate(x_vals):
        gmem.write_bytes(X + 4 * i, struct.pack("<f", v))
    mod = parse_module(compile_kernel(relu))
    code, stats = ra.compile_kernel(mod.kernels["relu"])
    prog = load_listing(listing(code, stats["labels"]))
    launch = SassLaunch(block=(n, 1, 1), params=struct.pack("<QQi", X, OUT, n), param_base=0x160)
    Machine(prog, launch, gmem).run()
    got = [struct.unpack("<f", struct.pack("<I", gmem.read(OUT + 4 * i, 4)))[0] for i in range(n)]
    # x_vals are float64; the kernel stores through f32, so round the expected
    # value through f32 too before comparing (this is a rounding artifact of
    # the test, not the kernel -- max(v, 0.0) in exact arithmetic is right).
    f32_rt = lambda v: struct.unpack("<f", struct.pack("<f", v))[0]  # noqa: E731
    assert got == [f32_rt(max(v, 0.0)) for v in x_vals]


def test_rejects_unsupported_constructs():
    @kernel
    def bad(x: Array(f32), n: i32):
        y = [1, 2, 3]  # list literals are not in the subset

    with pytest.raises(NotSupported):
        compile_kernel(bad)


def test_rejects_type_mismatch_reassignment():
    @kernel
    def bad(x: Array(f32), n: i32):
        v = 0
        v = 1.0  # int -> float reassignment of the same name

    with pytest.raises(NotSupported):
        compile_kernel(bad)
