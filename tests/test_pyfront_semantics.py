"""Differential test for openassembler.pyfront: the SAME kernel body is run three ways and
must agree --
  1. as plain Python (one call per thread, the language's real semantics),
  2. as the PTX pyfront emitted, on the reference PTX interpreter,
  3. as the SASS our backend generated, on the SASS emulator.

`compile --verify` only compares (2) with (3), so a bug in the Python -> PTX step (a dropped
`return`, a loop that runs the wrong number of times) passes it. These cases are exactly the
silent miscompiles found by auditing the docs; each one used to pass --verify."""

from __future__ import annotations

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

THREADS = 8
BASE = 0x10000


@kernel
def early_return(a: Array(i32), n: i32):
    i = tid_x()
    if i >= n:
        return
    a[i] = i + 1


@kernel
def bound_snapshot(o: Array(i32), n: i32):
    i = tid_x()
    m = n
    s = 0
    for j in range(0, m):
        m = 100
        s += 1
    o[i] = s


@kernel
def neg_step(o: Array(i32), n: i32):
    i = tid_x()
    s = 0
    for j in range(n, 0, -2):
        s += j
    o[i] = s


@kernel
def pos_step(o: Array(i32), n: i32):
    i = tid_x()
    s = 0
    for j in range(1, n, 3):
        s += j
    o[i] = s


@kernel
def empty_loop(o: Array(i32), n: i32):
    i = tid_x()
    s = 7
    for j in range(0, n):
        s += 1
    o[i] = s


@kernel
def elif_chain(a: Array(i32), o: Array(i32), n: i32):
    i = tid_x()
    v = a[i]
    if v < 0:
        o[i] = 0 - 1
    elif v < 10:
        o[i] = v
    elif v < 100:
        o[i] = v * 2
    else:
        o[i] = 1000


@kernel
def and_or(a: Array(i32), o: Array(i32), n: i32):
    i = tid_x()
    if i > 1 and i < 6:
        o[i] = a[i]
    if i < 1 or i > 6:
        o[i] = 0 - 5


@kernel
def nested_loops(o: Array(i32), n: i32):
    i = tid_x()
    s = 0
    for x in range(0, n):
        for y in range(x, n):
            s += x * y + i
    o[i] = s


def _ty(vals):
    return "f" if any(isinstance(v, float) for v in vals) else "i"


@kernel
def float_literals(a: Array(f32), o: Array(f32), n: i32):
    i = tid_x()
    o[i] = a[i] * 2.0 + 1.5
    if i < 4:
        o[i] = o[i] - 0.25


@kernel
def float_lhs_literal(a: Array(f32), o: Array(f32), n: i32):
    i = tid_x()
    o[i] = 0.5 - a[i] * 2.0


@kernel
def float_branch(a: Array(f32), o: Array(f32), n: i32):
    i = tid_x()
    v = a[i]
    if v > 0.5:
        o[i] = v * 4.0
    else:
        o[i] = v + 8.0


def _pack(vals, ty):
    return b"".join(struct.pack("<i" if ty == "i" else "<f", v) for v in vals)


def _read(mem, addr, count, ty):
    raw = [mem.read(addr + 4 * k, 4) for k in range(count)]
    return list(struct.unpack(f"<{count}{'i' if ty == 'i' else 'f'}", struct.pack(f"<{count}I", *raw)))


def _f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def run_three_ways(fn, arrays: dict, scalars: dict):
    """arrays: name -> list of ints; scalars: name -> int. Returns (python, interp, sass) as
    {array name: list}."""
    py = {k: [_f32(x) if isinstance(x, float) else x for x in v] for k, v in arrays.items()}
    body = fn.__wrapped__
    for t in range(THREADS):                       # real Python, one call per thread
        g = {"tid_x": lambda t=t: t, "ctaid_x": lambda: 0, "ntid_x": lambda: THREADS}
        body.__globals__.update(g)
        try:
            body(*[py[p] if p in py else scalars[p] for p in body.__code__.co_varnames[:body.__code__.co_argcount]])
        finally:
            for k in g:
                body.__globals__.pop(k, None)

    mod = parse_module(compile_kernel(fn))
    name = fn.__name__
    order = [p for p, _ in mod.kernels[name].params] if False else [n for _t, n in mod.kernels[name].params]
    addrs, base = {}, BASE
    for k in arrays:
        addrs[k] = base
        base += 0x1000

    def fresh():
        m = Memory()
        for k, v in arrays.items():
            m.write_bytes(addrs[k], _pack(v, _ty(v)))
        return m

    mem_i = fresh()
    run_kernel(mod, name, InterpLaunch(block=(THREADS, 1, 1),
                                         params={p: addrs.get(p, scalars.get(p)) for p in order}), mem_i)

    code, st = ra.compile_kernel(mod.kernels[name])
    prog = load_listing(listing(code, st["labels"]))
    blob = b""
    for ty, p in mod.kernels[name].params:
        if ty == "u64":
            blob += b"\0" * ((-len(blob)) % 8) + struct.pack("<Q", addrs[p])
        else:
            blob += b"\0" * ((-len(blob)) % 4) + struct.pack("<f" if ty == "f32" else "<i", scalars[p])
    mem_s = fresh()
    Machine(prog, SassLaunch(block=(THREADS, 1, 1), params=blob, param_base=0x160), mem_s).run()

    rd = lambda m: {k: _read(m, addrs[k], len(arrays[k]), _ty(arrays[k])) for k in arrays}  # noqa: E731
    return py, rd(mem_i), rd(mem_s)


CASES = [
    ("early_return", early_return, {"a": [-1] * THREADS}, {"n": 3}),
    ("early_return_none", early_return, {"a": [-1] * THREADS}, {"n": 8}),
    ("bound_snapshot", bound_snapshot, {"o": [0] * THREADS}, {"n": 5}),
    ("neg_step", neg_step, {"o": [0] * THREADS}, {"n": 9}),
    ("neg_step_empty", neg_step, {"o": [0] * THREADS}, {"n": 0}),
    ("pos_step", pos_step, {"o": [0] * THREADS}, {"n": 14}),
    ("empty_loop", empty_loop, {"o": [0] * THREADS}, {"n": 0}),
    ("elif_chain", elif_chain, {"a": [-4, 0, 9, 10, 55, 99, 100, 250], "o": [0] * THREADS}, {"n": 8}),
    ("and_or", and_or, {"a": [10, 11, 12, 13, 14, 15, 16, 17], "o": [0] * THREADS}, {"n": 8}),
    ("nested_loops", nested_loops, {"o": [0] * THREADS}, {"n": 5}),
    ("float_literals", float_literals, {"a": [0.0, 1.0, -2.5, 3.25, 100.0, -0.125, 0.5, 7.0], "o": [0.0] * THREADS}, {"n": 8}),
    ("float_lhs_literal", float_lhs_literal, {"a": [0.0, 1.0, -2.5, 3.25, 100.0, -0.125, 0.5, 7.0], "o": [0.0] * THREADS}, {"n": 8}),
    ("float_branch", float_branch, {"a": [0.0, 0.5, 0.75, -1.0, 2.0, 0.25, 9.0, -9.0], "o": [0.0] * THREADS}, {"n": 8}),
]


@pytest.mark.parametrize("label,fn,arrays,scalars", CASES, ids=[c[0] for c in CASES])
def test_python_interpreter_and_sass_agree(label, fn, arrays, scalars):
    py, interp, sass = run_three_ways(fn, arrays, scalars)
    assert interp == py, f"PTX pyfront emitted != plain Python\n python {py}\n ptx    {interp}"
    assert sass == py, f"our SASS != plain Python\n python {py}\n sass   {sass}"


REJECTED = {
    "for_else": ("for j in range(0, n):\n        a[0] = j\n    else:\n        a[0] = 9", "for ... else"),
    "assign_loop_var": ("for j in range(0, n):\n        j = j + 10\n        a[0] = j", "loop variable"),
    "nested_def": ("def helper():\n        a[0] = 42\n    a[1] = 1", "nested function"),
    "int_into_f32": ("a[0] = 1", "storing"),
    "float_index": ("a[f] = 1.0", "index"),
    "float_bound": ("for j in range(0, f):\n        a[0] = 1.0", "bounds"),
}


@pytest.mark.parametrize("label", sorted(REJECTED))
def test_construct_is_rejected_not_silently_miscompiled(label, tmp_path):
    body, needle = REJECTED[label]
    src = tmp_path / f"{label}.py"
    src.write_text("from openassembler.pyfront import Array, f32, i32, kernel\n\n@kernel\n"
                   f"def k(a: Array(f32), n: i32, f: f32):\n    {body}\n")
    from openassembler.pyfront.compile import find_kernel

    with pytest.raises(NotSupported, match=needle):
        compile_kernel(find_kernel(src, "k"))


def test_error_line_numbers_are_file_lines(tmp_path):
    src = tmp_path / "lines.py"
    src.write_text("from openassembler.pyfront import Array, f32, i32, kernel\n\n\n@kernel\n"
                   "def k(a: Array(f32), n: i32):\n    i = tid_x()\n    while i < n:\n        i = i + 1\n")
    from openassembler.pyfront.compile import find_kernel

    with pytest.raises(NotSupported, match=r"line 7\b"):         # `while` is on line 7 of the file
        compile_kernel(find_kernel(src, "k"))


def test_constant_cse_does_not_leak_across_a_join():
    """Two independent guarded stores. The constant 4 (for i*4) was first materialized inside
    the first branch and then reused after the join by threads that had skipped that branch,
    so they stored through a stale register: `if i<3: o[i]=7` then `if i>5: o[i]=9` left
    threads 6 and 7 writing nothing. Both frontends emit this pattern all the time."""
    src = (".version 8.7\n.target sm_80\n.address_size 64\n"
           ".visible .entry t(.param .u64 o, .param .s32 n)\n{\n"
           " .reg .pred %p<4>; .reg .b32 %r<16>; .reg .b64 %rd<8>;\n"
           " ld.param.u64 %rd1, [o]; cvta.to.global.u64 %rd1, %rd1;\n mov.u32 %r2, %tid.x;\n"
           " setp.lt.s32 %p1, %r2, 3;\n @!%p1 bra SKIP1;\n"
           " mul.wide.u32 %rd3, %r2, 4; add.s64 %rd3, %rd1, %rd3;\n mov.s32 %r5, 7; st.global.u32 [%rd3], %r5;\n"
           "SKIP1:\n setp.gt.s32 %p2, %r2, 5;\n @!%p2 bra SKIP2;\n"
           " mul.wide.u32 %rd4, %r2, 4; add.s64 %rd4, %rd1, %rd4;\n mov.s32 %r6, 9; st.global.u32 [%rd4], %r6;\n"
           "SKIP2:\n ret;\n}\n")
    mod = parse_module(src)
    code, st = ra.compile_kernel(mod.kernels["t"])
    mem = Memory()
    mem.write_bytes(BASE, b"\0" * 4 * THREADS)
    blob = struct.pack("<Qi", BASE, THREADS)
    Machine(load_listing(listing(code, st["labels"])),
            SassLaunch(block=(THREADS, 1, 1), params=blob, param_base=0x160, poison=0xDEADBEEF), mem).run()
    assert _read(mem, BASE, THREADS, "i") == [7, 7, 7, 0, 0, 0, 9, 9]
