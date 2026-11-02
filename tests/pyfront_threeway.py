"""Three-way runner for pyfront kernels (shared by tests/test_pyfront*.py): the SAME kernel
body is run
  1. as plain Python -- one real OS thread per GPU thread, so syncthreads() is a true
     threading.Barrier and shared_array() is one list shared by the block,
  2. as the PTX pyfront emitted, on the reference PTX interpreter,
  3. as the SASS our backend generated (and encoded: every word must have a template),
     on the SASS emulator.
Results are {array name: list}; f32 values compare by bits except that any NaN equals
any NaN (PTX canonicalizes NaN payloads, Python keeps them).

The SASS leg raises Pending (-> pytest.skip) when the backend or the encoder rejects the
kernel: pyfront may emit PTX (div.rn.f32, sqrt.rn.f32, cvt, ex2/lg2.approx, ...) that our
backend learns in a separate change; those legs turn on by themselves once it does."""

from __future__ import annotations

import math
import struct
import sys
import threading
from pathlib import Path

from openassembler.backend import ra
from openassembler.emu.machine import Launch as SassLaunch
from openassembler.emu.machine import Machine, Memory
from openassembler.emu.program import load_listing
from openassembler.isa.encode import EncodeError, Encoder
from openassembler.ptx.interp import Launch as InterpLaunch
from openassembler.ptx.interp import run_kernel
from openassembler.ptx.parse import parse_module
from openassembler.pyfront import compile_kernel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from bench_build import listing  # noqa: E402

BASE = 0x10000
_ENC: list[Encoder] = []


class Pending(Exception):
    """Our backend/encoder can't compile this kernel yet (the reason is the message)."""


def f32(x: float) -> float:
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError:
        return math.copysign(math.inf, x)


def _gpu_sqrt(x):
    return math.nan if x < 0 or math.isnan(x) else math.sqrt(x)   # math.sqrt raises on x < 0


def _gpu_log(x):
    return math.nan if x < 0 or math.isnan(x) else (-math.inf if x == 0 else math.log(x))


def _gpu_exp(x):
    try:
        return math.exp(x)
    except OverflowError:
        return math.inf


PY_INTRINSICS = {"sqrt": _gpu_sqrt, "rsqrt": lambda x: math.nan if x < 0 else
                 (math.copysign(math.inf, x) if x == 0 else 1 / math.sqrt(x)),
                 "exp": _gpu_exp, "log": _gpu_log, "tanh": math.tanh}


def _ty(vals):
    return "f" if any(isinstance(v, float) for v in vals) else "i"


def _pack(vals, ty):
    return b"".join(struct.pack("<i" if ty == "i" else "<f", v) for v in vals)


def _read(mem, addr, count, ty):
    raw = [mem.read(addr + 4 * k, 4) for k in range(count)]
    return list(struct.unpack(f"<{count}{'i' if ty == 'i' else 'f'}", struct.pack(f"<{count}I", *raw)))


def run_python(fn, arrays: dict, scalars: dict, threads: int) -> dict:
    py = {k: [f32(x) if isinstance(x, float) else x for x in v] for k, v in arrays.items()}
    body = fn.__wrapped__ if hasattr(fn, "__wrapped__") else fn
    tls = threading.local()
    barrier = threading.Barrier(threads, timeout=20)
    shared: dict[int, list] = {}
    lock = threading.Lock()

    def shared_array(ty, n):
        tls.n_shared = getattr(tls, "n_shared", 0) + 1
        with lock:
            return shared.setdefault(tls.n_shared, [0.0 if ty.ptx == "f32" else 0] * n)

    g = {"tid_x": lambda: tls.t, "tid_y": lambda: 0, "tid_z": lambda: 0, "ctaid_x": lambda: 0,
         "ctaid_y": lambda: 0, "ctaid_z": lambda: 0, "ntid_x": lambda: threads, "ntid_y": lambda: 1,
         "ntid_z": lambda: 1, "syncthreads": lambda: barrier.wait(), "shared_array": shared_array,
         **PY_INTRINSICS}
    saved = {k: body.__globals__[k] for k in g if k in body.__globals__}
    body.__globals__.update(g)
    errors = []
    argnames = body.__code__.co_varnames[:body.__code__.co_argcount]

    def thread(t):
        tls.t = t
        try:
            body(*[py[p] if p in py else scalars[p] for p in argnames])
        except BaseException as e:  # noqa: BLE001
            errors.append(e)
            barrier.abort()

    try:
        ts = [threading.Thread(target=thread, args=(t,)) for t in range(threads)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    finally:
        for k in g:
            body.__globals__.pop(k, None)
        body.__globals__.update(saved)
    real = [e for e in errors if not isinstance(e, threading.BrokenBarrierError)]
    if real or errors:
        raise (real or errors)[0]
    # what the kernel can observe: every f32 store rounds to f32
    return {k: [f32(x) if isinstance(x, float) else x for x in v] for k, v in py.items()}


def run_ptx_and_sass(fn, arrays: dict, scalars: dict, threads: int):
    """Returns (interp result, sass result or a Pending exception)."""
    mod = parse_module(compile_kernel(fn))
    name = fn.__name__
    k = mod.kernels[name]
    addrs, base = {}, BASE
    for a in arrays:
        addrs[a] = base
        base += 0x1000

    def fresh():
        m = Memory()
        for a, v in arrays.items():
            m.write_bytes(addrs[a], _pack(v, _ty(v)))
        return m

    rd = lambda m: {a: _read(m, addrs[a], len(arrays[a]), _ty(arrays[a])) for a in arrays}  # noqa: E731
    mem_i = fresh()
    run_kernel(mod, name, InterpLaunch(block=(threads, 1, 1),
                                       params={p: addrs.get(p, scalars.get(p)) for _t, p in k.params}), mem_i)
    try:
        code, st = ra.compile_kernel(k)
        if not _ENC:
            _ENC.append(Encoder("sm_80"))
        ra.encode_code(_ENC[0], code, st["labels"])
    except (ra.NotSupported, EncodeError) as e:
        return rd(mem_i), Pending(f"backend support pending: {type(e).__name__}: {e}")
    prog = load_listing(listing(code, st["labels"]))
    blob = b""
    for ty, p in k.params:
        if ty == "u64":
            blob += b"\0" * ((-len(blob)) % 8) + struct.pack("<Q", addrs[p])
        else:
            blob += b"\0" * ((-len(blob)) % 4) + struct.pack("<f" if ty == "f32" else "<i", scalars[p])
    mem_s = fresh()
    Machine(prog, SassLaunch(block=(threads, 1, 1), params=blob, param_base=0x160), mem_s).run()
    return rd(mem_i), rd(mem_s)


def _bits(x):
    if isinstance(x, float):
        return "nan" if math.isnan(x) else struct.pack("<f", x)
    return x


def same(a: dict, b: dict) -> bool:
    return {k: [_bits(x) for x in v] for k, v in a.items()} == {k: [_bits(x) for x in v] for k, v in b.items()}


def close(a: dict, b: dict, rel: float, abs_: float = 0.0) -> bool:
    """For approximate intrinsics: equal up to rel/abs tolerance; inf/NaN must match exactly."""
    for k in a:
        for x, y in zip(a[k], b[k]):
            if isinstance(x, float) and (math.isnan(x) or math.isinf(x) or math.isnan(y) or math.isinf(y)):
                if _bits(x) != _bits(y):
                    return False
            elif abs(x - y) > max(abs_, rel * abs(x)):
                return False
    return True
