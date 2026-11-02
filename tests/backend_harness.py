"""Shared harness for tests/test_backend_ops.py and tests/test_backend_loops.py.

compile_and_diff(): PTX -> our backend (openassembler.backend.ra) -> every instruction
encoded by our encoder (so a missing template fails the test, not just the emulator)
-> SASS emulator vs PTX reference interpreter on caller-sized buffers, CTA by CTA,
and once more with registers poisoned (a read of a never-written register changes the
result on real hardware, so it must change nothing here).

Unlike openassembler.verify.run_and_diff (every array = n_elems f32), the caller says
how big each buffer is and what it holds, which loop kernels (a[row*n + j], C[i*N + j])
need: an out-of-bounds read would otherwise silently read zeros in both engines.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from openassembler.backend import ra  # noqa: E402
from openassembler.emu.machine import Launch as SassLaunch  # noqa: E402
from openassembler.emu.machine import Machine, Memory  # noqa: E402
from openassembler.emu.program import load_listing  # noqa: E402
from openassembler.isa.encode import Encoder  # noqa: E402
from openassembler.ptx.interp import Launch as InterpLaunch  # noqa: E402
from openassembler.ptx.interp import run_kernel  # noqa: E402
from openassembler.ptx.parse import parse_module  # noqa: E402

_ENC = None


def encoder() -> Encoder:
    global _ENC
    if _ENC is None:
        _ENC = Encoder("sm_80")
    return _ENC


def listing(code, labels) -> str:
    """Emulator-readable listing (same format as tools/bench_build.listing)."""
    inv = {}
    for name, i in labels.items():
        inv.setdefault(i, []).append(name)
    lines = []
    for i, (t, _c) in enumerate(code):
        for name in inv.get(i, []):
            lines.append(f"{name}:")
        lines.append(f"        /*{16 * i:04x}*/ {t} ; /* 0x{0:016x} */\n        /* 0x{0:016x} */")
    return "\n".join(lines)


def compile_ptx(ptx: str, kname: str):
    """-> (code, stats, words). Raises ra.NotSupported / EncodeError."""
    k = parse_module(ptx).kernels[kname]
    code, st = ra.compile_kernel(k)
    words = ra.encode_code(encoder(), code, st["labels"])
    return code, st, words


def _param_bytes(k, values: dict) -> bytes:
    out = b""
    for ty, name in k.params:
        sz = 8 if ty.endswith("64") else 4
        if len(out) % sz:
            out += b"\0" * (sz - len(out) % sz)
        out += struct.pack("<Q" if sz == 8 else "<I", values[name] & ((1 << (8 * sz)) - 1))
    return out


def compile_and_diff(ptx: str, kname: str, arrays: dict[str, bytes], scalars: dict[str, int],
                     block=(32, 1, 1), grid=(1, 1, 1), max_steps: int = 3_000_000, reference_sass: str | None = None):
    """arrays: param name -> initial bytes (each placed in its own window);
    scalars: param name -> raw 32-bit value (f32 as bits). Returns (code, stats, [mismatch])
    where a mismatch is (param, byte offset, ours, expected) per differing 32-bit word."""
    mod = parse_module(ptx)
    k = mod.kernels[kname]
    code, st, _words = compile_ptx(ptx, kname)
    prog = load_listing(listing(code, st["labels"]))
    addrs, base = {}, 0x7F00_0000_0000
    for name, data in arrays.items():
        addrs[name] = base
        base += (len(data) + 0x10FF) // 0x1000 * 0x1000
    vals = {**{n: a for n, a in addrs.items()}, **scalars}
    pbytes = _param_bytes(k, vals)
    results = []
    for poison in (None, 0xDEADBEEF):
        mem = Memory()
        for name, data in arrays.items():
            mem.write_bytes(addrs[name], data)
        for cz in range(grid[2]):
            for cy in range(grid[1]):
                for cx in range(grid[0]):
                    Machine(prog, SassLaunch(grid=grid, block=block, ctaid=(cx, cy, cz), params=pbytes,
                                             smem_static=st.get("smem", 0), poison=poison), mem
                            ).run(max_steps=max_steps)
        results.append(mem)
    ref = Memory()
    for name, data in arrays.items():
        ref.write_bytes(addrs[name], data)
    for cz in range(grid[2]):
        for cy in range(grid[1]):
            for cx in range(grid[0]):
                if reference_sass is None:
                    run_kernel(mod, kname, InterpLaunch(block=block, grid=grid, ctaid=(cx, cy, cz), params=vals), ref)
                else:
                    Machine(load_listing(reference_sass, kernel=kname), SassLaunch(grid=grid, block=block,
                            ctaid=(cx, cy, cz), params=pbytes), ref).run(max_steps=max_steps)
    bad = []
    for name, data in arrays.items():
        for off in range(0, len(data) - 3, 4):
            e = ref.read(addrs[name] + off, 4)
            for tag, mem in zip(("", "poisoned "), results):
                g = mem.read(addrs[name] + off, 4)
                if g != e:
                    bad.append((tag + name, off, g, e))
    return code, st, bad


def f32b(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def pack_f32(xs) -> bytes:
    return b"".join(struct.pack("<f", x) for x in xs)


def pack_u32(xs) -> bytes:
    return b"".join(struct.pack("<I", x & 0xFFFFFFFF) for x in xs)
