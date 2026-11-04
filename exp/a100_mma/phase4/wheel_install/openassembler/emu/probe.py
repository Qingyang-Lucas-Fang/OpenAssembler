"""Run one probe kernel (tools/ptx_probes.py ABI) on the SASS emulator.

Probe ABI: params (out: u64, in: u64, k: u32, kf: f32) at c[0x0][param_base..+24].
Each thread t reads 16 bytes at in + 16*t and writes 8 bytes at out + 16*t.
"""

from __future__ import annotations

import struct
from pathlib import Path

from openassembler.emu.machine import Launch, Machine, Memory
from openassembler.emu.program import load_listing

IN_BASE = 0x7F00_0000_0000
OUT_BASE = 0x7F00_1000_0000
PARAM_BASE = {"sm_80": 0x160, "sm_86": 0x160, "sm_89": 0x160, "sm_90": 0x210, "sm_100": 0x380}


def probe_params(k: int, kf_bits: int) -> bytes:
    return struct.pack("<QQII", OUT_BASE, IN_BASE, k & 0xFFFFFFFF, kf_bits & 0xFFFFFFFF)


def cubin_consts(cubin: bytes) -> dict[int, bytes]:
    """Constant banks other than bank 0 that ptxas initialized in the cubin
    (.nv.constant2.<k>: jump tables for BRX; .nv.constant3: user .const data)."""
    import re

    from openassembler.container.elf import Elf

    out = {}
    for s in Elf.parse(cubin).sections:
        if m := re.match(r"\.nv\.constant(\d+)", s.name):
            if int(m[1]) != 0 and s.type != 8:
                out[int(m[1])] = s.data
    return out


def run_probe(sass_text: str, inputs: list[tuple[int, int, int, int]], k: int, kf_bits: int,
              arch: str = "sm_80", poison: int | None = None,
              consts: dict[int, bytes] | None = None) -> list[tuple[int, int]]:
    """inputs[t] = 4 x u32 for thread t; returns (out_lo, out_hi) per thread."""
    gmem = Memory()
    for t, words in enumerate(inputs):
        gmem.write_bytes(IN_BASE + 16 * t, struct.pack("<4I", *(w & 0xFFFFFFFF for w in words)))
    launch = Launch(block=(len(inputs), 1, 1), params=probe_params(k, kf_bits),
                    param_base=PARAM_BASE[arch], poison=poison)
    m = Machine(load_listing(sass_text), launch, gmem)
    for bank, data in (consts or {}).items():
        m.cbank.setdefault(bank, Memory()).write_bytes(0, data)
    m.run()
    return [(gmem.read(OUT_BASE + 16 * t, 4), gmem.read(OUT_BASE + 16 * t + 4, 4))
            for t in range(len(inputs))]


def run_probe_file(path: Path, *args, **kw):
    return run_probe(Path(path).read_text(), *args, **kw)
