"""Write a kernel's code into a cubin built by ptxas for the same PTX signature.

We reuse ptxas's ELF skeleton (symbols, .nv.constant0, .nv.info, relocations) and
replace `.text.<kernel>`, then fix every place that describes the code:
  * .text.<k> sh_info bits 24..31  register count (the driver allocates from this;
    a stale value makes the kernel fault with CUDA_ERROR_ILLEGAL_INSTRUCTION)
  * .nv.info  EIATTR_REGCOUNT      register count
  * .nv.info.<k> EXIT_INSTR_OFFSETS
  * the function symbol's size     (nvdisasm rejects a mismatch)
"""

from __future__ import annotations

import struct

from openassembler.container import nvinfo
from openassembler.container.elf import Elf


def required_regcount(texts: list[str]) -> int:
    """Register count to declare for code that touches the given SASS texts.

    Measured on A100 (corpus/sm_80/gpu_a100_40gb_bisect5.json, 77 variants):
    an instruction writing Rd faults with CUDA_ERROR_ILLEGAL_INSTRUCTION unless
    the declared count >= Rd + 3 (the top two slots are reserved), with no
    fault for any Rd <= 13 at the minimum count of 16 used by ptxas. Wide
    operands (.64 = 2, .128 = 4 registers) count their last register.
    """
    import re

    hi = 0
    for t in texts:
        mn = t.split()[0] if not t.startswith("@") else t.split()[1]
        base = mn.split(".")[0]
        # Conservative: over-counting is harmless, under-counting faults.
        if ".128" in mn or base in ("HMMA", "IMMA", "DMMA", "LDSM"):
            wide = 4
        elif ".64" in mn or "WIDE" in mn or base.startswith("D") or base in ("F2F", "I2F", "F2I"):
            wide = 2
        else:
            wide = 1
        for m in re.finditer(r"\bR(\d+)\b(\.64)?", t):
            width = max(wide, 2 if m[2] else 1)
            hi = max(hi, int(m[1]) + width - 1)
    return max(16, hi + 3)


def words_to_bytes(words: list[int]) -> bytes:
    return b"".join(struct.pack("<QQ", w & (2**64 - 1), w >> 64) for w in words)


def splice_text(cubin: bytes, kname: str, words: list[int], nregs: int, exits: list[int], *,
                replace_helpers: bool = False) -> bytes:
    e = Elf.parse(cubin)
    e.replace_data(f".text.{kname}", words_to_bytes(words))
    t = e.section(f".text.{kname}")
    t.info = (t.info & 0x00FFFFFF) | (nregs << 24)
    if replace_helpers:
        # Selection flattened arithmetic helpers into the replacement kernel. Old
        # helper symbols/relocations describe offsets in the removed code. Retain
        # symbol indices (nv.info uses them), but retire their code ranges.
        from openassembler.container.elf import SHT_SYMTAB, SHT_REL, SHT_RELA, _SYM
        tid = e.sections.index(t)
        symtab = next(sec for sec in e.sections if sec.type == SHT_SYMTAB)
        data = bytearray(symtab.data)
        for i, sym in enumerate(e.symbols):
            if sym.shndx == tid and sym.name != kname and sym.info & 15 == 2:
                sym.value = sym.size = 0
                sym.info &= 0xf0  # STT_NOTYPE, with binding preserved
                _SYM.pack_into(data, i * _SYM.size, sym.name_off, sym.info, sym.other,
                               sym.shndx, sym.value, sym.size)
        symtab.data = bytes(data)
        for sec in e.sections:
            if sec.type in (SHT_REL, SHT_RELA) and sec.info == tid:
                e.replace_data(sec.name, b"")

    recs = nvinfo.parse(e.section(".nv.info").data)
    for r in recs:
        if r.name == "REGCOUNT":
            fn = struct.unpack_from("<I", r.data, 0)[0]
            r.data = struct.pack("<II", fn, nregs)
    e.replace_data(".nv.info", nvinfo.serialize(recs))
    ks = e.section(f".nv.info.{kname}")
    recs = nvinfo.parse(ks.data)
    for r in recs:
        if r.name == "EXIT_INSTR_OFFSETS":
            r.data = struct.pack(f"<{len(exits)}I", *exits)
    e.replace_data(ks.name, nvinfo.serialize(recs))
    return e.serialize()


def text_words(cubin: bytes, kname: str) -> list[int]:
    data = Elf.parse(cubin).section(f".text.{kname}").data
    return [int.from_bytes(data[i:i + 8], "little") | (int.from_bytes(data[i + 8:i + 16], "little") << 64)
            for i in range(0, len(data), 16)]


def regcount(cubin: bytes, kname: str) -> int:
    return Elf.parse(cubin).section(f".text.{kname}").info >> 24
