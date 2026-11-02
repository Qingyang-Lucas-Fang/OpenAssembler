"""Parse nvdisasm operand text into typed operands."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Reg:
    file: str          # "R" or "UR"
    idx: int           # RZ = 255, URZ = 63
    neg: bool = False  # "-R1"
    bnot: bool = False  # "~R1"
    abs: bool = False  # "|R1|"
    sel: str = ""      # suffix such as "H0_H0", "B1", "64", "X4"


@dataclass(frozen=True)
class Pred:
    file: str          # "P" or "UP"
    idx: int           # PT / UPT = 7
    neg: bool = False


_F32_SPECIAL = {"+INF": 0x7F800000, "INF": 0x7F800000, "-INF": 0xFF800000,
                "NAN": 0x7FC00000, "+QNAN": 0x7FC00000, "-QNAN": 0xFFC00000}
_F64_SPECIAL = {"+INF": 0x7FF00000_00000000, "INF": 0x7FF00000_00000000, "-INF": 0xFFF00000_00000000,
                "NAN": 0x7FF80000_00000000, "+QNAN": 0x7FF80000_00000000, "-QNAN": 0xFFF80000_00000000}


def _f32_text_to_bits(text: str) -> int:
    """nvdisasm's decimal text for a float immediate, as its raw 32 bits.
    f32-range values (FADD/FSETP/FFMA/...) round-trip through struct directly;
    a double-precision instruction's immediate (DMUL/DSETP/...) can exceed
    f32's range, and only its high 32 bits sit in a 32-bit immediate slot --
    see ops.py's _fsel note on FSEL selecting raw f64 high words."""
    import struct

    key = text.strip()
    if key in _F32_SPECIAL:
        return _F32_SPECIAL[key]
    try:
        return struct.unpack("<I", struct.pack("<f", float(key)))[0]
    except OverflowError:
        bits64 = _F64_SPECIAL.get(key) or struct.unpack("<Q", struct.pack("<d", float(key)))[0]
        return bits64 >> 32


@dataclass(frozen=True)
class Imm:
    text: str          # raw text; integer vs float meaning is up to the opcode

    def int32(self) -> int:
        """The raw 32 bits nvdisasm's text represents. Most immediates are
        plain decimal/hex integers; FSETP/FADD/etc. print their immediate as
        a decimal float in scientific notation (e.g. "1.175494350822287508e-38")
        or +INF/-INF/NAN/+QNAN -- those parse as the IEEE-754 f32 bit pattern
        of that value, which is what actually sits in the word's immediate
        field for a float-typed instruction."""
        try:
            return int(self.text, 0) & 0xFFFFFFFF
        except ValueError:
            return _f32_text_to_bits(self.text)

    def f32bits(self) -> int:
        """The text read as a FLOAT, whatever it looks like: nvdisasm prints some float
        immediates as plain integers ("FMUL R0, R1, 4096" means 4096.0, bits 0x45800000),
        which int32() would take as the raw bits 0x1000. Text that is explicitly hex
        ("0x40000000", which is how our own compiler writes a float constant) IS the raw bits."""
        if self.text.lstrip("+-").lower().startswith("0x"):
            return int(self.text, 0) & 0xFFFFFFFF
        return _f32_text_to_bits(self.text)


@dataclass(frozen=True)
class CBank:
    bank: int
    off: int
    ur: int | None = None    # cx[URn][off]: bank from a uniform register
    reg: Reg | None = None   # c[bank][Rn+off]: register-indexed (LDC)
    neg: bool = False
    bnot: bool = False
    abs: bool = False


@dataclass(frozen=True)
class Mem:
    base: Reg | None   # None for [0x...] absolute
    wide: bool         # "R2.64"
    off: int
    ur: int | None = None  # "[R2.64+UR4]" or desc[URn]
    scale: int = 1     # "[R9.X16+0x10]": base register scaled (LDS/STS)


@dataclass(frozen=True)
class Label:
    name: str


@dataclass(frozen=True)
class Raw:
    text: str          # SR_TID.X, B0, SB0, etc.


Operand = Reg | Pred | Imm | CBank | Mem | Label | Raw

_REG = re.compile(r"(U?R)(Z|\d+)((?:\.[A-Z0-9_]+)*)")
_PRED = re.compile(r"(U?P)(T|\d)")
_CB = re.compile(r"c(x?)\[(0x[0-9a-f]+|UR\d+)\]\[(?:(U?R(?:Z|\d+))\s*\+?\s*)?(-?0x[0-9a-f]+)?\]")
_MEM = re.compile(r"(?:desc\[UR(\d+)\])?\[(?:(U?R(?:Z|\d+))(\.64)?(?:\.X(\d+))?)?\s*(?:\+\s*UR(\d+))?\s*(?:([+-])?\s*(0x[0-9a-f]+))?\]")
_IMM = re.compile(r"[+-]?(0x[0-9a-f]+|\d+(\.\d*)?(e[+-]?\d+)?|INF|QNAN|NAN)", re.I)


def _strip_mods(t: str) -> tuple[str, bool, bool, bool]:
    neg = bnot = ab = False
    t = t.replace(".reuse", "")  # operand-cache hint; no semantic effect
    if t.startswith("-") and not t.startswith("-0x") and not t[1:2].isdigit():
        neg, t = True, t[1:]
    if t.startswith("~"):
        bnot, t = True, t[1:]
    if t.startswith("|") and t.endswith("|"):
        ab, t = True, t[1:-1]
    return t, neg, bnot, ab


def parse_reg(t: str) -> Reg | None:
    t, neg, bnot, ab = _strip_mods(t.strip())
    m = _REG.fullmatch(t)
    if not m:
        return None
    zero = 255 if m[1] == "R" else 63
    sel = ".".join(s for s in m[3].split(".") if s and s != "reuse")
    return Reg(m[1], zero if m[2] == "Z" else int(m[2]), neg, bnot, ab, sel)


def parse(t: str) -> Operand:
    t = t.strip()
    if t.startswith("`(") and t.endswith(")"):
        return Label(t[2:-1])
    if t.startswith("!") or _PRED.fullmatch(t):
        m = _PRED.fullmatch(t.lstrip("!"))
        if m:
            return Pred(m[1], 7 if m[2] == "T" else int(m[2]), t.startswith("!"))
    if r := parse_reg(t):
        return r
    core, neg, bnot, ab = _strip_mods(t)
    if m := _CB.fullmatch(core):
        bank_t = m[2]
        ur = int(bank_t[2:]) if bank_t.startswith("UR") else None
        reg = parse_reg(m[3]) if m[3] else None
        return CBank(0 if ur is not None else int(bank_t, 16), int(m[4] or "0", 16), ur, reg, neg, bnot, ab)
    if m := _MEM.fullmatch(t):
        base = parse_reg(m[2]) if m[2] else None
        off = int(m[7], 16) if m[7] else 0
        ur = int(m[1]) if m[1] else (int(m[5]) if m[5] else None)
        return Mem(base, bool(m[3]), -off if m[6] == "-" else off, ur, int(m[4]) if m[4] else 1)
    if _IMM.fullmatch(t):
        return Imm(t)
    return Raw(t)


def split_operands(s: str) -> list[str]:
    """Split on top-level commas (none occur inside [] in practice, but be safe)."""
    out, depth, cur = [], 0, []
    for ch in s:
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        out.append("".join(cur))
    return [o.strip() for o in out]
