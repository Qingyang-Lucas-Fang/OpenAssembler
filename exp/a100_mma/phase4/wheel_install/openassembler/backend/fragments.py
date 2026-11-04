"""ptxas's own SASS sequences for PTX ops we don't hand-select (IEEE div/rcp/sqrt, the
.approx MUFU wrappers, integer div/rem by a register, f32<->int cvt with rounding).

Why copy ptxas instead of writing our own: these sequences encode numerical facts
(Newton steps around an approximate MUFU, FCHK's slow-path trigger, the I2F.RP/F2I
reciprocal trick with its two fix-up steps, the scale-by-2^24 denormal wrappers) that
are only "right" because they match NVIDIA's implementation bit for bit. Copying is the
project rule ("never guess"), and the copy is checked against ptxas's own SASS for the
same probe on the emulator (tests/test_backend_ops.py).

Data: ptxas_fragments.json, generated from CUDA 12.8 probes. Each record
contains inputs (original register -> source index), output, renamed-label-ready
lines, and the original PTX/SASS for independent regression replay. Arithmetic
helpers are flattened to branches; registers and predicates are private at each
use site. No NVIDIA toolkit binaries or source files are packaged.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parent / "ptxas_fragments.json"

_REG = re.compile(r"(?<![\w.])R(\d+)(?!\w)")
_PRED = re.compile(r"(?<![\w.])(!?)P([0-6])(?!\w)")
_GUARD = re.compile(r"^@(!?)P([0-6])\s+")


class FragmentError(Exception):
    pass


@lru_cache(maxsize=1)
def library() -> dict:
    return json.loads(DATA.read_text())["ops"] if DATA.exists() else {}


def split(text: str):
    """'@!P0 FFMA R1, -R2, R3, 1' -> (guard ('!', 0) | None, 'FFMA', ['R1', '-R2', 'R3', '1'])."""
    g = None
    m = _GUARD.match(text)
    if m:
        g = (m[1], int(m[2]))
        text = text[m.end():]
    head, _, rest = text.strip().partition(" ")
    ops = [o.strip() for o in _split_top(rest)] if rest.strip() else []
    return g, head, ops


def _split_top(s: str):
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return out


# Opcodes a fragment may contain, with the operand positions they WRITE. Everything else
# in the operand list (and the guard) is read. Positions are from the SASS syntax
# (nvdisasm) and the emulator's handlers (openassembler/emu/ops.py), e.g.
#   ISETP.cmp.bop  Pd, Pd2, a, b, Pc        LOP3.LUT Pd, Rd, a, b, c, lut, Pc
#   IADD3 Rd, Pcarry, a, b, c               LEA Rd, Pcarry, a, b, sh
#   PLOP3.LUT Pd, Pd2, a, b, c, lut, lut2   FCHK Pd, a, b
# 64-bit-result forms (F64, .WIDE, ...) write a register PAIR and are refused here.
_DEF_ALL_P_PREFIX = ("ISETP", "FSETP", "PLOP3", "FCHK", "DSETP", "HSETP2")
_WHITELIST = {"MUFU", "FCHK", "FFMA", "FADD", "FMUL", "IMAD", "IADD3", "LOP3", "SHF", "ISETP", "FSETP",
              "PLOP3", "SEL", "FSEL", "LEA", "IABS", "I2F", "F2I", "FRND", "MOV", "HFMA2", "BRA",
              "FMNMX", "IMNMX"}
_WIDE_MODS = {"F64", "S64", "U64", "WIDE", "64"}


def roles(text: str):
    """-> (guard, opcode, operands, kinds) with kinds[i] in
    {"rd", "ru", "pd", "pu", None}: register def/use, predicate def/use, other."""
    g, head, ops = split(text)
    base, *mods = head.split(".")
    if base not in _WHITELIST:
        raise FragmentError(f"opcode {head} not allowed in a fragment")
    if _WIDE_MODS & set(mods) and base != "SHF":   # SHF.*.U64 still writes ONE 32-bit register
        raise FragmentError(f"64-bit result form {head}")
    kinds = []
    for i, o in enumerate(ops):
        bare = o.replace(".reuse", "").split(".")[0].strip("-~!|")
        isreg = bool(re.fullmatch(r"R\d+", bare))
        ispred = bool(re.fullmatch(r"P[0-6]", bare))
        if not (isreg or ispred):
            kinds.append(None)
            continue
        d = False
        if base in _DEF_ALL_P_PREFIX:
            d = i < 2 and ispred
        elif base == "LOP3":
            d = (i == 0) or (i == 1 and isreg and ops[0].startswith("P"))
        elif base in ("IADD3", "LEA") and "X" not in mods:
            d = i == 0 or (i == 1 and ispred)
        elif base == "BRA":
            d = False
        else:
            d = i == 0
        if base == "IMAD" and "HI" in mods and i == 3 and isreg:
            kinds.append("r2u")
        else:
            kinds.append(("rd" if isreg else "pd") if d else ("ru" if isreg else "pu"))
    return g, head, ops, kinds


def lower_line(text: str, reg, pred, pair=None):
    """One fragment SASS line -> (fmt, defs, uses, guard, target) for ra.Op, with
    reg(n) / pred(n) mapping ptxas register / predicate numbers to virtual ones.
    fmt uses {dK}/{sK} for register defs/uses, {pK} for predicate defs, {qK} for
    predicate uses (the negation travels in the use tuple, (PV, neg)), {target} for
    a branch label. `.reuse` is dropped (our scheduler sets reuse bits itself)."""
    g, head, ops, kinds = roles(text)
    out, defs, uses, pdefs, target = [], [], [], [], None
    nq, nr = 0, 0
    for o, k in zip(ops, kinds):
        o = o.replace(".reuse", "")
        if k in ("rd", "ru", "r2u"):
            n = int(_REG.search(o)[1])
            if k == "rd":
                ph = f"{{d{len(defs)}}}"
                defs.append(reg(n))
            else:
                ph = f"{{s{nr}}}"
                nr += 1
                uses.append(pair(n) if k == "r2u" else reg(n))
            out.append(_REG.sub(ph.replace("\\", ""), o, count=1))
        elif k == "pd":
            out.append(f"{{p{len(pdefs)}}}")
            pdefs.append(pred(int(o[-1])))
        elif k == "pu":
            out.append(f"{{q{nq}}}")
            nq += 1
            uses.append((pred(int(o[-1])), o.startswith("!")))
        elif o.startswith("`("):
            target = o[2:-1]
            out.append("`({target})")
        else:
            out.append(o)
    fmt = head + (" " + ", ".join(out) if out else "")
    guard = (pred(g[1]), g[0] == "!") if g else None
    return fmt, defs + pdefs, uses, guard, target


def reads_writes(text: str):
    """(register numbers read, written, predicates read, written) of one fragment line."""
    g, head, ops, kinds = roles(text)
    rr, rw, pr, pw = set(), set(), set(), set()
    for o, k in zip(ops, kinds):
        if k == "rd":
            rw.add(int(_REG.search(o)[1]))
        elif k in ("ru", "r2u"):
            n = int(_REG.search(o)[1])
            rr.add(n)
            if k == "r2u":
                rr.add(n + 1)
        elif k == "pd":
            pw.add(int(o[-1]))
        elif k == "pu":
            pr.add(int(o[-1]))
    if g:
        pr.add(g[1])
    return rr, rw, pr, pw
