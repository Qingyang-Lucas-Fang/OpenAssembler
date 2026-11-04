"""Generic fields of a Volta+ (sm_70..sm_120) 128-bit SASS instruction word.

These are the fields shared by (almost) every instruction. Per-opcode fields
(modifiers, operand kinds) come from specs/<arch>/*.json, discovered by tools/.
Bit positions are verified on real sm_80 words; see tests/test_word.py.
"""

from __future__ import annotations

from dataclasses import dataclass

RZ = 255   # zero register for R fields
URZ = 63   # zero register for UR fields
PT = 7     # always-true predicate


def bits(w: int, lo: int, width: int) -> int:
    """Extract `width` bits starting at bit `lo`."""
    return (w >> lo) & ((1 << width) - 1)


def set_bits(w: int, lo: int, width: int, value: int) -> int:
    mask = ((1 << width) - 1) << lo
    return (w & ~mask) | ((value << lo) & mask)


@dataclass(frozen=True)
class Control:
    """Scheduling info compiled into bits 105..125 of every instruction."""

    stall: int      # cycles to wait before issuing the next instruction (0-15)
    yield_: bool    # True = allow the warp scheduler to switch warps
    wbar: int       # scoreboard set when the result is written (7 = none)
    rbar: int       # scoreboard set when source operands are read (7 = none)
    wait: int       # 6-bit mask of scoreboards to wait on before issue
    reuse: int      # 4-bit operand reuse-cache flags (slots a, b, c, d)

    def __str__(self) -> str:
        sb = lambda b: "-" if b == 7 else str(b)  # noqa: E731
        wait = "".join(str(i) if self.wait >> i & 1 else "-" for i in range(6))
        return (f"S{self.stall:02d} {'Y' if self.yield_ else '-'} "
                f"W{sb(self.wbar)} R{sb(self.rbar)} wait[{wait}] reuse={self.reuse:04b}")


def control_valid(c: Control) -> bool:
    """nvdisasm (sm_80) rejects no-yield (bit 109 = 1) with stall 0 or 12..15;
    yield (bit 109 = 0) allows any stall. Measured by sweeping stall x bit 109
    on 8 seeds of different opcodes."""
    return c.yield_ or 1 <= c.stall <= 11


def control(w: int) -> Control:
    # Field positions match Mesa NAK's sm70 encoder (set_instr_deps). Bit 109 is
    # active-low: nvdisasm's sm_120 annotations print bit 109 = 0 as
    # "?WAITn_END_GROUP" (warp may switch) and 1 as "?transN". NAK's "yld" names
    # the raw bit, i.e. the opposite convention. ptxas still sets bits 109 and
    # 122..125 on sm_120 (docs/research/02).
    return Control(
        stall=bits(w, 105, 4),
        yield_=not bits(w, 109, 1),
        wbar=bits(w, 110, 3),
        rbar=bits(w, 113, 3),
        wait=bits(w, 116, 6),
        reuse=bits(w, 122, 4),
    )


@dataclass(frozen=True)
class Fields:
    """Common operand slots. Whether a slot is meaningful depends on the opcode."""

    opcode: int     # bits 0..8: operation
    form: int       # bits 9..11: operand form (which of b/c are reg/imm/cbank/UR)
    pred: int       # bits 12..14: guard predicate (7 = PT, i.e. unguarded)
    pred_neg: bool  # bit 15: guard is negated (@!Px)
    rd: int         # bits 16..23
    ra: int         # bits 24..31
    rb: int         # bits 32..39 (register form)
    imm32: int      # bits 32..63 (immediate form)
    cb_off: int     # bits 38..53: constant-bank byte offset
    cb_bank: int    # bits 54..58: constant-bank index
    rc: int         # bits 64..71


def fields(w: int) -> Fields:
    return Fields(
        opcode=bits(w, 0, 9),
        form=bits(w, 9, 3),
        pred=bits(w, 12, 3),
        pred_neg=bool(bits(w, 15, 1)),
        rd=bits(w, 16, 8),
        ra=bits(w, 24, 8),
        rb=bits(w, 32, 8),
        imm32=bits(w, 32, 32),
        cb_off=bits(w, 38, 16),
        cb_bank=bits(w, 54, 5),
        rc=bits(w, 64, 8),
    )
