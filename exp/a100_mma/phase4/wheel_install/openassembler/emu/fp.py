"""Exact IEEE-754 binary32/binary64 arithmetic for the emulator.

Finite values become exact Fractions, the operation is done exactly, and the
result is rounded once with the requested mode. That gives correctly rounded
FADD/FMUL/FFMA/I2F/F2F under RN/RZ/RM/RP, with flush-to-zero (FTZ) as an option.

Hardware-approximate ops (MUFU) are modeled as correctly rounded, marked
HYPOTHESIS: real MUFU.RCP etc. have ~1 ulp error, so emulated results can
legitimately differ from a GPU. Integer division sequences built on MUFU.RCP
are exact by construction, which the differential harness checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

# NVIDIA's canonical NaN. HYPOTHESIS for arithmetic results; confirm on a GPU.
CANON_NAN32 = 0x7FFFFFFF
CANON_NAN64 = 0xFFFFFFFFFFFFFFFF  # placeholder until measured; DADD etc. not yet emulated


@dataclass(frozen=True)
class Fmt:
    ebits: int
    mbits: int  # explicit mantissa bits

    @property
    def bias(self) -> int:
        return (1 << (self.ebits - 1)) - 1

    @property
    def width(self) -> int:
        return 1 + self.ebits + self.mbits


F32 = Fmt(8, 23)
F64 = Fmt(11, 52)
F16 = Fmt(5, 10)
BF16 = Fmt(8, 7)


@dataclass(frozen=True)
class Val:
    """Decoded float: exactly one of nan / inf / finite (with sign for zero)."""
    neg: bool
    nan: bool = False
    inf: bool = False
    q: Fraction = Fraction(0)  # magnitude for finite values
    pl: int = 0                # NaN payload, left-aligned to 52 bits (0 = generated NaN)


def decode(bits: int, f: Fmt = F32, ftz: bool = False) -> Val:
    neg = bool(bits >> (f.width - 1) & 1)
    e = (bits >> f.mbits) & ((1 << f.ebits) - 1)
    mant = bits & ((1 << f.mbits) - 1)
    if e == (1 << f.ebits) - 1:
        return Val(neg, nan=mant != 0, inf=mant == 0, pl=mant << (52 - f.mbits))
    if e == 0:
        if ftz:
            return Val(neg)
        return Val(neg, q=Fraction(mant, 1 << (f.mbits + f.bias - 1)))
    return Val(neg, q=Fraction((1 << f.mbits) | mant) * Fraction(2) ** (e - f.bias - f.mbits))


def _round_int(q: Fraction, mode: str, neg: bool) -> int:
    """Round a non-negative rational to an integer under mode (for |value|)."""
    fl = q.numerator // q.denominator
    rem = q - fl
    if rem == 0:
        return fl
    if mode == "RZ":
        return fl
    if mode == "RP":
        return fl if neg else fl + 1
    if mode == "RM":
        return fl + 1 if neg else fl
    # RN, ties to even
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and fl & 1):
        return fl + 1
    return fl


def encode(v: Val, f: Fmt = F32, mode: str = "RN", ftz: bool = False, keep_nan: bool = False) -> int:
    """NaN results (measured on A100, corpus/sm_80/gpu_a100_40gb.json):
      f32 arithmetic       -> canonical 0x7fffffff
      f64 arithmetic       -> the input NaN, payload and sign kept, quieted
      F2F (keep_nan=True)  -> sign kept, payload shifted to the new width, quieted
    A NaN created from non-NaN inputs (inf - inf) in f64 is still a HYPOTHESIS."""
    sign = int(v.neg) << (f.width - 1)
    emax = (1 << f.ebits) - 1
    if v.nan and v.pl and (keep_nan or f is F64):
        return sign | (emax << f.mbits) | (v.pl >> (52 - f.mbits)) | (1 << (f.mbits - 1))
    if v.nan:
        # Canonical NaNs: f32 0x7fffffff; f16 0x7fff; bf16 0x7fff (HYPOTHESIS for 16-bit:
        # same all-ones-payload convention as f32). f64 uses the IEEE default quiet NaN.
        if f is F32:
            return CANON_NAN32
        if f in (F16, BF16):
            return 0x7FFF
        return (emax << f.mbits) | (1 << (f.mbits - 1))
    if v.inf:
        return sign | (emax << f.mbits)
    q = v.q
    if q == 0:
        return sign
    # exponent e such that 2^e <= q < 2^(e+1)
    e = q.numerator.bit_length() - q.denominator.bit_length()
    if Fraction(2) ** e > q:
        e -= 1
    emin = 1 - f.bias
    if e < emin:
        if ftz:
            return sign  # flush subnormal results to (signed) zero
        e = emin  # subnormal: fixed exponent, fewer significant bits
    scaled = q / Fraction(2) ** (e - f.mbits)
    m = _round_int(scaled, mode, v.neg)
    if m >> (f.mbits + 1):  # rounding carried into a new binade
        m >>= 1
        e += 1
    if e > f.bias:  # overflow
        to_inf = mode == "RN" or (mode == "RP" and not v.neg) or (mode == "RM" and v.neg)
        return sign | ((emax << f.mbits) if to_inf else ((emax - 1) << f.mbits) | ((1 << f.mbits) - 1))
    if m >> f.mbits == 0:  # subnormal (or rounded to zero)
        if ftz and m:
            return sign
        return sign | m
    return sign | ((e + f.bias) << f.mbits) | (m & ((1 << f.mbits) - 1))


def signed_frac(v: Val) -> Fraction:
    return -v.q if v.neg else v.q


def from_frac(x: Fraction, zero_neg: bool = False) -> Val:
    if x == 0:
        return Val(zero_neg)
    return Val(x < 0, q=abs(x))


def mode_of(mods: tuple[str, ...]) -> str:
    for m in ("RZ", "RM", "RP", "RN"):
        if m in mods:
            return m
    return "RN"


# ---- arithmetic ----------------------------------------------------------------

def first_nan(*vs: Val) -> Val | None:
    return next((v for v in vs if v.nan), None)


def add(a: Val, b: Val, mode: str) -> Val:
    if n := first_nan(a, b):
        return n
    if a.inf and b.inf and a.neg != b.neg:
        return Val(False, nan=True)
    if a.inf or b.inf:
        return a if a.inf else b
    s = signed_frac(a) + signed_frac(b)
    if s == 0:
        # IEEE: exact zero sum is -0 only if both are -0, or under RM
        return Val((a.neg and b.neg) if not (a.q or b.q) else mode == "RM")
    return from_frac(s)


def mul(a: Val, b: Val) -> Val:
    neg = a.neg != b.neg
    if n := first_nan(a, b):
        return n
    if (a.inf and b.q == 0 and not b.inf) or (b.inf and a.q == 0 and not a.inf):
        return Val(False, nan=True)
    if a.inf or b.inf:
        return Val(neg, inf=True)
    return Val(neg, q=a.q * b.q)


def fma(a: Val, b: Val, c: Val, mode: str) -> Val:
    if n := first_nan(a, b, c):
        return n
    p = mul(a, b)
    if p.nan:
        return p
    if p.inf or c.inf:
        return add(p, c, mode)
    s = signed_frac(p) + signed_frac(c)
    if s == 0:
        return Val((p.neg and c.neg) if not (p.q or c.q) else mode == "RM")
    return from_frac(s)


def cmp(a: Val, b: Val) -> int | None:
    """-1/0/1, or None if unordered."""
    if a.nan or b.nan:
        return None
    def key(v: Val):
        if v.inf:
            return (-1 if v.neg else 1, 0)
        return (0, signed_frac(v))
    ka, kb = key(a), key(b)
    return (ka > kb) - (ka < kb)


def to_int(v: Val, bits: int, signed: bool, mode: str, src_f64: bool = False) -> int:
    """F2I: round then saturate. NaN (PTX ISA 9.x cvt, confirmed on A100):
    0 if the source is not f64 and the destination is not 64-bit, else 1 << (bits - 1)."""
    lo, hi = (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if signed else (0, (1 << bits) - 1)
    if v.nan:
        return 1 << (bits - 1) if (src_f64 or bits == 64) else 0
    if v.inf:
        return lo if v.neg else hi
    n = _round_int(v.q, mode, v.neg)
    n = -n if v.neg else n
    return max(lo, min(hi, n)) & ((1 << bits) - 1)
