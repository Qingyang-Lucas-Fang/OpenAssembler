"""PTX floating-point semantics (exact, via openassembler.emu.fp), plus cvt.

Approximate instructions (rcp/sqrt/rsqrt/ex2/lg2/sin/cos/tanh .approx, div.approx)
have implementation-defined error in the spec. We compute the exact result and
tag it APPROX; the harness compares those lanes with a ulp tolerance instead of
bit-exactly.
"""

from __future__ import annotations

import dataclasses as _dc

import math
from fractions import Fraction

from openassembler.emu import fp
from openassembler.ptx.interp import M, NotModeled, Undefined, sext, types_of
from openassembler.ptx.parse import ImmF, ImmI

APPROX = {"rcp", "sqrt", "rsqrt", "ex2", "lg2", "sin", "cos", "tanh"}


def fmt_of(t: str):
    return {"f32": fp.F32, "f64": fp.F64, "f16": fp.F16, "bf16": fp.BF16}[t]


PACKED = {"f16x2": fp.F16, "bf16x2": fp.BF16}


def packed(ins):
    t = types_of(ins)[0]
    return PACKED.get(t)


def lanes2(it, ln, op, f, ins):
    v = it.get(ln, op, 32)
    return [fp.decode((v >> (16 * i)) & 0xFFFF, f, ftz=ins.has("ftz")) for i in (0, 1)]


def put2(it, ln, d, vals, f, ins, mode=None):
    mode = mode or rnd(ins)
    out = 0
    for i, v in enumerate(vals):
        b = fp.encode(v, f, mode, ftz=ins.has("ftz"))
        if ins.has("sat"):
            r = fp.decode(b, f)
            if r.nan or r.neg:
                b = 0
            elif r.inf or r.q > 1:
                b = fp.encode(fp.Val(False, q=Fraction(1)), f)
        out |= b << (16 * i)
    it.put(ln, d, out, 32)


def fsrc(it, ln, op, fmt, ins):
    if isinstance(op, ImmF):
        v = fp.decode(op.bits, fp.F32 if op.width == 32 else fp.F64)
    elif isinstance(op, ImmI):
        raise NotModeled("integer immediate in float op")
    else:
        v = fp.decode(it.get(ln, op, fmt.width), fmt, ftz=ins.has("ftz"))
    return v


def fout(it, ln, d, v, fmt, ins, mode=None):
    mode = mode or rnd(ins)
    b = fp.encode(v, fmt, mode, ftz=ins.has("ftz"))
    if ins.has("sat"):
        r = fp.decode(b, fmt)
        if r.nan or r.neg:
            b = 0
        elif r.inf or r.q > 1:
            b = fp.encode(fp.Val(False, q=Fraction(1)), fmt)
    it.put(ln, d, b, fmt.width)


def rnd(ins) -> str:
    for m, r in (("rn", "RN"), ("rz", "RZ"), ("rm", "RM"), ("rp", "RP")):
        if m in ins.mods:
            return r
    return "RN"


def _ft(ins):
    t = types_of(ins)[0]
    if t in PACKED:
        raise NotModeled(f"{t} in scalar path")
    return fmt_of(t)


def _binary2(it, ln, ins, fn):
    f = packed(ins)
    d, a, b = ins.args
    xa, xb = lanes2(it, ln, a, f, ins), lanes2(it, ln, b, f, ins)
    put2(it, ln, d, [fn(xa[i], xb[i]) for i in (0, 1)], f, ins)


def fadd(it, ln, ins):
    if packed(ins):
        neg = ins.op == "sub"
        return _binary2(it, ln, ins, lambda x, y: fp.add(x, _dc.replace(y, neg=not y.neg) if neg else y, rnd(ins)))
    f = _ft(ins)
    d, a, b = ins.args
    va, vb = fsrc(it, ln, a, f, ins), fsrc(it, ln, b, f, ins)
    if ins.op == "sub":
        vb = _dc.replace(vb, neg=not vb.neg)
    fout(it, ln, d, fp.add(va, vb, rnd(ins)), f, ins)


def fmul(it, ln, ins):
    if packed(ins):
        return _binary2(it, ln, ins, fp.mul)
    f = _ft(ins)
    d, a, b = ins.args
    fout(it, ln, d, fp.mul(fsrc(it, ln, a, f, ins), fsrc(it, ln, b, f, ins)), f, ins)


def ffma(it, ln, ins):
    if packed(ins):
        f = packed(ins)
        d, a, b, c = ins.args
        xs = [lanes2(it, ln, o, f, ins) for o in (a, b, c)]
        put2(it, ln, d, [fp.fma(xs[0][i], xs[1][i], xs[2][i], rnd(ins)) for i in (0, 1)], f, ins)
        return
    f = _ft(ins)
    d, a, b, c = ins.args
    fout(it, ln, d, fp.fma(*(fsrc(it, ln, x, f, ins) for x in (a, b, c)), rnd(ins)), f, ins)


def _exact_div(va, vb):
    if va.nan or vb.nan or (va.inf and vb.inf) or (va.q == 0 and vb.q == 0 and not va.inf and not vb.inf):
        return fp.Val(False, nan=True)
    neg = va.neg != vb.neg
    if va.inf or (vb.q == 0 and not vb.inf):
        return fp.Val(neg, inf=True)
    if vb.inf:
        return fp.Val(neg)
    return fp.Val(neg, q=va.q / vb.q)


def fdiv(it, ln, ins):
    f = _ft(ins)
    d, a, b = ins.args
    v = _exact_div(fsrc(it, ln, a, f, ins), fsrc(it, ln, b, f, ins))
    if ins.has("approx") or ins.has("full"):
        ln.regs.setdefault("%%approx", set()).add(d.name)
    fout(it, ln, d, v, f, ins)


def _minmax_val(va, vb, want_min, nan_prop):
    if va.nan or vb.nan:
        if nan_prop or (va.nan and vb.nan):
            return fp.Val(False, nan=True)
        return vb if va.nan else va
    c = fp.cmp(va, vb)
    if c == 0 and va.q == 0 and not va.inf:
        return (va if va.neg else vb) if want_min else (vb if va.neg else va)
    return va if (c <= 0) == want_min else vb


def fminmax(it, ln, ins):
    if packed(ins):
        return _binary2(it, ln, ins, lambda x, y: _minmax_val(x, y, ins.op == "min", ins.has("NaN")))
    f = _ft(ins)
    d, a, b = ins.args
    va, vb = fsrc(it, ln, a, f, ins), fsrc(it, ln, b, f, ins)
    want_min = ins.op == "min"
    if va.nan or vb.nan:
        if ins.has("NaN") or (va.nan and vb.nan):
            fout(it, ln, d, fp.Val(False, nan=True), f, ins)
        else:
            fout(it, ln, d, vb if va.nan else va, f, ins)
        return
    c = fp.cmp(va, vb)
    if c == 0 and va.q == 0 and not va.inf:
        # spec: -0.0 < +0.0 for min/max
        r = (va if va.neg else vb) if want_min else (vb if va.neg else va)
    else:
        r = va if (c <= 0) == want_min else vb
    fout(it, ln, d, r, f, ins)


def fabsneg(it, ln, ins):
    f = _ft(ins)
    d, a = ins.args
    bits_ = it.get(ln, a, f.width)
    sign = 1 << (f.width - 1)
    v = fp.decode(bits_, f, ftz=ins.has("ftz"))
    if v.nan:
        # abs/neg on NaN: spec says result is unspecified NaN; ptxas emits sign-bit ops.
        it.put(ln, d, bits_ & ~sign if ins.op == "abs" else bits_ ^ sign, f.width)
        return
    r = fp.Val(False if ins.op == "abs" else not v.neg, False, v.inf, v.q)
    it.put(ln, d, fp.encode(r, f, ftz=ins.has("ftz")), f.width)


def unary_approx(it, ln, ins):
    op = ins.op
    f = _ft(ins)
    d, a = ins.args
    v = fsrc(it, ln, a, f, ins)
    exact = not ins.has("approx")
    if op == "rcp":
        r = _exact_div(fp.Val(False, q=Fraction(1)), v)
        fout(it, ln, d, r, f, ins)
    elif op == "sqrt" and exact:
        if v.nan or (v.neg and (v.q != 0 or v.inf)):
            r = fp.Val(False, nan=True)
        elif v.inf or v.q == 0:
            r = v
        else:
            r = _exact_sqrt(v.q, f, rnd(ins))
            it.put(ln, d, r, f.width)
            return
        fout(it, ln, d, r, f, ins)
    else:
        if op in ("sin", "cos") and not (v.nan or v.inf) and abs(v.q) > 100 * Fraction(355, 113):
            # spec: outside [-100pi, 100pi] "only best effort ... no defined error guarantees"
            raise Undefined(f"{op}.approx outside [-100pi, 100pi]")
        x = math.nan if v.nan else (math.copysign(math.inf, -1 if v.neg else 1) if v.inf else (-0.0 if v.neg and v.q == 0 else float(fp.signed_frac(v))))
        try:
            y = {"sqrt": lambda x: math.sqrt(x) if x >= 0 else math.nan,
                 "rsqrt": lambda x: (math.copysign(math.inf, x) if x == 0 else 1 / math.sqrt(x)) if x >= 0 else math.nan,
                 "ex2": lambda x: 2.0 ** x, "lg2": lambda x: -math.inf if x == 0 else (math.log2(x) if x > 0 else math.nan),
                 "sin": math.sin, "cos": math.cos, "tanh": math.tanh}[op](x)
        except OverflowError:
            y = math.inf
        except ValueError:
            y = math.nan
        if math.isnan(y):
            r = fp.Val(False, nan=True)
        elif math.isinf(y):
            r = fp.Val(y < 0, inf=True)
        else:
            r = fp.from_frac(Fraction(y))
        fout(it, ln, d, r, f, ins)
    if not exact:
        ln.regs.setdefault("%%approx", set()).add(d.name)


def _exact_sqrt(q: Fraction, f, mode: str) -> int:
    """Correctly rounded sqrt: find the representable neighbours of sqrt(q) exactly."""
    approx = fp.encode(fp.from_frac(Fraction(math.sqrt(float(q)))), f, "RZ")
    cands = [approx + k for k in range(-3, 4) if approx + k >= 0]
    vals = [(c, fp.decode(c, f).q) for c in cands]
    lo = max((c for c, v in vals if v * v <= q), default=0)
    hi = lo + 1
    lv, hv = fp.decode(lo, f).q, fp.decode(hi, f).q
    if lv * lv == q:
        return lo
    if mode in ("RZ", "RM"):
        return lo
    if mode == "RP":
        return hi
    mid = (lv + hv) / 2
    return lo if q < mid * mid or (q == mid * mid and lo % 2 == 0) else hi


# ---- cvt ---------------------------------------------------------------------------------

def cvt(it, ln, ins):
    dt, st = types_of(ins)[:2]
    d, a = ins.args[0], ins.args[1]
    if dt in PACKED:
        # cvt.rn.{f16x2,bf16x2}.f32 d, a, b: d.hi = cvt(a), d.lo = cvt(b)
        f = PACKED[dt]
        _, a, b = ins.args
        put2(it, ln, d, [fsrc(it, ln, b, fp.F32, ins), fsrc(it, ln, a, fp.F32, ins)], f, ins)
        return
    if st in PACKED:
        raise NotModeled(f"cvt from {st}")
    fi = st.startswith(("f", "bf"))
    fo = dt.startswith(("f", "bf"))
    db = int(dt.lstrip("bfsu"))
    sb = int(st.lstrip("bfsu"))
    if not fi and not fo:
        v = it.get(ln, a, sb)
        if st.startswith("s"):
            v = sext(v, sb)
        if ins.has("sat"):
            lo, hi = (-(1 << (db - 1)), (1 << (db - 1)) - 1) if dt.startswith("s") else (0, M(db))
            v = max(lo, min(hi, v))
        v &= M(db)
        if dt.startswith("s") and db < 32:
            v = sext(v, db)
        it.put(ln, d, v, max(db, 32) if db < 32 else db)
        return
    if fi and not fo:
        v = fsrc(it, ln, a, fmt_of(st), ins)
        mode = next((r for m, r in (("rni", "RN"), ("rzi", "RZ"), ("rmi", "RM"), ("rpi", "RP")) if m in ins.mods), "RZ")
        it.put(ln, d, fp.to_int(v, db, dt.startswith("s"), mode, src_f64=st == "f64"), db)
        return
    if not fi and fo:
        v = it.get(ln, a, sb)
        if st.startswith("s"):
            v = sext(v, sb)
        fout(it, ln, d, fp.from_frac(Fraction(v)), fmt_of(dt), ins)
        return
    # float -> float
    v = fsrc(it, ln, a, fmt_of(st), ins)
    if v.nan and {st, dt} == {"f32", "f64"}:
        # spec: unspecified NaN. A100 keeps sign + shifted payload; match it so the
        # interpreter is not stricter than hardware.
        it.put(ln, d, fp.encode(v, fmt_of(dt), keep_nan=True), fmt_of(dt).width)
        return
    ri = next((r for m, r in (("rni", "RN"), ("rzi", "RZ"), ("rmi", "RM"), ("rpi", "RP")) if m in ins.mods), None)
    if ri:
        if not (v.nan or v.inf):
            n = fp._round_int(v.q, ri, v.neg)
            v = fp.Val(v.neg, q=Fraction(n))
        fout(it, ln, d, v, fmt_of(dt), ins, mode="RN")
        return
    fout(it, ln, d, v, fmt_of(dt), ins)
