"""Integer division / remainder by a constant -> multiply-high sequences.

Algorithms: Granlund & Montgomery (1994) / Hacker's Delight ch. 10, written here
as straight-line programs over a tiny SASS-like IR so the same object can be
(1) emitted as SASS, (2) emulated, and (3) proved equal to x / d with z3.

IR ops (32-bit unless noted):
  ("mulhi_u", dst, a, imm)       dst = (a * imm) >> 32        -> IMAD.HI.U32
  ("mulhi_s", dst, a, imm)       dst = (sext(a) * sext(imm)) >> 32   -> IMAD.HI
  ("shr_u", dst, a, k)           logical shift                 -> SHF.R.U32.HI
  ("shr_s", dst, a, k)           arithmetic shift              -> SHF.R.S32.HI
  ("sub", dst, a, b) / ("add", dst, a, b)                      -> IADD3
  ("lea_hi_sign", dst, a, b)     dst = a + (b >>> 31)          -> LEA.HI
  ("mad", dst, q, imm, x)        dst = q * imm + x             -> IMAD
  ("mov", dst, a)
Operand: register name (str) or int immediate.
"""

from __future__ import annotations

from dataclasses import dataclass

M32 = (1 << 32) - 1


@dataclass
class Seq:
    ops: list[tuple]
    result: str
    kind: str  # e.g. "pow2", "magic", "magic+add"

    @property
    def n(self) -> int:
        return len([o for o in self.ops if o[0] != "mov"])


def _log2(d: int) -> int | None:
    return d.bit_length() - 1 if d & (d - 1) == 0 else None


def udiv(d: int) -> Seq:
    assert 1 <= d <= M32
    if d == 1:
        return Seq([("mov", "q", "x")], "q", "id")
    k = _log2(d)
    if k is not None:
        return Seq([("shr_u", "q", "x", k)], "q", "pow2")
    if d > 0x80000000:  # q is 0 or 1
        return Seq([("geu", "q", "x", d)], "q", "cmp")
    # Find the smallest shift with a 33-bit magic m = ceil(2^(32+s) / d).
    for s in range(0, 33):
        m = -(-(1 << (32 + s)) // d)
        err = m * d - (1 << (32 + s))
        if err <= (1 << s):  # exact for all 32-bit x (Hacker's Delight thm 10-1)
            break
    if m < (1 << 32):
        ops = [("mulhi_u", "t", "x", m)]
        if s:
            ops.append(("shr_u", "q", "t", s))
            return Seq(ops, "q", "magic")
        return Seq(ops, "t", "magic")
    # 33-bit magic: q = (((x - t) >> 1) + t) >> (s - 1), t = mulhi(x, m - 2^32)
    m -= 1 << 32
    return Seq([("mulhi_u", "t", "x", m), ("sub", "u", "x", "t"), ("shr_u", "u", "u", 1),
                ("add", "u", "u", "t"), ("shr_u", "q", "u", s - 1)], "q", "magic+add")


def sdiv(d: int) -> Seq:
    """Signed division (truncating) by a positive constant 2 <= d < 2^31."""
    assert 2 <= d < (1 << 31)
    k = _log2(d)
    if k is not None:
        # q = (x + ((x >> 31) >>> (32 - k))) >> k
        return Seq([("shr_s", "s", "x", 31), ("shr_u", "s", "s", 32 - k), ("add", "t", "x", "s"),
                    ("shr_s", "q", "t", k)], "q", "pow2")
    # Hacker's Delight 10-1 (signed magic)
    two31 = 1 << 31
    ad = d
    t = two31
    anc = t - 1 - t % ad
    p = 31
    q1, r1 = two31 // anc, two31 - (two31 // anc) * anc
    q2, r2 = two31 // ad, two31 - (two31 // ad) * ad
    while True:
        p += 1
        q1, r1 = 2 * q1, 2 * r1
        if r1 >= anc:
            q1, r1 = q1 + 1, r1 - anc
        q2, r2 = 2 * q2, 2 * r2
        if r2 >= ad:
            q2, r2 = q2 + 1, r2 - ad
        delta = ad - r2
        if not (q1 < delta or (q1 == delta and r1 == 0)):
            break
    m = (q2 + 1) & M32
    s = p - 32
    ops = [("mulhi_s", "t", "x", m)]
    if m >= two31:  # magic is negative as s32: add x back
        ops.append(("add", "t", "t", "x"))
    if s:
        ops.append(("shr_s", "t", "t", s))
    ops.append(("lea_hi_sign", "q", "t", "t"))  # q = t + (t >>> 31): round toward zero
    return Seq(ops, "q", "magic")


def with_rem(seq: Seq, d: int) -> Seq:
    """Append r = x - q*d (one IMAD with -d)."""
    return Seq(seq.ops + [("mad", "r", seq.result, (-d) & M32, "x")], "r", seq.kind + "+rem")


# ---- evaluation (concrete) and proof (z3) -------------------------------------------

def run(seq: Seq, x: int) -> int:
    env = {"x": x & M32}
    val = lambda o: o if isinstance(o, int) else env[o]  # noqa: E731
    sx = lambda v: v - (1 << 32) if v >> 31 else v  # noqa: E731
    for op in seq.ops:
        k, dst = op[0], op[1]
        if k == "mov":
            r = val(op[2])
        elif k == "mulhi_u":
            r = (val(op[2]) * op[3]) >> 32
        elif k == "mulhi_s":
            r = (sx(val(op[2])) * sx(op[3])) >> 32
        elif k == "shr_u":
            r = val(op[2]) >> op[3]
        elif k == "shr_s":
            r = sx(val(op[2])) >> op[3]
        elif k == "add":
            r = val(op[2]) + val(op[3])
        elif k == "sub":
            r = val(op[2]) - val(op[3])
        elif k == "lea_hi_sign":
            r = val(op[2]) + (val(op[3]) >> 31)
        elif k == "mad":
            r = val(op[2]) * op[3] + val(op[4])
        elif k == "geu":
            r = int(val(op[2]) >= op[3])
        else:
            raise ValueError(k)
        env[dst] = r & M32
    return env[seq.result]


def prove(seq: Seq, d: int, signed: bool, rem: bool, timeout_ms: int = 60000) -> str:
    """Return 'proved', 'refuted: x=...' or 'unknown'.

    Division-free specification (SAT-friendly; avoids bit-blasting a divider):
      unsigned: q*d <= x < (q+1)*d                       (64-bit, no overflow)
      signed:   x >= 0: q*d <= x < (q+1)*d ;  x < 0: (q-1)*d < x <= q*d
    For `rem`, the sequence is <div prefix> ; r = q*(-d) + x. We prove the prefix
    computes the true quotient q; then r = x - q*d holds by construction, and
    0 <= |r| < d means no 32-bit wraparound changes it."""
    import z3

    ops = seq.ops
    qname = seq.result
    if rem:
        last = ops[-1]
        assert last[0] == "mad" and last[3] == (-d) & M32 and last[4] == "x", last
        ops, qname = ops[:-1], last[2]
    x = z3.BitVec("x", 32)
    env = {"x": x}
    val = lambda o: z3.BitVecVal(o & M32, 32) if isinstance(o, int) else env[o]  # noqa: E731
    for op in ops:
        k, dst = op[0], op[1]
        if k == "mov":
            r = val(op[2])
        elif k == "mulhi_u":
            r = z3.Extract(63, 32, z3.ZeroExt(32, val(op[2])) * z3.ZeroExt(32, val(op[3])))
        elif k == "mulhi_s":
            r = z3.Extract(63, 32, z3.SignExt(32, val(op[2])) * z3.SignExt(32, val(op[3])))
        elif k == "shr_u":
            r = z3.LShR(val(op[2]), op[3])
        elif k == "shr_s":
            r = val(op[2]) >> op[3]
        elif k == "add":
            r = val(op[2]) + val(op[3])
        elif k == "sub":
            r = val(op[2]) - val(op[3])
        elif k == "lea_hi_sign":
            r = val(op[2]) + z3.LShR(val(op[3]), 31)
        elif k == "mad":
            r = val(op[2]) * val(op[3]) + val(op[4])
        elif k == "geu":
            r = z3.If(z3.UGE(val(op[2]), val(op[3])), z3.BitVecVal(1, 32), z3.BitVecVal(0, 32))
        else:
            raise ValueError(k)
        env[dst] = r
    q = env[qname]
    D = z3.BitVecVal(d, 64)
    if signed:
        X, Q = z3.SignExt(32, x), z3.SignExt(32, q)
        ok = z3.If(X >= 0, z3.And(Q * D <= X, X < (Q + 1) * D),
                   z3.And((Q - 1) * D < X, X <= Q * D))
    else:
        X, Q = z3.ZeroExt(32, x), z3.ZeroExt(32, q)
        ok = z3.And(z3.ULE(Q * D, X), z3.ULT(X, (Q + 1) * D))
    s = z3.Solver()
    s.set("timeout", timeout_ms)
    s.add(z3.Not(ok))
    r = s.check()
    if r == z3.unsat:
        return "proved"
    if r == z3.sat:
        return f"refuted: x={s.model()[x]}"
    return "unknown"


def lower(op: str, signed: bool, d: int) -> Seq:
    base = sdiv(d) if signed else udiv(d)
    return with_rem(base, d) if op == "rem" else base


def to_sass(seq: Seq, xreg: str = "R0") -> list[str]:
    """Render as SASS text (registers R2.. allocated in order)."""
    names, nxt = {"x": xreg}, [2]

    def reg(n):
        if n not in names:
            names[n] = f"R{nxt[0]}"
            nxt[0] += 1
        return names[n]

    def o(v):
        return f"{v:#x}" if isinstance(v, int) else names[v]

    out = []
    for op in seq.ops:
        k = op[0]
        srcs = [o(v) for v in op[2:]]
        d = reg(op[1])
        out.append({
            "mov": lambda: f"MOV {d}, {srcs[0]}",
            "mulhi_u": lambda: f"IMAD.HI.U32 {d}, {srcs[0]}, {srcs[1]}, RZ",
            "mulhi_s": lambda: f"IMAD.HI {d}, {srcs[0]}, {srcs[1]}, RZ",
            "shr_u": lambda: f"SHF.R.U32.HI {d}, RZ, {op[3]:#x}, {srcs[0]}",
            "shr_s": lambda: f"SHF.R.S32.HI {d}, RZ, {op[3]:#x}, {srcs[0]}",
            "add": lambda: f"IADD3 {d}, {srcs[0]}, {srcs[1]}, RZ",
            "sub": lambda: f"IADD3 {d}, {srcs[0]}, -{srcs[1]}, RZ",
            "lea_hi_sign": lambda: f"LEA.HI {d}, {srcs[1]}, {srcs[0]}, RZ, 0x1",
            "mad": lambda: f"IMAD {d}, {srcs[0]}, {srcs[1]}, {srcs[2]}",
            "geu": lambda: f"ISETP.GE.U32.AND P0, PT, {srcs[0]}, {srcs[1]}, PT ; SEL {d}, RZ, 0x1, !P0",
        }[k]())
    return out
