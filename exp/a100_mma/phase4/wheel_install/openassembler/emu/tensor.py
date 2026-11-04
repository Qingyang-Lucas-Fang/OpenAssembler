"""Warp-level matrix ops: HMMA / IMMA / DMMA, LDSM, LDGSTS.

Fragment layouts are the PTX ISA's (9.7.16.5.x, "Matrix Fragments for mma.mMnNkK").
With groupID = lane >> 2 and tig = lane % 4:

  m16n8k16 f16/bf16  A[16x16]: a_i  row = g (+8 if i in {2,3,6,7})  col = 2*tig + (i&1) (+8 if i >= 4)
                     B[16x8]:  b_i  k = 2*tig + (i&1) (+8 if i >= 2)  n = g
  m16n8k8  tf32      A[16x8]:  a_i  row = g (+8 if i odd)            col = tig (+4 if i >= 2)
                     B[8x8]:   b_i  k = tig (+4 if i == 1)            n = g
  m16n8k32 s8/u8     A[16x32]: a_i  row = g (+8 if i in 4..7 or 12..15)  col = 4*tig + (i&3) (+16 if i >= 8)
                     B[32x8]:  b_i  k = 4*tig + (i&3) (+16 if i >= 4)    n = g
  m8n8k4   f64       A[8x4]:   row = g, col = tig;  B[4x8]: k = tig, n = g
  C/D (all m16n8):   c_i  row = g (+8 if i >= 2)  col = 2*tig + (i&1)
  C/D (m8n8k4 f64):  c_i  row = g, col = 2*tig + i

Accumulation numerics are a *model* (MMA_MODEL below) whose parameters are fitted
to A100 outputs by tools/fit_mma.py; the defaults are the fitted values.
"""

from __future__ import annotations

from fractions import Fraction

from openassembler.emu import fp
from openassembler.emu.machine import M32, Unsupported
from openassembler.emu.operands import Reg

# Fitted on A100 (corpus/sm_80/gpu_a100_40gb_tensor.raw.json.gz, corpus/.tmp/fit_nominal.py):
# 1024/1024 lanes bit-exact for each type; the next-best model in the family scores ~70%.
#   per step of `block` products: every product a*b is aligned at its NOMINAL exponent
#   e(a)+e(b) (not renormalized), together with the running f32 accumulator; each term
#   is truncated toward zero `bits` bits below the largest exponent, summed exactly,
#   and the step result is rounded to f32 toward zero.
# Only finite normal inputs were measured: inf/NaN/subnormal handling is a HYPOTHESIS.
MMA_MODEL = {
    "f16": {"block": 8, "bits": 24, "round": "RZ"},
    "bf16": {"block": 8, "bits": 24, "round": "RZ"},
    "tf32": {"block": 4, "bits": 24, "round": "RZ"},
}


def _exp2(q: Fraction) -> int:
    """floor(log2(q)) for q > 0."""
    e = q.numerator.bit_length() - q.denominator.bit_length()
    if Fraction(2) ** e > q:
        e -= 1
    return e


def aligned_sum(terms: list[Fraction], bits: int, trunc: str = "rz") -> Fraction:
    """Sum after aligning every term to the largest one and dropping bits more than
    `bits` below its leading bit. trunc="rz": sign-magnitude truncation (toward 0);
    trunc="floor": two's-complement right shift (toward -inf)."""
    nz = [t for t in terms if t != 0]
    if not nz:
        return Fraction(0)
    emax = max(_exp2(abs(t)) for t in nz)
    scale = Fraction(2) ** (bits - emax)
    tot = 0
    for t in nz:
        v = t * scale
        if trunc == "floor" or v >= 0:
            tot += v.numerator // v.denominator
        else:
            tot -= (-v).numerator // (-v).denominator
    return Fraction(tot) / scale


def dot_f32(pairs: list[tuple[fp.Val, fp.Val]], c: fp.Val, model: dict) -> int:
    """D = sum(a*b for a, b in pairs) + c as f32 bits, using the fitted model."""
    vals = [x for p in pairs for x in p] + [c]
    if any(v.nan for v in vals):
        return fp.CANON_NAN32
    prods = [fp.mul(a, b) for a, b in pairs]
    infs = [v for v in prods + [c] if v.inf or v.nan]
    if infs:  # HYPOTHESIS: IEEE-like specials (0*inf -> NaN, inf-inf -> NaN)
        if any(v.nan for v in infs) or len({v.neg for v in infs}) > 1:
            return fp.CANON_NAN32
        return 0xFF800000 if infs[0].neg else 0x7F800000
    acc = fp.signed_frac(c)
    for i in range(0, len(pairs), model["block"]):
        terms = [(fp.signed_frac(a) * fp.signed_frac(b), _exp2(a.q) + _exp2(b.q))
                 for a, b in pairs[i:i + model["block"]] if a.q and b.q]
        if acc:
            terms.append((acc, _exp2(abs(acc))))
        if not terms:
            continue
        scale = Fraction(2) ** (model["bits"] - max(e for _, e in terms))
        tot = 0
        for t, _ in terms:
            v = t * scale
            tot += v.numerator // v.denominator if v >= 0 else -((-v).numerator // (-v).denominator)
        r = fp.decode(fp.encode(fp.from_frac(Fraction(tot) / scale), fp.F32, model["round"]))
        if r.inf:
            return fp.encode(r)
        acc = fp.signed_frac(r)
    return fp.encode(fp.from_frac(acc), fp.F32, "RN")


# ---- fragment index maps (lane, element i) -> (row, col) -------------------------------

def a_rc_k16(lane, i):
    g, t = lane >> 2, lane & 3
    return g + (8 if i in (2, 3, 6, 7) else 0), 2 * t + (i & 1) + (8 if i >= 4 else 0)


def b_rc_k16(lane, i):
    g, t = lane >> 2, lane & 3
    return 2 * t + (i & 1) + (8 if i >= 2 else 0), g


def a_rc_k8_tf32(lane, i):
    g, t = lane >> 2, lane & 3
    return g + (8 if i & 1 else 0), t + (4 if i >= 2 else 0)


def b_rc_k8_tf32(lane, i):
    g, t = lane >> 2, lane & 3
    return t + (4 if i == 1 else 0), g


def a_rc_k32_i8(lane, i):
    g, t = lane >> 2, lane & 3
    return g + (8 if (4 <= i < 8 or i >= 12) else 0), 4 * t + (i & 3) + (16 if i >= 8 else 0)


def b_rc_k32_i8(lane, i):
    g, t = lane >> 2, lane & 3
    return 4 * t + (i & 3) + (16 if i >= 4 else 0), g


def c_rc(lane, i):
    g, t = lane >> 2, lane & 3
    return g + (8 if i >= 2 else 0), 2 * t + (i & 1)


def _regs(op: Reg, n: int) -> list[int]:
    return [255 if op.idx == 255 else op.idx + k for k in range(n)]


def _rd(th, idx):
    return 0 if idx == 255 else th.r[idx]


def mma(m, wp, active, insn):
    """HMMA.16816.F32[.BF16] / HMMA.1688.F32.TF32 / IMMA.16832.S8.S8 / DMMA.884."""
    if len(active) != 32:
        raise Unsupported("MMA needs a full, converged warp")
    by = {th.lane: th for th in active}
    d, a, b, c = insn.ops
    op, mods = insn.op, insn.mods
    shape = mods[0]
    if op == "HMMA" and shape == "16816" and "F32" in mods:
        elt = fp.BF16 if "BF16" in mods else fp.F16
        M, N, K = 16, 8, 16
        na, nb, nc = 4, 2, 4
        def unpack(regs, n):  # 2 halves per register, low half = lower element index
            out = []
            for r in regs:
                out += [r & 0xFFFF, r >> 16]
            return out[:n]
        A = [[None] * K for _ in range(M)]
        B = [[None] * N for _ in range(K)]
        for ln, th in by.items():
            av = unpack([_rd(th, r) for r in _regs(a, na)], 8)
            bv = unpack([_rd(th, r) for r in _regs(b, nb)], 4)
            for i, h in enumerate(av):
                r_, c_ = a_rc_k16(ln, i)
                A[r_][c_] = fp.decode(h, elt)
            for i, h in enumerate(bv):
                r_, c_ = b_rc_k16(ln, i)
                B[r_][c_] = fp.decode(h, elt)
        model = MMA_MODEL["bf16" if elt is fp.BF16 else "f16"]
    elif op == "HMMA" and shape == "1688" and "TF32" in mods:
        M, N, K = 16, 8, 8
        na, nb, nc = 4, 2, 4
        A = [[None] * K for _ in range(M)]
        B = [[None] * N for _ in range(K)]
        tf = lambda x: fp.decode(x & 0xFFFFE000)  # tf32: top 19 bits of an f32  # noqa: E731
        for ln, th in by.items():
            for i, r in enumerate(_regs(a, na)):
                r_, c_ = a_rc_k8_tf32(ln, i)
                A[r_][c_] = tf(_rd(th, r))
            for i, r in enumerate(_regs(b, nb)):
                r_, c_ = b_rc_k8_tf32(ln, i)
                B[r_][c_] = tf(_rd(th, r))
        model = MMA_MODEL["tf32"]
    elif op == "IMMA" and shape == "16832":
        return _imma(m, wp, by, insn)
    elif op == "DMMA" and shape == "884":
        return _dmma(m, wp, by, insn)
    else:
        raise Unsupported(insn.text)
    C = {}
    for ln, th in by.items():
        for i, r in enumerate(_regs(c, nc)):
            C[c_rc(ln, i)] = fp.decode(_rd(th, r))
    D = {}
    for i in range(M):
        for j in range(N):
            D[(i, j)] = dot_f32([(A[i][k], B[k][j]) for k in range(K)], C[(i, j)], model)
    for ln, th in by.items():
        for i, r in enumerate(_regs(d, nc)):
            if r != 255:
                th.r[r] = D[c_rc(ln, i)]


def _imma(m, wp, by, insn):
    d, a, b, c = insn.ops
    sa = "S8" == insn.mods[1]
    sb = "S8" == insn.mods[2]
    sx = lambda v, s: v - 256 if s and v >= 128 else v  # noqa: E731
    A = [[0] * 32 for _ in range(16)]
    B = [[0] * 8 for _ in range(32)]
    for ln, th in by.items():
        av = []
        for r in _regs(a, 4):
            v = _rd(th, r)
            av += [(v >> (8 * k)) & 0xFF for k in range(4)]
        bv = []
        for r in _regs(b, 2):
            v = _rd(th, r)
            bv += [(v >> (8 * k)) & 0xFF for k in range(4)]
        for i, x in enumerate(av):
            r_, c_ = a_rc_k32_i8(ln, i)
            A[r_][c_] = sx(x, sa)
        for i, x in enumerate(bv):
            r_, c_ = b_rc_k32_i8(ln, i)
            B[r_][c_] = sx(x, sb)
    C = {}
    for ln, th in by.items():
        for i, r in enumerate(_regs(c, 4)):
            v = _rd(th, r)
            C[c_rc(ln, i)] = v - (1 << 32) if v >> 31 else v
    for ln, th in by.items():
        for i, r in enumerate(_regs(d, 4)):
            row, col = c_rc(ln, i)
            s = C[(row, col)] + sum(A[row][k] * B[k][col] for k in range(32))
            if "SAT" in insn.mods:
                s = max(-(1 << 31), min((1 << 31) - 1, s))
            if r != 255:
                th.r[r] = s & M32


def _dmma(m, wp, by, insn):
    d, a, b, c = insn.ops
    rd64 = lambda th, op, k=0: _rd(th, op.idx + 2 * k) | (_rd(th, op.idx + 2 * k + 1) << 32)  # noqa: E731
    A = [[None] * 4 for _ in range(8)]
    B = [[None] * 8 for _ in range(4)]
    for ln, th in by.items():
        g, t = ln >> 2, ln & 3
        A[g][t] = fp.decode(rd64(th, a), fp.F64)
        B[t][g] = fp.decode(rd64(th, b), fp.F64)
    C = {}
    for ln, th in by.items():
        g, t = ln >> 2, ln & 3
        for i in range(2):
            C[(g, 2 * t + i)] = fp.decode(rd64(th, c, i), fp.F64)
    for ln, th in by.items():
        g, t = ln >> 2, ln & 3
        for i in range(2):
            col = 2 * t + i
            # HYPOTHESIS: DMMA is a chain of IEEE fp64 FMAs in k order (k = 0..3).
            acc = C[(g, col)]
            for k in range(4):
                v = fp.fma(A[g][k], B[k][col], acc, "RN")
                acc = fp.decode(fp.encode(v, fp.F64), fp.F64)
            bits = fp.encode(acc, fp.F64)
            r = d.idx + 2 * i
            th.r[r], th.r[r + 1] = bits & M32, bits >> 32


# ---- LDSM (ldmatrix) -----------------------------------------------------------------------

def ldsm(m, wp, active, insn):
    """LDSM.16.{M88,MT88}.{1,2,4} Rd, [Ra+off]  (PTX ldmatrix.sync.aligned.m8n8.xN[.trans].b16).
    Matrix j's 8 row addresses come from lanes 8j..8j+7. Each lane receives, for every
    matrix j, one 32-bit register: elements (row = lane>>2, col = 2*(lane&3) + {0,1});
    with .trans (MT88) the matrix is transposed first."""
    if len(active) != 32:
        raise Unsupported("LDSM needs a full warp")
    by = {th.lane: th for th in active}
    d, addr = insn.ops
    n = int(insn.mods[-1])
    trans = "MT88" in insn.mods
    rows = {}
    for j in range(n):
        for r in range(8):
            th = by[8 * j + r]
            base = m.addr(th, wp, addr)
            rows[(j, r)] = [m.smem.read(base + 2 * c, 2) for c in range(8)]
    for ln, th in by.items():
        g, t = ln >> 2, ln & 3
        for j in range(n):
            if trans:
                lo, hi = rows[(j, 2 * t)][g], rows[(j, 2 * t + 1)][g]
            else:
                lo, hi = rows[(j, g)][2 * t], rows[(j, g)][2 * t + 1]
            if d.idx + j != 255:
                th.r[d.idx + j] = lo | (hi << 16)


# ---- LDGSTS (cp.async) ---------------------------------------------------------------------

def ldgsts(m, th, wp, insn):
    """LDGSTS.E[.BYPASS][.64|.128] [Rs], [Rg.64]: global -> shared copy. Modeled as
    synchronous (it completes before LDGDEPBAR/DEPBAR, which are no-ops here)."""
    dst, src = insn.ops[0], insn.ops[1]
    n = 16 if "128" in insn.mods else 8 if "64" in insn.mods else 4
    g = m.addr(th, wp, src, wide=True)
    s = m.addr(th, wp, dst)
    if "ZFILL" in insn.mods:
        raise Unsupported("LDGSTS.ZFILL")
    m.smem.write(s, n, m.gmem.read(g, n))
