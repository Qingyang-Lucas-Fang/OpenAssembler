"""Fit the tensor-core accumulation model with NOMINAL-exponent alignment (the model
that matches the A100 on 100% of lanes; see openassembler/emu/tensor.py MMA_MODEL).

    PYTHONPATH=. uv run --no-project python tools/fit_mma_nominal.py
"""
import gzip, json, sys, itertools
sys.path.insert(0,'tools'); sys.path.insert(0,'.')
import fit_mma as F
from fractions import Fraction
from openassembler.emu import fp
from openassembler.emu.tensor import _exp2
raw=json.load(gzip.open('corpus/sm_80/gpu_a100_40gb_tensor.raw.json.gz','rt'))['probes']

def nom(v):  # nominal exponent of a decoded value (normal numbers)
    return _exp2(v.q) if v.q else None

def dot(pairs, c, block, bits, rnd, trunc):
    # pairs: list of (a, b) decoded values; terms carry nominal exponent ea+eb
    if any(x.nan or x.inf for p in pairs for x in p) or c.nan or c.inf: return None
    acc, acc_e = fp.signed_frac(c), nom(c)
    for i in range(0, len(pairs), block):
        terms = []
        for a, b in pairs[i:i+block]:
            if a.q == 0 or b.q == 0: continue
            terms.append((fp.signed_frac(a) * fp.signed_frac(b), nom(a) + nom(b)))
        if acc != 0: terms.append((acc, _exp2(abs(acc))))
        if not terms:
            acc = Fraction(0); continue
        emax = max(e for _, e in terms)
        scale = Fraction(2) ** (bits - emax)
        tot = 0
        for t, _ in terms:
            v = t * scale
            tot += (v.numerator // v.denominator) if (trunc == 'floor' or v >= 0) else -((-v).numerator // (-v).denominator)
        s = Fraction(tot) / scale
        acc = fp.signed_frac(fp.decode(fp.encode(fp.from_frac(s), fp.F32, rnd)))
    return fp.encode(fp.from_frac(acc), fp.F32, 'RN')

for probe, blocks, tfm in (('mma.f16', (8, 16), '-'), ('mma.bf16', (8, 16), '-'), ('mma.tf32', (4, 8), 'trunc')):
    rec = raw[probe]; cases = []
    for (inputs, _k, _kf), gpu in zip(rec['trials'], rec['out']):
        A, B, C, K = F.matrices(probe, inputs, tfm)
        for lane in range(32):
            for slot, idx in ((0, 0), (1, 3)):
                r, c = F.c_rc(lane, idx)
                cases.append(([(A[r][k], B[k][c]) for k in range(K)], C[(r, c)], gpu[lane][slot]))
    best = []
    for block, bits, rnd, trunc in itertools.product(blocks, range(22, 28), ('RZ', 'RN'), ('rz', 'floor')):
        hit = n = 0
        for pairs, c, g in cases:
            m = dot(pairs, c, block, bits, rnd, trunc)
            if m is None: continue
            n += 1; hit += m == g
        best.append((hit / n, hit, n, block, bits, rnd, trunc))
    best.sort(reverse=True)
    print(probe, [f"{a:.2%} blk={b} bits={bi} {r} {t}" for a, _, _, b, bi, r, t in best[:3]])
