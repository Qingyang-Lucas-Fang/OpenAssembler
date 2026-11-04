"""Finite-input references with explicit transform/output quantization."""
from __future__ import annotations

import math
import random
import struct

from openassembler.fastmm.factors import naive, strassen


def f16(x):
    return struct.unpack('<e', struct.pack('<e', x))[0]


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def inputs(m, n, family):
    rng = random.Random(8191)
    if family == 'integer':
        return ([rng.randrange(-3,4) for _ in range(m*32)],
                [rng.randrange(-3,4) for _ in range(n*32)])
    if family == 'floating':
        return ([f16(rng.uniform(-1,1)) for _ in range(m*32)],
                [f16(rng.uniform(-1,1)) for _ in range(n*32)])
    if family == 'cancellation':
        # Adjacent quadrants nearly cancel in some compile-time transforms.
        av = [f16(rng.uniform(-2,2)) for _ in range(m*32)]
        bv = [f16(rng.uniform(-2,2)) for _ in range(n*32)]
        for values, rows in ((av,m),(bv,n)):
            for row in range(rows):
                for q in range(16):
                    values[row*32+16+q] = f16(-values[row*32+q]+(2**-10 if q&1 else 0))
        return av,bv
    raise ValueError(family)


def reference(av, bv, m, n, scheme_name):
    """FP16 transforms; exact finite products; FP32 product/output rounding.

    A length-16 dot of FP16 values is evaluated in Python binary64 and then
    rounded once to FP32. This oracle deliberately does not assume an exact
    hardware HMMA internal accumulation order; floating checks use tolerance.
    The integer family is exactly representable throughout and is bit-exact.
    """
    scheme = strassen() if scheme_name == 'strassen' else naive(2,2,2)
    out = [0.0]*(m*n)
    for bm in range(m//32):
        for bn in range(n//32):
            aq = [[[av[(bm*32+(q//2)*16+r)*32+(q%2)*16+k]
                    for k in range(16)] for r in range(16)] for q in range(4)]
            # B factor blocks are [K,N]; physical storage is [N,K].
            bq = [[[bv[(bn*32+(q%2)*16+c)*32+(q//2)*16+k]
                    for k in range(16)] for c in range(16)] for q in range(4)]
            cq = [[[0.0]*16 for _ in range(16)] for _ in range(4)]
            for product in range(scheme.rank):
                at = [[f16(sum(float(row[product])*aq[q][r][k]
                               for q,row in enumerate(scheme.u) if row[product]))
                       for k in range(16)] for r in range(16)]
                bt = [[f16(sum(float(row[product])*bq[q][c][k]
                               for q,row in enumerate(scheme.v) if row[product]))
                       for k in range(16)] for c in range(16)]
                pp = [[f32(sum(at[r][k]*bt[c][k] for k in range(16)))
                       for c in range(16)] for r in range(16)]
                for q,row in enumerate(scheme.w):
                    coefficient = int(row[product])
                    if coefficient:
                        for r in range(16):
                            for c in range(16):
                                cq[q][r][c] = f32(cq[q][r][c]+coefficient*pp[r][c])
            for q in range(4):
                for r in range(16):
                    for c in range(16):
                        out[(bm*32+(q//2)*16+r)*n+bn*32+(q%2)*16+c] = cq[q][r][c]
    return out


def direct_reference(av,bv,m,n):
    return [sum(av[r*32+k]*bv[c*32+k] for k in range(32))
            for r in range(m) for c in range(n)]


def compare(actual,expected,integer=False):
    errors = [abs(x-y) for x,y in zip(actual,expected)]
    bad = [i for i,(x,y) in enumerate(zip(actual,expected))
           if not math.isfinite(x) or abs(x-y) > (0 if integer else 2e-5+2e-6*abs(y))]
    return {'passed':len(actual)==len(expected) and not bad,'values_checked':len(expected),
            'mismatches':len(bad),'examples':[(i,actual[i],expected[i]) for i in bad[:8]],
            'max_absolute_error':max(errors,default=0),
            'atol':0 if integer else 2e-5,'rtol':0 if integer else 2e-6}
