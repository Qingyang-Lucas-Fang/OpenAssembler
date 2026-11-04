"""Load and verify bilinear matrix-multiplication schemes (rank-R tensor decompositions).

A scheme for <n, m, p> (A: n x m, B: m x p, C = A B: n x p) is three integer/rational
matrices u [n*m, R], v [m*p, R], w [n*p, R]:

    M_r = (sum_{ij} u[i*m+j, r] A_ij) * (sum_{jk} v[j*p+k, r] B_jk)        r = 0..R-1
    C_ik = sum_r w[?, r] M_r

AlphaTensor stores w for the *transposed* product (index k*n+i, see
alphatensor/benchmarking/utils.py: "factors[2] = factors[2].transpose(1, 0, 2)").
verify() checks the exact tensor identity over the rationals, and load() normalizes w to
row-major C (index i*p+k).

Sources (third_party/alphatensor, fetched from github.com/google-deepmind/alphatensor):
  algorithms/factorizations_r.npz        93 schemes over the reals (int / dyadic coefs)
  benchmarking/factorizations.py         the V100-tuned 4x4x4 rank-49 scheme, Strassen^2
The .npz holds pickled numpy object arrays; tools/fastmm_import.py converts it once into
specs/fastmm/schemes.json.gz (sparse, stdlib-readable).
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
AT = ROOT / "third_party" / "alphatensor"


@dataclass
class Scheme:
    name: str
    n: int
    m: int
    p: int
    u: list      # [n*m][R] Fraction
    v: list      # [m*p][R]
    w: list      # [n*p][R]  row-major C
    source: str

    @property
    def rank(self) -> int:
        return len(self.u[0])

    def naive(self) -> int:
        return self.n * self.m * self.p

    def additions(self) -> tuple[int, int, int]:
        """Nonzeros-minus-one per product for the A sums, B sums, and C accumulation:
        the scalar/block additions a direct (no CSE) evaluation needs."""
        R = self.rank
        a = sum(max(0, sum(1 for row in self.u if row[r]) - 1) for r in range(R))
        b = sum(max(0, sum(1 for row in self.v if row[r]) - 1) for r in range(R))
        c = sum(max(0, sum(1 for x in row if x) - 1) for row in self.w)
        return a, b, c


SPEC = ROOT / "specs" / "fastmm" / "schemes.json.gz"   # written by tools/fastmm_import.py


def load_spec(path: Path = SPEC) -> dict[str, Scheme]:
    import gzip
    import json

    out = {}
    with gzip.open(path, "rt") as f:
        raw = json.load(f)
    for name, d in raw.items():
        n, m, p, R = d["n"], d["m"], d["p"], d["rank"]

        def dense(nz, rows):
            M = [[Fraction(0)] * R for _ in range(rows)]
            for i, r, x in nz:
                M[i][r] = Fraction(x)
            return M
        out[name] = Scheme(name, n, m, p, dense(d["u"], n * m), dense(d["v"], m * p),
                           dense(d["w"], n * p), d["source"])
    return out


def strassen() -> Scheme:
    """Strassen 1969, <2,2,2> rank 7, written in this module's convention."""
    U = [[1, 0, 1, 0, 1, -1, 0], [0, 0, 0, 0, 1, 0, 1], [0, 1, 0, 0, 0, 1, 0], [1, 1, 0, 1, 0, 0, -1]]
    V = [[1, 1, 0, -1, 0, 1, 0], [0, 0, 1, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0, 1], [1, 0, -1, 0, 1, 0, 1]]
    W = [[1, 0, 0, 1, -1, 0, 1], [0, 0, 1, 0, 1, 0, 0], [0, 1, 0, 1, 0, 0, 0], [1, -1, 1, 0, 0, 1, 0]]
    f = lambda M: [[Fraction(x) for x in r] for r in M]  # noqa: E731
    return Scheme("strassen_2x2x2", 2, 2, 2, f(U), f(V), f(W), "Strassen 1969")


def naive(n: int, m: int, p: int) -> Scheme:
    R = n * m * p
    u = [[Fraction(0)] * R for _ in range(n * m)]
    v = [[Fraction(0)] * R for _ in range(m * p)]
    w = [[Fraction(0)] * R for _ in range(n * p)]
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u[i * m + j][r] = v[j * p + k][r] = w[i * p + k][r] = Fraction(1)
                r += 1
    return Scheme(f"naive_{n}x{m}x{p}", n, m, p, u, v, w, "definition")


def verify(s: Scheme) -> bool:
    """Exact check of sum_r u[a,r] v[b,r] w[c,r] == T[a,b,c] for every (a,b,c)."""
    n, m, p, R = s.n, s.m, s.p, s.rank
    nzu = [[(a, s.u[a][r]) for a in range(n * m) if s.u[a][r]] for r in range(R)]
    nzv = [[(b, s.v[b][r]) for b in range(m * p) if s.v[b][r]] for r in range(R)]
    nzw = [[(c, s.w[c][r]) for c in range(n * p) if s.w[c][r]] for r in range(R)]
    acc = {}
    for r in range(R):
        for a, x in nzu[r]:
            for b, y in nzv[r]:
                xy = x * y
                for c, z in nzw[r]:
                    acc[(a, b, c)] = acc.get((a, b, c), 0) + xy * z
    want = {(i * m + j, j * p + k, i * p + k): 1 for i in range(n) for j in range(m) for k in range(p)}
    keys = set(acc) | set(want)
    return all(acc.get(t, 0) == want.get(t, 0) for t in keys)


def all_schemes() -> dict[str, Scheme]:
    out = {"strassen_2x2x2": strassen()}
    if SPEC.exists():
        out.update(load_spec())
    return out
