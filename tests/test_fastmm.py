"""Fast matmul schemes, plans, and the compiler features they rely on."""

import random
import struct
from fractions import Fraction

import pytest

from openassembler.backend.compile import PRED_TO_GUARD, Control, _branch_pred_latency, hw_latency
from openassembler.fastmm import factors as F
from openassembler.fastmm import plan as P


def test_strassen_and_naive_verify():
    assert F.verify(F.strassen())
    assert F.verify(F.naive(2, 3, 4))
    s = F.strassen()
    s.w[0][0] += 1              # break it
    assert not F.verify(s)


@pytest.mark.skipif(not F.SPEC.exists(), reason="specs/fastmm/schemes.json.gz not imported")
def test_alphatensor_schemes_verify():
    S = F.all_schemes()
    for name in ("at_2x2x2", "at_3x3x3", "at_gpu_4x4x4", "strassen2_4x4x4", "at_tpu_4x4x4"):
        assert F.verify(S[name]), name
    assert S["at_gpu_4x4x4"].rank == 49 and S["at_3x3x3"].rank == 23


def _apply(plan, n, A, B):
    """Execute a plan on the host (exact rational arithmetic) to check its structure."""
    mem = {"A": A, "B": B}
    for name, size in plan.buffers.items():
        mem.setdefault(name, [Fraction(0)] * (size // 4))

    def view(b, off, ld, rows, cols):
        return [[mem[b][off + r * ld + c] for c in range(cols)] for r in range(rows)]

    for mname, kname, grid, block, smem, args in plan.steps:
        if mname == "@memset":
            mem[args[0][1]] = [Fraction(0)] * len(mem[args[0][1]])
            continue
        if mname.startswith("lincomb"):
            k = int(mname[7:])
            ob, oo = args[0][1], args[0][2] // 4
            old = args[1][1]
            ins = [(args[2 + 3 * j][1], args[2 + 3 * j][2] // 4, args[3 + 3 * j][1],
                    struct.unpack("<f", struct.pack("<I", args[4 + 3 * j][1]))[0]) for j in range(k)]
            cols, rows = args[-1][1], grid[1]
            vals = [[sum(Fraction(c) * mem[b][o + r * ld + q] for b, o, ld, c in ins) for q in range(cols)]
                    for r in range(rows)]
            for r in range(rows):
                for q in range(cols):
                    mem[ob][oo + r * old + q] = vals[r][q]
            continue
        if mname.startswith("sgemm"):
            (_, ab, ao), (_, bb, bo), (_, cb, co) = args[:3]
            lda, ldb, ldc, K = (a[1] for a in args[3:])
            t = int(mname[5:])
            M, N = grid[1] * t, grid[0] * t
            for i in range(M):
                for j in range(N):
                    mem[cb][co // 4 + i * ldc + j] = sum(mem[ab][ao // 4 + i * lda + k] * mem[bb][bo // 4 + k * ldb + j]
                                                         for k in range(K))
            continue
        raise AssertionError(mname)
    return mem["C"]


@pytest.mark.parametrize("scheme", [F.strassen(), F.naive(2, 2, 2)])
def test_plan_structure_exact(scheme, monkeypatch):
    monkeypatch.setattr(P, "GEMM", "sgemm2")    # tiny tiles so the host check is fast
    n = 16
    rng = random.Random(0)
    A = [Fraction(rng.randint(-3, 3)) for _ in range(n * n)]
    B = [Fraction(rng.randint(-3, 3)) for _ in range(n * n)]
    pl = P.build(scheme, n)
    C = _apply(pl, n, A, B)
    for i in range(n):
        for j in range(n):
            assert C[i * n + j] == sum(A[i * n + k] * B[k * n + j] for k in range(n))


def test_measured_latencies():
    assert hw_latency("IADD3", "IADD3") == 4 and hw_latency("IMAD", "IADD3") == 5
    assert hw_latency("FFMA", "FFMA") == 4 and hw_latency("IADD3", "FFMA") == 5


def test_guard_needs_13_cycles():
    c = Control(stall=1, yield_=False, wbar=7, rbar=7, wait=0, reuse=0)
    out = _branch_pred_latency([("ISETP.LT.U32.AND P0, PT, R0, R1, PT", c), ("@P0 IADD3 R2, R2, 0x1, RZ", c)])
    assert out[0][1].stall >= PRED_TO_GUARD and out[0][1].yield_    # stall > 11 needs yield


def test_multiwarp_barrier():
    """Two warps exchange values through shared memory across BAR.SYNC."""
    from openassembler.emu.machine import Launch, Machine, Memory
    from openassembler.emu.program import load_listing

    lines = ["S2R R0, SR_TID.X", "SHF.L.U32 R1, R0, 0x2, RZ", "STS [R1], R0",
             "BAR.SYNC.DEFER_BLOCKING 0x0", "LOP3.LUT R2, R0, 0x20, RZ, 0x3c, !PT",
             "SHF.L.U32 R2, R2, 0x2, RZ", "LDS R3, [R2]", "STS [R1+0x100], R3", "EXIT"]
    sass = "\n".join(f"        /*{16 * i:04x}*/ {t} ; /* 0x{0:016x} */\n        /* 0x{0:016x} */"
                     for i, t in enumerate(lines))
    m = Machine(load_listing(sass), Launch(block=(64, 1, 1)), Memory())
    m.run()
    for t in range(64):
        assert m.smem.read(0x100 + 4 * t, 4) == t ^ 32
