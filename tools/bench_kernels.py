"""Benchmark kernels (probe ABI: out, in, k, kf) for timing ours vs ptxas on GPUs.

Each kernel loops `k` times (k is the kernel param), so run time is dominated by the
loop body and is easy to compare. Every kernel stores its result so nothing is dead.

    BENCH = {name: ptx_body}  (body inserted into tools/ptx_probes.KERNEL)
"""

from __future__ import annotations

# %r1..%r4: inputs, %rk: trip count (param k), result must end in %r10/%r11.
BENCH = {
    # dependent integer chain: latency-bound (1 warp in flight per SM partition)
    "chain.imad_lop3": (
        " mov.u32 %r10, 0; mov.u32 %r20, %r1;"
        " LOOP: mad.lo.u32 %r20, %r20, %r2, %r3; xor.b32 %r20, %r20, %r4;"
        " add.u32 %r10, %r10, 1; setp.lt.u32 %p1, %r10, %rk; @%p1 bra LOOP;"
        " mov.u32 %r11, %r20;"),
    # 4 independent chains: throughput-bound (ILP inside one thread)
    "ilp4.imad": (
        " mov.u32 %r10, 0; mov.u32 %r20, %r1; mov.u32 %r21, %r2; mov.u32 %r22, %r3; mov.u32 %r23, %r4;"
        " LOOP: mad.lo.u32 %r20, %r20, %r2, %r3; mad.lo.u32 %r21, %r21, %r3, %r4;"
        " mad.lo.u32 %r22, %r22, %r4, %r1; mad.lo.u32 %r23, %r23, %r1, %r2;"
        " add.u32 %r10, %r10, 1; setp.lt.u32 %p1, %r10, %rk; @%p1 bra LOOP;"
        " xor.b32 %r20, %r20, %r21; xor.b32 %r22, %r22, %r23; xor.b32 %r11, %r20, %r22;"),
    # FP32 FMA chain
    "chain.ffma": (
        " mov.u32 %r10, 0; mov.f32 %f20, %f1;"
        " LOOP: fma.rn.f32 %f20, %f20, %f2, %f3;"
        " add.u32 %r10, %r10, 1; setp.lt.u32 %p1, %r10, %rk; @%p1 bra LOOP;"
        " mov.b32 %r11, %f20;"),
    # 4 independent FP32 FMA chains
    "ilp4.ffma": (
        " mov.u32 %r10, 0; mov.f32 %f20, %f1; mov.f32 %f21, %f2; mov.f32 %f22, %f3; mov.f32 %f23, %f1;"
        " LOOP: fma.rn.f32 %f20, %f20, %f2, %f3; fma.rn.f32 %f21, %f21, %f3, %f1;"
        " fma.rn.f32 %f22, %f22, %f1, %f2; fma.rn.f32 %f23, %f23, %f2, %f1;"
        " add.u32 %r10, %r10, 1; setp.lt.u32 %p1, %r10, %rk; @%p1 bra LOOP;"
        " add.f32 %f20, %f20, %f21; add.f32 %f22, %f22, %f23; add.f32 %f20, %f20, %f22; mov.b32 %r11, %f20;"),
    # integer division by a constant in a loop (our magic-number lowering vs ptxas's reciprocal)
    "loop.div7": (
        " mov.u32 %r10, 0; mov.u32 %r20, 0; mov.u32 %r21, %r1;"
        " LOOP: div.u32 %r22, %r21, 7; add.u32 %r20, %r20, %r22; add.u32 %r21, %r21, %r2;"
        " add.u32 %r10, %r10, 1; setp.lt.u32 %p1, %r10, %rk; @%p1 bra LOOP;"
        " mov.u32 %r11, %r20;"),
    "loop.rem1000": (
        " mov.u32 %r10, 0; mov.u32 %r20, 0; mov.u32 %r21, %r1;"
        " LOOP: rem.u32 %r22, %r21, 1000; add.u32 %r20, %r20, %r22; add.u32 %r21, %r21, %r2;"
        " add.u32 %r10, %r10, 1; setp.lt.u32 %p1, %r10, %rk; @%p1 bra LOOP;"
        " mov.u32 %r11, %r20;"),
    # shared-memory round trip in a loop (STS -> BAR -> LDS)
    "loop.smem": (
        " mov.u32 %r10, 0; mov.u32 %r20, %r1; mov.u32 %r30, smem; shl.b32 %r31, %r0, 2; add.u32 %r30, %r30, %r31;"
        " LOOP: st.shared.u32 [%r30], %r20; bar.sync 0; ld.shared.u32 %r21, [%r30]; add.u32 %r20, %r21, %r2;"
        " add.u32 %r10, %r10, 1; setp.lt.u32 %p1, %r10, %rk; @%p1 bra LOOP;"
        " mov.u32 %r11, %r20;"),
}
SHARED = {"loop.smem"}
