"""Full-tile SM80 RCR GEMM with an explicitly synchronized tensor pipeline.

The default schedule uses a 128x128x64 CTA tile, four 64x64 warp tiles,
FP16/BF16 inputs, FP32 accumulation/output, and two shared-memory stages.
It combines paired B-fragment LDSM x4 loads, shared-copy displacements,
constant-stage loop pairs and an aligned vector output epilogue. A/B are
contiguous MxK/NxK row-major arrays; C is contiguous MxN and equals A @ B.T.

This module only generates PTX. Native tensor instruction selection, direct
64-bit pointer updates and scheduling belong to the standard tensor backend.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GemmPlan:
    """Validated full-tile GEMM contract; boundary tiles are not supported.

    ``stagepair`` is the measured default. ``combined`` retains a dynamic
    two-stage loop with paired B loads and vector output for comparison.
    Launch exactly ``launch``'s grid/threads with its dynamic shared memory.
    A/B must be 16-byte aligned and C must be eight-byte aligned.
    """

    m: int
    n: int
    k: int
    dtype: str = "f16"
    schedule: str = "stagepair"
    output_dtype: str = "f32"
    layout: str = "rcr"
    arch: str = "sm_80"

    def __post_init__(self) -> None:
        for name in ("m", "n", "k"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0 or value % 128:
                raise ValueError(f"{name} must be a positive integer multiple of 128")
        if self.dtype not in ("f16", "bf16"):
            raise ValueError("GEMM input dtype must be f16 or bf16")
        if self.schedule not in ("stagepair", "combined"):
            raise ValueError("GEMM schedule must be stagepair or combined")
        if self.output_dtype != "f32" or self.layout != "rcr" or self.arch != "sm_80":
            raise ValueError("GEMM requires f32 output, rcr layout and sm_80")
        # The proven generator computes element offsets with mad.lo.u32 before
        # widening to byte addresses. Reject shapes that could wrap that index.
        if max(self.m*self.k, self.n*self.k, self.m*self.n) > 1 << 32:
            raise ValueError("GEMM element indices must fit in unsigned 32 bits")
        if self.n//128 > 65535:
            raise ValueError("GEMM N tile count exceeds the CUDA grid-y limit")

    @property
    def launch(self) -> dict:
        """CUDA launch configuration; ``smem`` is dynamic shared bytes."""
        return {"grid": [self.m//128, self.n//128, 1], "threads": 128, "smem": 65536}

    @property
    def metadata(self) -> dict:
        """Serializable precision/layout and fixed-geometry contract."""
        return {"kernel": "gemm", "arch": self.arch, "m": self.m, "n": self.n,
                "k": self.k, "dtype": self.dtype, "output_dtype": self.output_dtype,
                "accumulation": "f32", "layout": self.layout, "schedule": self.schedule,
                "tile_mnk": [128, 128, 64], "warp_mn": [64, 64], "stages": 2,
                "shared_swizzle": [7, 7], "input_alignment": 16, "output_alignment": 8}


def _prologue(plan: GemmPlan) -> list[str]:
    lines = [".version 8.7\n.target sm_80\n.address_size 64\n.extern .shared .align 16 .b8 smem[];",
             ".visible .entry gemm(.param .u64 A,.param .u64 B,.param .u64 C) {",
             ".reg .b32 %r<1000>; .reg .b64 %d<100>; .reg .f32 %c<128>; .reg .f32 %fBias<32>; .reg .pred %p; .reg .pred %pstage;",
             "", "ld.param.u64 %d0,[A]; ld.param.u64 %d1,[B]; ld.param.u64 %d2,[C];",
             "mov.u32 %r0,%tid.x; mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ctaid.y;",
             "mul.lo.u32 %r1,%r1,128; mul.lo.u32 %r2,%r2,128;",
             "and.b32 %r3,%r0,31; shr.u32 %r4,%r0,5;",
             "shr.u32 %r5,%r4,1; mul.lo.u32 %r5,%r5,64;",
             "and.b32 %r6,%r4,1; mul.lo.u32 %r6,%r6,64;",
             "mov.u32 %r8,0; mov.u32 %r9,0;"]
    lines += [f"mov.f32 %c{i},0f00000000;" for i in range(128)]
    for which, block, shared_base in ((0, 1, 0), (1, 2, 16384)):
        for chunk in range(8):
            cid = which*8+chunk
            sr, gr = 100+cid, 10+cid
            lines += [f"add.u32 %r20,%r0,{chunk*128};",
                      "shr.u32 %r21,%r20,3;",
                      "and.b32 %r22,%r20,7; mul.lo.u32 %r22,%r22,8;",
                      f"add.u32 %r23,%r21,%r{block}; mad.lo.u32 %r24,%r23,{plan.k},%r22;",
                      f"mul.wide.u32 %d3,%r24,2; add.u64 %d{gr},%d{which},%d3;",
                      "and.b32 %r25,%r21,7; mul.lo.u32 %r25,%r25,8; xor.b32 %r26,%r22,%r25;",
                      f"mad.lo.u32 %r{sr},%r21,64,%r26; mul.lo.u32 %r{sr},%r{sr},2;",
                      f"add.u32 %r{sr},%r{sr},{shared_base};"]
    for ik in range(4):
        a, b = 200+ik*4, 300+ik*8
        lines += ["and.b32 %r30,%r3,15; add.u32 %r30,%r30,%r5;",
                  "add.u32 %r30,%r30,0; shr.u32 %r31,%r3,4; mul.lo.u32 %r31,%r31,8;",
                  f"add.u32 %r31,%r31,{ik*16}; and.b32 %r32,%r30,7;",
                  "mul.lo.u32 %r32,%r32,8; xor.b32 %r31,%r31,%r32;",
                  f"mad.lo.u32 %r{a},%r30,64,%r31; mul.lo.u32 %r{a},%r{a},2;",
                  "and.b32 %r30,%r3,7; add.u32 %r30,%r30,%r6;",
                  "add.u32 %r30,%r30,0; shr.u32 %r31,%r3,3; and.b32 %r31,%r31,1;",
                  "mul.lo.u32 %r31,%r31,8;",
                  f"add.u32 %r31,%r31,{ik*16}; and.b32 %r32,%r30,7;",
                  "mul.lo.u32 %r32,%r32,8; xor.b32 %r31,%r31,%r32;",
                  f"mad.lo.u32 %r{b},%r30,64,%r31; mul.lo.u32 %r{b},%r{b},2;",
                  f"add.u32 %r{b},%r{b},16384;"]
    return lines


def _copies(stage: str, *, folded: bool) -> list[str]:
    lines = []
    for which in range(2):
        for chunk in range(8):
            cid = which*8+chunk
            if folded:
                target = 80+which
                if chunk == 0:
                    lines += [f"add.u32 %r{target},%r{100+which*8},{stage};"]
                offset = f"+{chunk*2048}" if chunk else ""
            else:
                target, offset = 40, ""
                lines += [f"add.u32 %r40,%r{100+cid},{stage};"]
            lines += [f"cp.async.cg.shared.global [%r{target}{offset}],[%d{10+cid}],16;"]
    return lines+["cp.async.commit_group;"]


def _advance() -> list[str]:
    return [f"add.u64 %d{10+cid},%d{10+cid},128;" for cid in range(16)]


def _compute(dtype: str, stage: str) -> list[str]:
    # Upper sixteen lanes supply the adjacent N fragment's row addresses for
    # each paired B x4 load. Each aligned half feeds one HMMA B operand.
    lines = ["shr.u32 %r47,%r3,4;", "mul.lo.u32 %r47,%r47,1024;"]
    for ik in range(4):
        lines += [f"add.u32 %r41,%r{200+ik*4},{stage};",
                  f"add.u32 %r42,%r{300+ik*8},{stage};",
                  "add.u32 %r42,%r42,%r47;"]
        for mi in range(4):
            aa = ",".join(f"%r{400+mi*4+j}" for j in range(4))
            lines += [f"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{aa}}},[%r41+{mi*2048}];"]
        for ni in range(0, 8, 2):
            bb = ",".join(f"%r{500+ni*2+j}" for j in range(4))
            lines += [f"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{bb}}},[%r42+{ni*1024}];"]
        for ni in range(8):
            bb = ",".join(f"%r{500+ni*2+j}" for j in range(2))
            for mi in range(4):
                aa = ",".join(f"%r{400+mi*4+j}" for j in range(4))
                cc = ",".join(f"%c{(mi*8+ni)*4+j}" for j in range(4))
                lines += [f"mma.sync.aligned.m16n8k16.row.col.f32.{dtype}.{dtype}.f32 "
                          f"{{{cc}}},{{{aa}}},{{{bb}}},{{{cc}}};"]
    return lines


def _body(dtype: str, current: str, following: str, *, folded: bool) -> list[str]:
    return (["cp.async.wait_group 0; bar.sync 0;"]+_copies(following, folded=folded)
            +_compute(dtype, current)+["bar.sync 0;"]+_advance())


def _epilogue(n: int) -> list[str]:
    lines = ["shr.u32 %r50,%r3,2; add.u32 %r50,%r50,%r5; add.u32 %r50,%r50,%r1;",
             "and.b32 %r51,%r3,3; mul.lo.u32 %r51,%r51,2; add.u32 %r51,%r51,%r6; add.u32 %r51,%r51,%r2;"]
    # The lane owns adjacent columns; even N and 128-column tile origins make
    # each output pair naturally eight-byte aligned. Form each row base once.
    for mi in range(4):
        for j, dr in ((0, 0), (2, 8)):
            lines += [f"add.u32 %r52,%r50,{mi*16+dr}; mad.lo.u32 %r53,%r52,{n},%r51;",
                      "mul.wide.u32 %d3,%r53,4; add.u64 %d4,%d2,%d3;"]
            for ni in range(8):
                c = (mi*8+ni)*4+j
                offset = f"+{ni*32}" if ni else ""
                lines += [f"st.global.v2.f32 [%d4{offset}],{{%c{c},%c{c+1}}};"]
    return lines


def generate(plan: GemmPlan) -> str:
    """Generate deterministic PTX for a validated plan, without compiler effects."""
    if not isinstance(plan, GemmPlan):
        raise TypeError("generate expects a GemmPlan")
    lines = _prologue(plan)
    if plan.schedule == "stagepair":
        lines += _copies("0", folded=True)+_advance()
        first = _body(plan.dtype, "0", "32768", folded=True)
        second = _body(plan.dtype, "32768", "0", folded=True)
        if plan.k > 128:
            lines += (["LOOP_STAGE_PAIR:"]+first+second+
                      [f"add.u32 %r8,%r8,128; setp.lt.u32 %p,%r8,{plan.k-128}; @%p bra LOOP_STAGE_PAIR;"])
        # Tile T-2 copies T-1 into stage one. The drain has no speculative copy.
        lines += first+["cp.async.wait_group 0; bar.sync 0;"]+_compute(plan.dtype, "32768")+["bar.sync 0;"]
    else:
        lines += _copies("%r9", folded=False)+_advance()+["LOOP:",
                  "cp.async.wait_group 0; bar.sync 0;", "xor.b32 %r10,%r9,32768;"]
        lines += _copies("%r10", folded=False)+_compute(plan.dtype, "%r9")+["bar.sync 0;"]+_advance()
        lines += ["add.u32 %r8,%r8,64; xor.b32 %r9,%r9,32768;",
                  f"setp.lt.u32 %p,%r8,{plan.k-64}; @%p bra LOOP;",
                  "cp.async.wait_group 0; bar.sync 0;"]+_compute(plan.dtype, "%r9")+["bar.sync 0;"]
    return "\n".join(lines+_epilogue(plan.n)+["ret;", "}", ""])
