"""Generate full-tile RCR GEMM PTX for the experimental backend.

A is MxK row-major, B is NxK row-major (logical B transpose), C is MxN f32.
No edge masks: the caller must enforce tile divisibility and block geometry.
"""
from __future__ import annotations
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent


def warp_gemm(m: int, n: int, k: int, dtype="f16"):
    if m % 16 or n % 8 or k % 16 or min(m, n, k) <= 0 or dtype not in ("f16", "bf16"):
        raise ValueError("positive M%16=N%8=K%16=0 and f16/bf16 required")
    lines = [".version 8.7", ".target sm_80", ".address_size 64",
             ".visible .entry gemm(.param .u64 A, .param .u64 B, .param .u64 C) {",
             ".reg .b32 %r<80>; .reg .b64 %d<20>; .reg .f32 %c<4>; .reg .pred %p;",
             "ld.param.u64 %d0,[A]; ld.param.u64 %d1,[B]; ld.param.u64 %d2,[C];",
             "mov.u32 %r0,%tid.x; mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ctaid.y;",
             "shr.u32 %r3,%r0,2; and.b32 %r4,%r0,3; mul.lo.u32 %r5,%r4,2;",
             "mad.lo.u32 %r6,%r1,16,%r3; mad.lo.u32 %r7,%r2,8,%r3;",
             "mov.u32 %r8,0;",
             *[f"mov.f32 %c{i},0f00000000;" for i in range(4)], "LOOP:"]
    for i, (dr, dk) in enumerate(((0, 0), (8, 0), (0, 8), (8, 8))):
        lines += [f"add.u32 %r10,%r6,{dr}; mad.lo.u32 %r11,%r10,{k},%r5;",
                  f"add.u32 %r11,%r11,{dk}; add.u32 %r11,%r11,%r8;",
                  "mul.wide.u32 %d3,%r11,2; add.u64 %d4,%d0,%d3;",
                  f"ld.global.b32 %r{20+i},[%d4];"]
    for i, dk in enumerate((0, 8)):
        lines += [f"mad.lo.u32 %r11,%r7,{k},%r5; add.u32 %r11,%r11,{dk};",
                  "add.u32 %r11,%r11,%r8; mul.wide.u32 %d3,%r11,2; add.u64 %d4,%d1,%d3;",
                  f"ld.global.b32 %r{24+i},[%d4];"]
    lines += [f"mma.sync.aligned.m16n8k16.row.col.f32.{dtype}.{dtype}.f32 "
              "{%c0,%c1,%c2,%c3},{%r20,%r21,%r22,%r23},{%r24,%r25},{%c0,%c1,%c2,%c3};",
              f"add.u32 %r8,%r8,16; setp.lt.u32 %p,%r8,{k}; @%p bra LOOP;",
              "mad.lo.u32 %r30,%r2,8,%r5;"]
    for i, (dr, dc) in enumerate(((0, 0), (0, 1), (8, 0), (8, 1))):
        lines += [f"add.u32 %r31,%r6,{dr}; mad.lo.u32 %r32,%r31,{n},%r30;",
                  f"add.u32 %r32,%r32,{dc}; mul.wide.u32 %d3,%r32,4; add.u64 %d4,%d2,%d3;",
                  f"st.global.f32 [%d4],%c{i};"]
    return "\n".join(lines + ["ret;", "}", ""])


def cta_gemm(m: int, n: int, k: int, dtype="f16", async_copy=False, tile=64):
    """64/128-square CTA; 32x32 warp tiles, XOR shared layout, LDSM, f32 HMMA.

    Initial version has synchronous wide copies and one shared stage. It is a
    pipeline/compiler validation baseline, not the final throughput candidate.
    """
    if tile not in (64, 128) or min(m, n, k) <= 0 or m % tile or n % tile or k % 64:
        raise ValueError("CTA kernel requires M/N divisible by tile=64/128 and K divisible by 64")
    if dtype not in ("f16", "bf16"):
        raise ValueError(dtype)
    lines = [".version 8.7", ".target sm_80", ".address_size 64",
             ".visible .entry gemm(.param .u64 A, .param .u64 B, .param .u64 C) {",
             ".reg .b32 %r<300>; .reg .b64 %d<20>; .reg .f32 %c<32>; .reg .pred %p;",
             f".shared .align 16 .b8 smem[{tile*256}];",
             "ld.param.u64 %d0,[A]; ld.param.u64 %d1,[B]; ld.param.u64 %d2,[C];",
             "mov.u32 %r0,%tid.x; mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ctaid.y;",
             f"mul.lo.u32 %r1,%r1,{tile}; mul.lo.u32 %r2,%r2,{tile};",
             "and.b32 %r3,%r0,31; shr.u32 %r4,%r0,5;",
             f"shr.u32 %r5,%r4,{1 if tile == 64 else 2}; mul.lo.u32 %r5,%r5,32;",
             f"and.b32 %r6,%r4,{tile//32-1}; mul.lo.u32 %r6,%r6,32;",
             "mov.u32 %r8,0;",
             *[f"mov.f32 %c{i},0f00000000;" for i in range(32)], "LOOP:"]
    threads = (tile // 32)**2 * 32
    copies = tile * 8 // threads
    for which, base_s, block_r in ((0, 0, 1), (1, tile*128, 2)):
        for chunk in range(copies):
            lines += [f"add.u32 %r10,%r0,{chunk*threads};",
                      "shr.u32 %r11,%r10,3; and.b32 %r12,%r10,7; mul.lo.u32 %r12,%r12,8;",
                      f"add.u32 %r13,%r11,%r{block_r}; mad.lo.u32 %r14,%r13,{k},%r12;",
                      "add.u32 %r14,%r14,%r8; mul.wide.u32 %d3,%r14,2;",
                      f"add.u64 %d4,%d{which},%d3;"]
            if not async_copy:
                lines += ["ld.global.v4.b32 {%r20,%r21,%r22,%r23},[%d4];"]
            lines += [
                      "and.b32 %r15,%r11,7; mul.lo.u32 %r15,%r15,8; xor.b32 %r16,%r12,%r15;",
                      "mad.lo.u32 %r17,%r11,64,%r16; mul.lo.u32 %r17,%r17,2;",
                      f"add.u32 %r17,%r17,{base_s};"]
            lines += (["cp.async.cg.shared.global [%r17],[%d4],16;"] if async_copy else
                      ["st.shared.v4.b32 [%r17],{%r20,%r21,%r22,%r23};"])
    if async_copy:
        lines += ["cp.async.commit_group; cp.async.wait_group 0;"]
    lines += ["bar.sync 0;"]
    for ik in range(0, 64, 16):
        for mi in range(2):
            dest = ",".join(f"%r{100+mi*4+j}" for j in range(4))
            lines += ["and.b32 %r30,%r3,15; add.u32 %r30,%r30,%r5;",
                      f"add.u32 %r30,%r30,{mi*16};",
                      "shr.u32 %r31,%r3,4; mul.lo.u32 %r31,%r31,8;",
                      f"add.u32 %r31,%r31,{ik};",
                      "and.b32 %r32,%r30,7; mul.lo.u32 %r32,%r32,8; xor.b32 %r31,%r31,%r32;",
                      "mad.lo.u32 %r33,%r30,64,%r31; mul.lo.u32 %r33,%r33,2;",
                      f"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{dest}}},[%r33];"]
        for ni in range(4):
            dest = ",".join(f"%r{200+ni*2+j}" for j in range(2))
            lines += ["and.b32 %r30,%r3,7; add.u32 %r30,%r30,%r6;",
                      f"add.u32 %r30,%r30,{ni*8};",
                      "shr.u32 %r31,%r3,3; and.b32 %r31,%r31,1; mul.lo.u32 %r31,%r31,8;",
                      f"add.u32 %r31,%r31,{ik};",
                      "and.b32 %r32,%r30,7; mul.lo.u32 %r32,%r32,8; xor.b32 %r31,%r31,%r32;",
                      f"mad.lo.u32 %r33,%r30,64,%r31; mul.lo.u32 %r33,%r33,2; add.u32 %r33,%r33,{tile*128};",
                      f"ldmatrix.sync.aligned.m8n8.x2.shared.b16 {{{dest}}},[%r33];"]
        for mi in range(2):
            aa = ",".join(f"%r{100+mi*4+j}" for j in range(4))
            for ni in range(4):
                bb = ",".join(f"%r{200+ni*2+j}" for j in range(2))
                cc = ",".join(f"%c{(mi*4+ni)*4+j}" for j in range(4))
                lines += [f"mma.sync.aligned.m16n8k16.row.col.f32.{dtype}.{dtype}.f32 "
                          f"{{{cc}}},{{{aa}}},{{{bb}}},{{{cc}}};"]
    lines += ["bar.sync 0;", f"add.u32 %r8,%r8,64; setp.lt.u32 %p,%r8,{k}; @%p bra LOOP;",
              "shr.u32 %r40,%r3,2; add.u32 %r40,%r40,%r5; add.u32 %r40,%r40,%r1;",
              "and.b32 %r41,%r3,3; mul.lo.u32 %r41,%r41,2; add.u32 %r41,%r41,%r6; add.u32 %r41,%r41,%r2;"]
    for mi in range(2):
        for ni in range(4):
            for j, (dr, dc) in enumerate(((0, 0), (0, 1), (8, 0), (8, 1))):
                lines += [f"add.u32 %r42,%r40,{mi*16+dr}; mad.lo.u32 %r43,%r42,{n},%r41;",
                          f"add.u32 %r43,%r43,{ni*8+dc}; mul.wide.u32 %d3,%r43,4; add.u64 %d4,%d2,%d3;",
                          f"st.global.f32 [%d4],%c{(mi*4+ni)*4+j};"]
    return "\n".join(lines + ["ret;", "}", ""])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=64)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--dtype", choices=("f16", "bf16"), default="f16")
    p.add_argument("--kind", choices=("warp", "cta", "async"), default="warp")
    p.add_argument("--tile", type=int, choices=(64, 128), default=64)
    a = p.parse_args()
    target = HERE / "ptx" / f"{a.kind}{a.tile if a.kind != 'warp' else ''}_{a.m}_{a.n}_{a.k}_{a.dtype}.ptx"
    target.parent.mkdir(exist_ok=True)
    text = (warp_gemm(a.m, a.n, a.k, a.dtype) if a.kind == "warp" else
            cta_gemm(a.m, a.n, a.k, a.dtype, async_copy=a.kind == "async", tile=a.tile))
    target.write_text(text)
    print(target)
