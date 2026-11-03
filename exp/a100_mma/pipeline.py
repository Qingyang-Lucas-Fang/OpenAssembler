"""A dedicated double-buffered SM80 GEMM PTX generator for our own backend.

The prologue loads stage 0. Each loop iteration waits for the current stage,
starts the next global-to-shared copy, then computes on the current stage.
The final stage is peeled so all copies are in bounds without speculative reads.
"""
from __future__ import annotations


def pipelined_gemm(m, n, k, dtype="f16", bm=128, bn=128, bk=32, wm=64, wn=32, compact=False):
    if (dtype not in ("f16", "bf16") or bm % wm or bn % wn or wm % 16 or wn % 8
            or bk not in (32, 64) or m % bm or n % bn or k % bk or k < 2*bk):
        raise ValueError("unsupported full-tile pipeline dimensions")
    threads = bm//wm * (bn//wn) * 32
    if threads > 512 or (bn//wn) & (bn//wn-1):
        raise ValueError("warp grid must be a power of two in N and <= 512 threads")
    stride = 2*bk*(bm+bn)
    if stride*2 > 98304:
        raise ValueError("two stages require <= 96 KiB")
    ma, nb = wm//16, wn//8
    accs = ma*nb*4
    shift = (bn//wn).bit_length()-1
    text = [".version 8.7\n.target sm_80\n.address_size 64" +
            ("\n.extern .shared .align 16 .b8 smem[];" if stride*2>49152 else ""),
            ".visible .entry gemm(.param .u64 A,.param .u64 B,.param .u64 C) {",
            f".reg .b32 %r<1000>; .reg .b64 %d<100>; .reg .f32 %c<{accs}>; .reg .pred %p;",
            (f".shared .align 16 .b8 smem[{stride*2}];" if stride*2<=49152 else ""),
            "ld.param.u64 %d0,[A]; ld.param.u64 %d1,[B]; ld.param.u64 %d2,[C];",
            "mov.u32 %r0,%tid.x; mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ctaid.y;",
            f"mul.lo.u32 %r1,%r1,{bm}; mul.lo.u32 %r2,%r2,{bn};",
            "and.b32 %r3,%r0,31; shr.u32 %r4,%r0,5;",
            f"shr.u32 %r5,%r4,{shift}; mul.lo.u32 %r5,%r5,{wm};",
            f"and.b32 %r6,%r4,{bn//wn-1}; mul.lo.u32 %r6,%r6,{wn};",
            "mov.u32 %r8,0; mov.u32 %r9,0;",
            *[f"mov.f32 %c{i},0f00000000;" for i in range(accs)]]
    cp = []
    cid = 0
    for which, rows, block, shared_base in ((0,bm,1,0), (1,bn,2,2*bm*bk)):
        chunks = rows*bk//8
        if chunks % threads:
            raise ValueError("each warp must have the same number of 16-byte copies")
        for chunk in range(chunks//threads):
            sr, gr = 100+cid, 10+cid
            text += [f"add.u32 %r20,%r0,{chunk*threads};",
                     f"shr.u32 %r21,%r20,{(bk//8).bit_length()-1};",
                     f"and.b32 %r22,%r20,{bk//8-1}; mul.lo.u32 %r22,%r22,8;",
                     f"add.u32 %r23,%r21,%r{block}; mad.lo.u32 %r24,%r23,{k},%r22;",
                     f"mul.wide.u32 %d3,%r24,2; add.u64 %d{gr},%d{which},%d3;",
                     f"and.b32 %r25,%r21,{bk//8-1}; mul.lo.u32 %r25,%r25,8; xor.b32 %r26,%r22,%r25;",
                     f"mad.lo.u32 %r{sr},%r21,{bk},%r26; mul.lo.u32 %r{sr},%r{sr},2;",
                     f"add.u32 %r{sr},%r{sr},{shared_base};"]
            cp.append((sr,gr));cid += 1
    frag_a, frag_b = {}, {}
    for ik in range(bk//16):
        for mi in range(1 if compact else ma):
            reg = 200+ik*ma+mi
            text += ["and.b32 %r30,%r3,15; add.u32 %r30,%r30,%r5;",
                     f"add.u32 %r30,%r30,{mi*16}; shr.u32 %r31,%r3,4; mul.lo.u32 %r31,%r31,8;",
                     f"add.u32 %r31,%r31,{ik*16}; and.b32 %r32,%r30,{bk//8-1};",
                     "mul.lo.u32 %r32,%r32,8; xor.b32 %r31,%r31,%r32;",
                     f"mad.lo.u32 %r{reg},%r30,{bk},%r31; mul.lo.u32 %r{reg},%r{reg},2;"]
            frag_a[ik,mi]=reg
        for ni in range(1 if compact else nb):
            reg=300+ik*nb+ni
            text += ["and.b32 %r30,%r3,7; add.u32 %r30,%r30,%r6;",
                     f"add.u32 %r30,%r30,{ni*8}; shr.u32 %r31,%r3,3; and.b32 %r31,%r31,1;",
                     "mul.lo.u32 %r31,%r31,8;",
                     f"add.u32 %r31,%r31,{ik*16}; and.b32 %r32,%r30,{bk//8-1};",
                     "mul.lo.u32 %r32,%r32,8; xor.b32 %r31,%r31,%r32;",
                     f"mad.lo.u32 %r{reg},%r30,{bk},%r31; mul.lo.u32 %r{reg},%r{reg},2;",
                     f"add.u32 %r{reg},%r{reg},{2*bm*bk};"]
            frag_b[ik,ni]=reg

    def copies(stage):
        lines=[]
        for shared,glob in cp:
            lines += [f"add.u32 %r40,%r{shared},%r{stage};",
                      f"cp.async.cg.shared.global [%r40],[%d{glob}],16;"]
        return lines+["cp.async.commit_group;"]

    def advance():
        return [f"add.u64 %d{g},%d{g},{2*bk};" for _,g in cp]

    def compute():
        lines=[]
        for ik in range(bk//16):
            if compact:
                lines += [f"add.u32 %r41,%r{frag_a[ik,0]},%r9;",
                          f"add.u32 %r42,%r{frag_b[ik,0]},%r9;"]
            for mi in range(ma):
                dst=','.join(f'%r{400+mi*4+j}' for j in range(4))
                if compact:
                    lines += [f"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{dst}}},[%r41+{mi*16*bk*2}];"]
                else:
                    lines += [f"add.u32 %r40,%r{frag_a[ik,mi]},%r9;",
                              f"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{dst}}},[%r40];"]
            for ni in range(nb):
                dst=','.join(f'%r{500+ni*2+j}' for j in range(2))
                if compact:
                    lines += [f"ldmatrix.sync.aligned.m8n8.x2.shared.b16 {{{dst}}},[%r42+{ni*8*bk*2}];"]
                else:
                    lines += [f"add.u32 %r40,%r{frag_b[ik,ni]},%r9;",
                              f"ldmatrix.sync.aligned.m8n8.x2.shared.b16 {{{dst}}},[%r40];"]
            for mi in range(ma):
                aa=','.join(f'%r{400+mi*4+j}' for j in range(4))
                for ni in range(nb):
                    bb=','.join(f'%r{500+ni*2+j}' for j in range(2))
                    cc=','.join(f'%c{(mi*nb+ni)*4+j}' for j in range(4))
                    lines += [f"mma.sync.aligned.m16n8k16.row.col.f32.{dtype}.{dtype}.f32 "
                              f"{{{cc}}},{{{aa}}},{{{bb}}},{{{cc}}};"]
        return lines

    toggle_next = (f"xor.b32 %r10,%r9,{stride};" if stride & (stride-1) == 0 else
                   f"sub.u32 %r10,{stride},%r9;")
    toggle_current = (f"xor.b32 %r9,%r9,{stride};" if stride & (stride-1) == 0 else
                      f"sub.u32 %r9,{stride},%r9;")
    text += copies(9)+advance()+["LOOP:", "cp.async.wait_group 0; bar.sync 0;",toggle_next]
    text += copies(10)+compute()+["bar.sync 0;"]+advance()
    text += [f"add.u32 %r8,%r8,{bk}; {toggle_current}",
             f"setp.lt.u32 %p,%r8,{k-bk}; @%p bra LOOP;",
             "cp.async.wait_group 0; bar.sync 0;"]+compute()+["bar.sync 0;"]
    text += ["shr.u32 %r50,%r3,2; add.u32 %r50,%r50,%r5; add.u32 %r50,%r50,%r1;",
             "and.b32 %r51,%r3,3; mul.lo.u32 %r51,%r51,2; add.u32 %r51,%r51,%r6; add.u32 %r51,%r51,%r2;"]
    for mi in range(ma):
        for ni in range(nb):
            for j,(dr,dc) in enumerate(((0,0),(0,1),(8,0),(8,1))):
                text += [f"add.u32 %r52,%r50,{mi*16+dr}; mad.lo.u32 %r53,%r52,{n},%r51;",
                         f"add.u32 %r53,%r53,{ni*8+dc}; mul.wide.u32 %d3,%r53,4; add.u64 %d4,%d2,%d3;",
                         f"st.global.f32 [%d4],%c{(mi*nb+ni)*4+j};"]
    return '\n'.join(text+["ret;", "}", ""])

