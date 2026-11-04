"""Software producer/consumer warp GEMM with ordinary SM80 CTA barriers.

Each warp has one uniform role. Producer warps stage the next shared tile while
consumer warps execute the current tile. Matching BAR 0 generations join both
paths; there is no cross-CTA synchronization or assumption of physical clusters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))
import backend
from openassembler.backend import ra
from openassembler.ptx.parse import Addr, PairDst, Var, Vec, parse_module


def cfg_carried_registers(kernel):
    """Find PTX values live across any backedge, including vector destinations.

    Scope: this generator's warp-uniform role branches and two counted loops.
    Guards on data-producing instructions are rejected to avoid partial defs.
    """
    body = kernel.body
    if any(ins.guard is not None and ins.op != 'bra' for ins in body):
        raise ra.NotSupported('producer experiment only allows guarded branches')

    def regs(operand):
        if isinstance(operand, Var):
            return {operand.name} if operand.name.startswith('%') else set()
        if isinstance(operand, Addr):
            return {operand.base} if operand.base.startswith('%') else set()
        if isinstance(operand, Vec):
            return set().union(*(regs(v) for v in operand.elems))
        if isinstance(operand, PairDst):
            return regs(operand.a) | regs(operand.b)
        return set()

    uses, defs, successors, backedges = [], [], [], []
    for i, ins in enumerate(body):
        writes = ins.op not in ('st', 'bra', 'ret', 'bar', 'cp', 'red', 'membar')
        defs.append(regs(ins.args[0]) if writes and ins.args else set())
        srcs = ins.args[1:] if writes else ins.args
        read = set().union(*(regs(v) for v in srcs))
        if ins.guard:
            read.add(ins.guard[0])
        uses.append(read)
        succ = [] if ins.op == 'ret' else [i + 1] if i + 1 < len(body) else []
        if ins.op == 'bra':
            target = kernel.labels[ins.args[0].name]
            if ins.guard is None:
                succ = [target]
            else:
                succ.append(target)
            if target <= i:
                backedges.append((i, target))
        successors.append(succ)
    live = [set() for _ in body]
    changed = True
    while changed:
        changed = False
        for i in range(len(body) - 1, -1, -1):
            after = set().union(*(live[j] for j in successors[i]))
            before = uses[i] | (after - defs[i])
            if before != live[i]:
                live[i] = before
                changed = True
    carried = set()
    for latch, head in backedges:
        carried |= live[head] & set().union(*defs[head:latch + 1])
    return carried


def producer_gemm(m, n, k, dtype='f16', bm=128, bn=128, bk=64, wm=64, wn=64,
                  producer_warps=1):
    if (dtype not in ('f16', 'bf16') or bm % wm or bn % wn or wm % 16 or wn % 8
            or bk not in (32, 64) or m % bm or n % bn or k % bk or k < bk * 2
            or producer_warps not in (1, 2, 4)):
        raise ValueError('unsupported full-tile producer/consumer configuration')
    pw_threads = producer_warps * 32
    consumer_threads = bm // wm * (bn // wn) * 32
    total_threads = pw_threads + consumer_threads
    if total_threads > 1024 or bn // wn & (bn // wn - 1):
        raise ValueError('unsupported consumer warp grid')
    stride = 2 * bk * (bm + bn)
    shared_bytes = 2 * stride
    if shared_bytes > 98304:
        raise ValueError('producer experiment caps shared allocation at 96 KiB')
    ma, nb = wm // 16, wn // 8
    accs = ma * nb * 4
    text = ['.version 8.7\n.target sm_80\n.address_size 64' +
            ('\n.extern .shared .align 16 .b8 smem[];' if shared_bytes > 49152 else ''),
            '.visible .entry gemm(.param .u64 A,.param .u64 B,.param .u64 C)',
            '{',
            f'.reg .b32 %r<1000>; .reg .b64 %d<100>; .reg .f32 %c<{accs}>;',
            '.reg .pred %pRole,%pLoop;',
            f'.shared .align 16 .b8 smem[{shared_bytes}];' if shared_bytes <= 49152 else '',
            'ld.param.u64 %d0,[A];ld.param.u64 %d1,[B];ld.param.u64 %d2,[C];',
            'mov.u32 %r0,%tid.x;mov.u32 %r1,%ctaid.x;mov.u32 %r2,%ctaid.y;',
            f'mul.lo.u32 %r1,%r1,{bm};mul.lo.u32 %r2,%r2,{bn};',
            f'setp.lt.u32 %pRole,%r0,{pw_threads};@%pRole bra PRODUCER;',
            # All threads in a warp select the same path. Each consumer warp
            # collectively executes LDSM/MMA with lane IDs unchanged.
            'and.b32 %r3,%r0,31;shr.u32 %r4,%r0,5;',
            f'sub.u32 %r4,%r4,{producer_warps};',
            f'shr.u32 %r5,%r4,{(bn // wn).bit_length()-1};mul.lo.u32 %r5,%r5,{wm};',
            f'and.b32 %r6,%r4,{bn//wn-1};mul.lo.u32 %r6,%r6,{wn};',
            'mov.u32 %r8,0;mov.u32 %r9,0;',
            *[f'mov.f32 %c{i},0f00000000;' for i in range(accs)]]
    # Consumer fragment-address bases. The row XOR low bits stay unchanged
    # across the 16-row A and 8-row B instruction tiles.
    for ik in range(bk // 16):
        for matrix in ('a', 'b'):
            reg = (200 if matrix == 'a' else 300) + ik
            text += [('and.b32 %r30,%r3,15;add.u32 %r30,%r30,%r5;' if matrix == 'a' else
                      'and.b32 %r30,%r3,7;add.u32 %r30,%r30,%r6;'),
                     ('shr.u32 %r31,%r3,4;' if matrix == 'a' else
                      'shr.u32 %r31,%r3,3;and.b32 %r31,%r31,1;'),
                     f'mul.lo.u32 %r31,%r31,8;add.u32 %r31,%r31,{ik*16};',
                     f'and.b32 %r32,%r30,{bk//8-1};mul.lo.u32 %r32,%r32,8;xor.b32 %r31,%r31,%r32;',
                     f'mad.lo.u32 %r{reg},%r30,{bk},%r31;mul.lo.u32 %r{reg},%r{reg},2;',
                     f'add.u32 %r{reg},%r{reg},{0 if matrix == "a" else 2*bm*bk};']

    def compute():
        lines = []
        for ik in range(bk // 16):
            lines += [f'add.u32 %r41,%r{200+ik},%r9;add.u32 %r42,%r{300+ik},%r9;']
            for mi in range(ma):
                aa = ','.join(f'%r{400+mi*4+j}' for j in range(4))
                lines += [f'ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{aa}}},[%r41+{mi*16*bk*2}];']
            for ni in range(nb):
                bb = ','.join(f'%r{500+ni*2+j}' for j in range(2))
                lines += [f'ldmatrix.sync.aligned.m8n8.x2.shared.b16 {{{bb}}},[%r42+{ni*8*bk*2}];']
            for ni in range(nb):
                bb = ','.join(f'%r{500+ni*2+j}' for j in range(2))
                for mi in range(ma):
                    aa = ','.join(f'%r{400+mi*4+j}' for j in range(4))
                    cc = ','.join(f'%c{(mi*nb+ni)*4+j}' for j in range(4))
                    lines += [f'mma.sync.aligned.m16n8k16.row.col.f32.{dtype}.{dtype}.f32 '
                              f'{{{cc}}},{{{aa}}},{{{bb}}},{{{cc}}};']
        return lines

    toggle = (f'xor.b32 %r9,%r9,{stride};' if stride & (stride-1) == 0 else
              f'sub.u32 %r9,{stride},%r9;')
    text += ['bar.sync 0;', 'CONSUME_LOOP:'] + compute()
    text += ['bar.sync 0;', f'add.u32 %r8,%r8,{bk};', toggle,
             f'setp.lt.u32 %pLoop,%r8,{k-bk};@%pLoop bra CONSUME_LOOP;'] + compute()
    # No shared buffer can be overwritten after the producers finish. The
    # consumers can store and exit independently after the final ready barrier.
    text += ['shr.u32 %r50,%r3,2;add.u32 %r50,%r50,%r5;add.u32 %r50,%r50,%r1;',
             'and.b32 %r51,%r3,3;mul.lo.u32 %r51,%r51,2;add.u32 %r51,%r51,%r6;add.u32 %r51,%r51,%r2;']
    for mi in range(ma):
        for ni in range(nb):
            for j, (dr, dc) in enumerate(((0, 0), (0, 1), (8, 0), (8, 1))):
                text += [f'add.u32 %r52,%r50,{mi*16+dr};mad.lo.u32 %r53,%r52,{n},%r51;',
                         f'add.u32 %r53,%r53,{ni*8+dc};mul.wide.u32 %d3,%r53,4;add.u64 %d4,%d2,%d3;',
                         f'st.global.f32 [%d4],%c{(mi*nb+ni)*4+j};']
    text += ['ret;', 'PRODUCER:', 'mov.u32 %r8,0;mov.u32 %r9,0;']
    cp = []
    cid = 0
    for which, rows, block, shared_base in ((0, bm, 1, 0), (1, bn, 2, 2*bm*bk)):
        chunks = rows*bk//8
        if chunks % pw_threads:
            raise ValueError('producer load distribution must be exact')
        for chunk in range(chunks // pw_threads):
            sr, gr = 100+cid, 10+cid
            text += [f'add.u32 %r20,%r0,{chunk*pw_threads};',
                     f'shr.u32 %r21,%r20,{(bk//8).bit_length()-1};',
                     f'and.b32 %r22,%r20,{bk//8-1};mul.lo.u32 %r22,%r22,8;',
                     f'add.u32 %r23,%r21,%r{block};mad.lo.u32 %r24,%r23,{k},%r22;',
                     f'mul.wide.u32 %d3,%r24,2;add.u64 %d{gr},%d{which},%d3;',
                     f'and.b32 %r25,%r21,{bk//8-1};mul.lo.u32 %r25,%r25,8;xor.b32 %r26,%r22,%r25;',
                     f'mad.lo.u32 %r{sr},%r21,{bk},%r26;mul.lo.u32 %r{sr},%r{sr},2;',
                     f'add.u32 %r{sr},%r{sr},{shared_base};']
            cp.append((sr, gr))
            cid += 1

    def copies():
        lines = []
        for shared, glob in cp:
            lines += [f'add.u32 %r40,%r{shared},%r9;',
                      f'cp.async.cg.shared.global [%r40],[%d{glob}],16;']
        return lines + ['cp.async.commit_group;cp.async.wait_group 0;']

    def advance():
        return [f'add.u64 %d{glob},%d{glob},{2*bk};' for shared, glob in cp]

    text += copies() + ['bar.sync 0;'] + advance() + [toggle, 'PRODUCE_LOOP:']
    text += copies() + ['bar.sync 0;'] + advance() + [toggle,
             f'add.u32 %r8,%r8,{bk};setp.lt.u32 %pLoop,%r8,{k-bk};@%pLoop bra PRODUCE_LOOP;',
             'ret;', '}']
    return '\n'.join(text) + '\n'


def configure():
    backend.install()
    backend.OPTIMIZE_LOCALS = True
    backend.loop_carried_registers = cfg_carried_registers
    backend.HOIST_LOOP_INVARIANTS = False
    backend.MINED_SCALAR = True
    backend.RELAX_TENSOR = True
    backend.TENSOR_ISSUE_GAP = 8
    backend.LDSM_READ_BARRIER = False
    backend.INLINE_POINTER_INCREMENTS = True
    backend.TENSOR_REUSE = True
    backend.LIST_SCHEDULE = False
    backend.EXPERIMENTAL_RECYCLE_WITHOUT_WAIT = False


def compile_one(name, m, n, k, dtype='f16', bm=128, bn=128, bk=64, wm=64, wn=64, pw=1):
    from openassembler.cli import main
    configure()
    source = HERE / f'producer_{name}.ptx'
    source.write_text(producer_gemm(m, n, k, dtype, bm, bn, bk, wm, wn, pw))
    output = HERE / 'producer_build' / name
    backend.STAGE_RECORDS.clear()
    start = time.perf_counter()
    rc = main(['compile', str(source), '--out-dir', str(output), '--opt-level', 'safe'])
    if rc:
        raise RuntimeError(f'{name}: compilation failed (rc={rc})')
    geometry = dict(m=m,n=n,k=k,bm=bm,bn=bn,bk=bk,wm=wm,wn=wn,producer_warps=pw)
    (output / 'experiment.json').write_text(json.dumps({
        'compiler':'OpenAssembler experimental producer/consumer selector', 'fallback':False,
        'reference_and_ELF_container':'NVIDIA ptxas', 'cli_wall_ms':(time.perf_counter()-start)*1000,
        'geometry':geometry, 'kernels':backend.STAGE_RECORDS.copy(),
        'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'synchronization':'warp-uniform roles; same BAR0 generation on both paths; producer wait_group0 before ready barrier',
    }, indent=2) + '\n')
    shared_bytes = 4*bk*(bm+bn)
    return dict(build=str(output.relative_to(BASE)), name=name, m=m,n=n,k=k,dtype=dtype,
                tile_m=bm,tile_n=bn,threads=pw*32+bm//wm*(bn//wn)*32,
                smem=shared_bytes if shared_bytes>49152 else 0)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--large',action='store_true')
    args=parser.parse_args()
    cases=[]; failures=[]
    for bk, wm, wn, pw in ((32,32,32,1),(32,64,32,1),(64,64,64,1),(64,64,64,2),(64,64,64,4)):
        # The 544-thread paired smoke launch exhausted a cubin's register
        # resource limit on real A100. Preserve its evidence; do not rerun it.
        if args.large and (bk,wm,wn,pw)==(32,32,32,1):
            continue
        m,n,k=(4096,4096,4096) if args.large else (256,384,256)
        name=f'producer_k{bk}_w{wm}x{wn}_p{pw}_{"large" if args.large else "smoke"}'
        try:cases.append(compile_one(name,m,n,k,'f16',128,128,bk,wm,wn,pw))
        except RuntimeError as exc:failures.append({'name':name,'error':str(exc)})
    # The existing modal_run expects build paths relative to BASE/build. Our
    # outputs are deliberately all in our own phase3 directory, so normalize
    # with a relative path that remains inside BASE and is accepted by it.
    for case in cases:case['build']='../'+case['build']
    tag='large' if args.large else 'smoke'
    (HERE/f'producer_{tag}_suite.json').write_text(json.dumps(cases,indent=2)+'\n')
    (HERE/f'producer_{tag}_failures.json').write_text(json.dumps(failures,indent=2)+'\n')
