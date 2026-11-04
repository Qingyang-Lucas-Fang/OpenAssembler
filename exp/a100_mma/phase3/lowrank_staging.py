"""Compile-time rank-7 Strassen with register-cached, fused operand staging.

Proof-of-lowering scope: one warp computes a 32x32x32 macro product. The 16x16
quadrants are loaded once into registers; constant +/-1 transforms are rounded
to FP16, stored directly to shared memory, and immediately consumed by HMMA.
Only one 16x16 FP32 product tile is live before reconstruction into four output
quadrants. A generated classical rank-8 control uses the identical tile/layout
and output mode. This is not an optimized large-GEMM implementation.
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
from openassembler.fastmm.factors import naive, strassen, verify
from openassembler.ptx.parse import Var


class LowrankTensorSelector(backend.TensorSelector):
    """Experiment-local packed FP16 transform instruction selection."""
    def select(self, ins):
        if ins.op in ('add', 'sub') and ins.mods == ('rn', 'f16x2'):
            if ins.guard or len(ins.args) != 3 or not isinstance(ins.args[0], Var):
                raise backend.ra.NotSupported('lowrank packed transform form')
            dst = self.new()
            a, b = (self.reg32(arg) for arg in ins.args[1:])
            # Current encoder has the positive register-source HADD2 template;
            # flip both packed half signs explicitly for subtraction.
            if ins.op == 'sub':
                neg = self.new()
                self.emit('LOP3.LUT {d0}, {s0}, 0x80008000, RZ, 0x3c, !PT', [neg], [b])
                b = neg
            self.emit('HADD2 {d0}, {s0}, {s1}', [dst], [a, b])
            self.bind(ins.args[0].name, dst)
            return
        return super().select(ins)


def lowrank_gemm(m=96, n=96, scheme_name='strassen'):
    if m % 32 or n % 32:
        raise ValueError('proof-of-lowering uses full 32x32 output tiles and K=32')
    scheme = strassen() if scheme_name == 'strassen' else naive(2, 2, 2)
    if not verify(scheme):
        raise AssertionError('factor identity')
    ptx = ['.version 8.7\n.target sm_80\n.address_size 64',
           '.visible .entry gemm(.param .u64 A,.param .u64 B,.param .u64 C) {',
           '.reg .b32 %r<1000>; .reg .b64 %d<20>; .reg .f32 %c<32>,%p<8>;',
           '.shared .align 16 .b8 smem[1024];',
           'ld.param.u64 %d0,[A];ld.param.u64 %d1,[B];ld.param.u64 %d2,[C];',
           'mov.u32 %r0,%tid.x;mov.u32 %r1,%ctaid.x;mov.u32 %r2,%ctaid.y;',
           'mul.lo.u32 %r1,%r1,32;mul.lo.u32 %r2,%r2,32;',
           *[f'mov.f32 %c{i},0f00000000;' for i in range(32)]]
    # Each lane caches four packed elements from each of eight base quadrants.
    # For B, factor index is (K quadrant, N quadrant), while storage is [N,K].
    for operand, base_reg, ptr, block in (('A',100,0,1), ('B',200,1,2)):
        for q in range(4):
            row_offset = (q // 2 if operand == 'A' else q % 2) * 16
            k_offset = (q % 2 if operand == 'A' else q // 2) * 16
            for t in range(4):
                ptx += [f'add.u32 %r20,%r0,{t*32};shr.u32 %r21,%r20,3;',
                        'and.b32 %r22,%r20,7;mul.lo.u32 %r22,%r22,2;',
                        f'add.u32 %r21,%r21,{row_offset};add.u32 %r21,%r21,%r{block};',
                        f'add.u32 %r22,%r22,{k_offset};mad.lo.u32 %r23,%r21,32,%r22;',
                        f'mul.wide.u32 %d3,%r23,2;add.u64 %d4,%d{ptr},%d3;',
                        f'ld.global.b32 %r{base_reg+q*4+t},[%d4];']
    # Swizzled shared destinations correspond to packed row-major [16,16].
    for t in range(4):
        ptx += [f'add.u32 %r20,%r0,{t*32};shr.u32 %r21,%r20,3;',
                'and.b32 %r22,%r21,1;mul.lo.u32 %r22,%r22,4;',
                f'xor.b32 %r{40+t},%r20,%r22;mul.lo.u32 %r{40+t},%r{40+t},4;']
    ptx += ['and.b32 %r60,%r0,15;shr.u32 %r61,%r0,4;mul.lo.u32 %r61,%r61,8;',
            'and.b32 %r62,%r60,1;mul.lo.u32 %r62,%r62,8;xor.b32 %r61,%r61,%r62;',
            'mad.lo.u32 %r63,%r60,16,%r61;mul.lo.u32 %r63,%r63,2;',
            'and.b32 %r60,%r0,7;shr.u32 %r61,%r0,3;and.b32 %r61,%r61,1;',
            'mul.lo.u32 %r61,%r61,8;and.b32 %r62,%r60,1;',
            'mul.lo.u32 %r62,%r62,8;xor.b32 %r61,%r61,%r62;',
            'mad.lo.u32 %r64,%r60,16,%r61;mul.lo.u32 %r64,%r64,2;add.u32 %r64,%r64,512;']

    def transform(matrix, product, base_reg, destination):
        terms = [(q, int(row[product])) for q, row in enumerate(matrix) if row[product]]
        if not 1 <= len(terms) <= 2 or any(abs(s) != 1 for _, s in terms):
            raise ValueError('microprototype supports one/two compile-time +/-1 terms')
        lines = []
        for t in range(4):
            q, sign = terms[0]
            src = base_reg + q*4 + t
            dst = destination + t
            lines += [f'mov.b32 %r{dst},%r{src};' if sign == 1 else
                      f'xor.b32 %r{dst},%r{src},0x80008000;']
            for q, sign in terms[1:]:
                lines += [f'{"add" if sign == 1 else "sub"}.rn.f16x2 '
                          f'%r{dst},%r{dst},%r{base_reg+q*4+t};']
        return lines

    for product in range(scheme.rank):
        ptx += [f'// product {product}: all factors are compile-time constants']
        ptx += transform(scheme.u, product, 100, 300)
        ptx += transform(scheme.v, product, 200, 310)
        for t in range(4):
            ptx += [f'st.shared.b32 [%r{40+t}],%r{300+t};',
                    f'add.u32 %r70,%r{40+t},512;st.shared.b32 [%r70],%r{310+t};']
        ptx += ['bar.sync 0;',
                'ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r500,%r501,%r502,%r503},[%r63];',
                *[f'mov.f32 %p{i},0f00000000;' for i in range(8)]]
        for ni in range(2):
            pp = ','.join(f'%p{ni*4+j}' for j in range(4))
            ptx += [f'ldmatrix.sync.aligned.m8n8.x2.shared.b16 {{%r510,%r511}},[%r64+{ni*256}];',
                    'mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 '
                    f'{{{pp}}},{{%r500,%r501,%r502,%r503}},{{%r510,%r511}},{{{pp}}};']
        # Products are consumed in factor-column order into FP32 output tiles.
        for q, row in enumerate(scheme.w):
            coefficient = int(row[product])
            if coefficient:
                for t in range(8):
                    ptx += [f'{"add" if coefficient == 1 else "sub"}.f32 '
                            f'%c{q*8+t},%c{q*8+t},%p{t};']
        # Explicit read-completion boundary before shared memory is recycled.
        ptx += ['bar.sync 0;']
    ptx += ['shr.u32 %r80,%r0,2;and.b32 %r81,%r0,3;mul.lo.u32 %r81,%r81,2;',
            'add.u32 %r80,%r80,%r1;add.u32 %r81,%r81,%r2;']
    for q in range(4):
        for ni in range(2):
            for j, (dr, dc) in enumerate(((0,0),(0,1),(8,0),(8,1))):
                ptx += [f'add.u32 %r82,%r80,{(q//2)*16+dr};mad.lo.u32 %r83,%r82,{n},%r81;',
                        f'add.u32 %r83,%r83,{(q%2)*16+ni*8+dc};mul.wide.u32 %d3,%r83,4;',
                        f'add.u64 %d4,%d2,%d3;st.global.f32 [%d4],%c{q*8+ni*4+j};']
    return '\n'.join(ptx + ['ret;', '}']) + '\n'


def configure():
    backend.install()
    backend.TensorSelector = LowrankTensorSelector
    backend.OPTIMIZE_LOCALS = False
    backend.HOIST_LOOP_INVARIANTS = False
    backend.MINED_SCALAR = True
    backend.RELAX_TENSOR = True
    backend.TENSOR_ISSUE_GAP = 8
    backend.LDSM_READ_BARRIER = False
    backend.INLINE_POINTER_INCREMENTS = False
    backend.TENSOR_REUSE = True
    backend.LIST_SCHEDULE = False
    backend.EXPERIMENTAL_RECYCLE_WITHOUT_WAIT = False


def compile_one(scheme, m=96, n=96):
    from openassembler.cli import main
    configure()
    name = f'lowrank_{scheme}_{m}x{n}x32'
    source = HERE / f'{name}.ptx'
    source.write_text(lowrank_gemm(m, n, scheme))
    output = HERE / 'lowrank_build' / name
    backend.STAGE_RECORDS.clear()
    start = time.perf_counter()
    rc = main(['compile', str(source), '--out-dir', str(output), '--opt-level', 'safe'])
    if rc:
        raise RuntimeError(f'{name}: compilation failed ({rc})')
    rank = 7 if scheme == 'strassen' else 8
    record = {'compiler':'OpenAssembler experiment-local packed transform selector',
              'fallback':False, 'cli_wall_ms':(time.perf_counter()-start)*1000,
              'm':m,'n':n,'k':32,'tile_m':32,'tile_n':32,'threads':32,'static_shared':1024,
              'rank':rank,'HMMA_per_CTA':rank*2,'max_live_product_tiles':1,
              'cached_base_quadrants':8,'device_coefficient_loads':0,
              'global_transform_buffers':0,'kernels':backend.STAGE_RECORDS.copy(),
              'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'precision_policy':'FP16 inputs; packed FP16 RN sums of at most two terms, equivalent to FP32 sum then one FP16 quantization; FP32 HMMA products and ordered FP32 reconstruction; FP32 output',
              'limitations':'one warp, K=32, full tiles; algorithm-only control, not competitive large GEMM'}
    (output / 'experiment.json').write_text(json.dumps(record, indent=2)+'\n')
    return dict(name=name, scheme=scheme, build=str(output.relative_to(BASE)),
                m=m,n=n,k=32,tile_m=32,tile_n=32,threads=32,smem=0,dtype='f16')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--m',type=int,default=96)
    parser.add_argument('--n',type=int,default=96)
    args = parser.parse_args()
    cases = [compile_one(s,args.m,args.n) for s in ('strassen','naive')]
    (HERE / 'lowrank_suite.json').write_text(json.dumps(cases,indent=2)+'\n')
