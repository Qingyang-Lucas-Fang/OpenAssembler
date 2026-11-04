"""Independent-accumulator issue sweep, complementing the dependent RAW chain."""
import dataclasses
import json
from backend import HERE, generate, encoder, prepare
from openassembler import build
from openassembler.ptx.parse import parse_module
from openassembler.isa.word import control


def source(groups=8,steps=16):
    text=['.version 8.7\n.target sm_80\n.address_size 64',
          '.visible .entry gemm(.param .u64 out,.param .u64 inp,.param .u32 k,.param .f32 kf) {',
          f'.reg .b32 %r<20>; .reg .b64 %d<10>; .reg .f32 %c<{groups*4}>;',
          'ld.param.u64 %d0,[out]; ld.param.u64 %d1,[inp]; mov.u32 %r0,%tid.x;',
          'mul.wide.u32 %d2,%r0,16; add.u64 %d3,%d1,%d2; add.u64 %d4,%d0,%d2;']
    text += [f'ld.global.v4.f32 {{{",".join(f"%c{g*4+j}" for j in range(4))}}},[%d3];'
             for g in range(groups)]
    text += [f'mov.b32 %r{i},0x3c003c00;' for i in range(1,7)]
    for step in range(steps):
        for g in range(groups):
            cc=','.join(f'%c{g*4+j}' for j in range(4))
            text += ['mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 '
                     f'{{{cc}}},{{%r1,%r2,%r3,%r4}},{{%r5,%r6}},{{{cc}}};']
    for g in range(1,groups):
        text += [f'add.f32 %c{j},%c{j},%c{g*4+j};' for j in range(4)]
    text += ['st.global.v4.f32 [%d4],{%c0,%c1,%c2,%c3}; ret; }']
    return '\n'.join(text)+'\n'


if __name__=='__main__':
    prepare();root=HERE/'hmma';root.mkdir(exist_ok=True)
    ptx=source();(root/'independent.ptx').write_text(ptx)
    code,labels,original,exits,regs=generate(parse_module(ptx).kernels['gemm'],'sm_80','safe')
    reference=build.ptxas_build(ptx,'sm_80')
    positions=[i for i,(t,c) in enumerate(code) if t.startswith('HMMA')]
    enc=encoder();suite=[]
    for gap in (2,3,4,5,6,7,8,10,12,16,24,30):
        words=list(original)
        for i in positions[:-1]:
            first=min(15,gap-1)
            for pos,stall in ((i,first),(i+1,gap-first)):
                words[pos]=enc._ctl(words[pos],dataclasses.replace(control(words[pos]),stall=stall,yield_=stall>11))
        target=root/f'independent-gap{gap:02d}.cubin'
        target.write_bytes(build.splice_kernels(reference,{'gemm':(words,regs,exits)}))
        suite.append(dict(kind='hmma',name=f'hmma_independent_gap{gap:02d}',gap=gap,steps=128,
                          input_factor=8,family='eight-independent-accumulators',
                          cubin_path=str(target.relative_to(HERE))))
    (HERE/'hmma-issue-suite.json').write_text(json.dumps(suite,indent=2)+'\n')
