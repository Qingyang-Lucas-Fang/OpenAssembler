"""Clocked single-warp HMMA experiments, with SASS and expected checksums."""
import argparse,dataclasses,hashlib,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE))
import backend
from backend import encoder
from openassembler import build,oracle
from openassembler.ptx.parse import parse_module
from openassembler.isa.word import Control


def generate(family,n):
    groups=1 if family=='dependent' else 8
    text=['.version 8.7\n.target sm_80\n.address_size 64',
          '.visible .entry gemm(.param .u64 out,.param .u64 inp,.param .u32 k,.param .f32 kf) {',
          f'.reg .b32 %r<30>; .reg .b64 %d<8>; .reg .f32 %c<{groups*4}>;',
          'ld.param.u64 %d0,[out]; mov.u32 %r0,%tid.x; mul.wide.u32 %d1,%r0,16; add.u64 %d2,%d0,%d1;']
    text += [f'mov.b32 %r{i},0x3c003c00;' for i in range(1,7)]
    text += [f'mov.f32 %c{i},0f00000000;' for i in range(groups*4)]
    for i in range(n):
        g=i%groups;cc=','.join(f'%c{g*4+j}' for j in range(4))
        text += ['mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 '
                 f'{{{cc}}},{{%r1,%r2,%r3,%r4}},{{%r5,%r6}},{{{cc}}};']
    for g in range(1,groups):text += [f'add.f32 %c0,%c0,%c{g*4};']
    text += ['st.global.f32 [%d2+4],%c0; ret; }']
    ptx='\n'.join(text)+'\n'
    backend.RELAX_TENSOR=True;backend.MINED_SCALAR=True;backend.OPTIMIZE_LOCALS=False
    backend.LDSM_READ_BARRIER=False;backend.LIST_SCHEDULE=False;backend.TENSOR_ISSUE_GAP=2
    code,labels,words,exits,regs=backend.generate(parse_module(ptx).kernels['gemm'],'sm_80','safe')
    positions=[i for i,(t,c) in enumerate(code) if t.startswith('HMMA')]
    start,end=positions[0],positions[-1]+2  # include epoch drain NOP before clock observer
    enc=encoder();fixed=lambda stall:Control(stall,stall>11,7,7,0,0)
    transformed=[]
    for i,item in enumerate(code):
        if i==start:transformed.append(('CS2R.32 R240, SR_CLOCKLO',fixed(6)))
        if i==end:
            transformed += [('CS2R.32 R241, SR_CLOCKLO',fixed(6)),
                            ('IADD3 R242, R241, -R240, RZ',fixed(6))]
        # Clone the checksum store address; R242 is raw u32 cycle count.
        if item[0].startswith('STG.E'):
            prefix=item[0].rsplit(', ',1)[0].replace('+0x4','')
            transformed.append((prefix+', R242',item[1]))
        transformed.append(item)
    if labels:raise AssertionError('clock probe must be straight line')
    words=ra_encode(enc,transformed)
    ref=build.ptxas_build(ptx,'sm_80')
    exits=[16*i for i,(t,c) in enumerate(transformed) if t=='EXIT']
    binary=build.splice_kernels(ref,{'gemm':(words,246,exits)})
    return ptx,transformed,binary


def ra_encode(enc,code):
    from openassembler.backend import ra
    words=ra.encode_code(enc,code,{})
    words.append(enc.encode_branch('BRA `(.L_x_0)',16*len(words),16*len(words)))
    while len(words)%8:words.append(enc.encode('NOP'))
    return words


if __name__=='__main__':
    backend.prepare();out=BASE/'phase2/clocked';out.mkdir(exist_ok=True)
    suite=[]
    for family in ('dependent','independent'):
        for n in (8,32,64,128,256):
            ptx,code,binary=generate(family,n);stem=out/f'{family}-{n}'
            stem.with_suffix('.ptx').write_text(ptx);stem.with_suffix('.cubin').write_bytes(binary)
            stem.with_suffix('.sass').write_text(build.disasm(binary))
            suite.append(dict(name=f'clock_{family}_{n}',kind='clock',family=family,steps=n,
                              cubin_path=str(stem.with_suffix('.cubin').relative_to(BASE)),
                              sha256=hashlib.sha256(binary).hexdigest()))
    (out/'suite.json').write_text(json.dumps(suite,indent=2)+'\n')
