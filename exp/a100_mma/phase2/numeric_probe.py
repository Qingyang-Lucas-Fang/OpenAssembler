"""Paper-motivated counterexample to describing MMA as globally IEEE RTZ."""
import hashlib,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE))
import backend
from openassembler import build
from openassembler.ptx.parse import parse_module
backend.prepare()
ptx='''.version 8.7
.target sm_80
.address_size 64
.visible .entry gemm(.param .u64 out,.param .u64 inp,.param .u32 k,.param .f32 kf) {
.reg .b32 %r<10>; .reg .b64 %d<5>; .reg .f32 %c<4>;
ld.param.u64 %d0,[out];mov.u32 %r0,%tid.x;
mul.wide.u32 %d1,%r0,16;add.u64 %d2,%d0,%d1;
mov.b32 %r1,0x40004000;mov.b32 %r2,0x40004000;mov.b32 %r3,0x40004000;mov.b32 %r4,0x40004000;
mov.b32 %r5,0x3c003c00;mov.b32 %r6,0x3c003c00;
mov.f32 %c0,0fab800000;mov.f32 %c1,0fab800000;mov.f32 %c2,0fab800000;mov.f32 %c3,0fab800000;
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
{%c0,%c1,%c2,%c3},{%r1,%r2,%r3,%r4},{%r5,%r6},{%c0,%c1,%c2,%c3};
st.global.v2.f32 [%d2],{%c0,%c1};ret;
}
'''
root=BASE/'phase2/numeric';root.mkdir(exist_ok=True);(root/'truncation.ptx').write_text(ptx)
code,labels,words,exits,regs=backend.generate(parse_module(ptx).kernels['gemm'],'sm_80','safe')
reference=build.ptxas_build(ptx,'sm_80')
ours=build.splice_kernels(reference,{'gemm':(words,regs,exits)})
suite=[]
for tag,binary in (('ours',ours),('ptxas',reference)):
    p=root/f'{tag}.cubin';p.write_bytes(binary);p.with_suffix('.sass').write_text(build.disasm(binary))
    suite.append(dict(name=f'truncation_{tag}',family='numerical-capability',steps=1,expected_raw=0x42000000,
                      exact_real_expression='32 - 2**-40',global_IEEE_RTZ_f32=0x41ffffff,
                      cubin_path=str(p.relative_to(BASE)),sha256=hashlib.sha256(binary).hexdigest()))
(root/'suite.json').write_text(json.dumps(suite,indent=2)+'\n')
