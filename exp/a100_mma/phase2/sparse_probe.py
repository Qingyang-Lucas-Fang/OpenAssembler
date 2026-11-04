"""Native A100 sparse-MMA capability oracle; not an OA sparse compiler or GEMM."""
import hashlib,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE))
from backend import prepare
from openassembler import build
prepare()
root=BASE/'phase2/sparse';root.mkdir(exist_ok=True);suite=[]
for tag,half,expected in (('ones',0x3c003c00,16.0),('negative_b',0xbc00bc00,-16.0),('zero_b',0,0.0)):
    ptx=f'''.version 8.7
.target sm_80
.address_size 64
.visible .entry gemm(.param .u64 out,.param .u64 inp,.param .u32 k,.param .f32 kf) {{
.reg .b32 %a<4>,%b<4>,%meta,%rThread,%offset;
.reg .b64 %ptr,%address,%off;
.reg .f32 %c<4>;
ld.param.u64 %ptr,[out]; mov.u32 %rThread,%tid.x;
mul.wide.u32 %off,%rThread,16;add.u64 %address,%ptr,%off;
mov.b32 %a0,0x3c003c00;mov.b32 %a1,0x3c003c00;mov.b32 %a2,0x3c003c00;mov.b32 %a3,0x3c003c00;
mov.b32 %b0,{half};mov.b32 %b1,{half};mov.b32 %b2,{half};mov.b32 %b3,{half};
mov.u32 %meta,0x44444444;
mov.f32 %c0,0f00000000;mov.f32 %c1,0f00000000;mov.f32 %c2,0f00000000;mov.f32 %c3,0f00000000;
mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32
{{%c0,%c1,%c2,%c3}},{{%a0,%a1,%a2,%a3}},{{%b0,%b1,%b2,%b3}},{{%c0,%c1,%c2,%c3}},%meta,0;
st.global.v2.f32 [%address],{{%c0,%c1}};ret;
}}
'''
    path=root/f'{tag}.ptx';path.write_text(ptx)
    try:
        binary=build.ptxas_build(ptx,'sm_80');path.with_suffix('.cubin').write_bytes(binary)
        path.with_suffix('.sass').write_text(build.disasm(binary))
        import struct
        suite.append(dict(name=f'sparse_{tag}',family='native-sparse-capability',steps=1,
                          expected_raw=struct.unpack('<I',struct.pack('<f',expected))[0],
                          cubin_path=str(path.with_suffix('.cubin').relative_to(BASE)),
                          sha256=hashlib.sha256(binary).hexdigest(),compiler='NVIDIA ptxas capability oracle only'))
    except Exception as exc:
        (root/f'{tag}.error').write_text(str(exc)+'\n'+str(getattr(exc,'stderr','')))
(root/'suite.json').write_text(json.dumps(suite,indent=2)+'\n')
if len(suite)!=3:raise SystemExit('all three sparse capability probes must compile')
