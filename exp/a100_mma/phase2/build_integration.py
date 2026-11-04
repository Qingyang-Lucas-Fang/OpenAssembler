"""Compile all three autograd GEMM shapes for non-tile-aligned dimensions."""
import base64,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE))
import backend
from build_pipeline import compile_one

if __name__=='__main__':
    backend.install();backend.TENSOR_ISSUE_GAP=8;backend.LDSM_READ_BARRIER=False
    shapes=set()
    m,n,k=130,198,257
    round128=lambda x:(x+127)//128*128
    round64=lambda x:max(128,(x+63)//64*64)
    for mm,nn,kk in ((m,n,k),(m,k,n),(n,k,m)):shapes.add((round128(mm),round128(nn),round64(kk)))
    cases=[]
    for dtype in ('f16','bf16'):
        binaries=[]
        for mm,nn,kk in sorted(shapes):
            fused=(mm,nn,kk)==(round128(m),round128(n),round64(k))
            name=f'p2_hook_fused_{mm}_{nn}_{kk}_{dtype}'
            config=compile_one(name,mm,nn,kk,dtype,128,128,64,64,64,compact=True,bias=fused)
            root=BASE/'build'/name;manifest=json.loads((root/'manifest.json').read_text());entry=manifest['entries'][0]
            binaries.append({'shape':[mm,nn,kk],'binary':base64.b64encode((root/entry['cubin']).read_bytes()).decode(),
                             'smem':config['smem'],'build':name,'bias_fused':fused})
        cases.append({'name':f'linear_boundary_{dtype}','dtype':dtype,'logical_shape':[m,n,k],'binaries':binaries})
    (BASE/'phase2/integration-cases.json').write_text(json.dumps(cases,indent=2)+'\n')
