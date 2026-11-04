"""Repeat the candidate compiler improvement on FP16/BF16 projection shapes."""
import json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE))
import backend
from build_pipeline import compile_one
backend.install();backend.TENSOR_ISSUE_GAP=8;backend.LDSM_READ_BARRIER=False
suite=[]
for dtype in ('f16','bf16'):
    for tag,m,n in (('square',4096,4096),('mlp',4096,16384),('tall',16384,4096)):
        alternatives=[]
        for enabled in (False,True):
            backend.INLINE_POINTER_INCREMENTS=enabled
            name=f'p2_confirm_{tag}_{dtype}_{"inline" if enabled else "base"}'
            alternatives.append(compile_one(name,m,n,4096,dtype,128,128,64,64,64,compact=True))
        root=alternatives[0];root['name']=f'p2_confirm_{tag}_{dtype}';root['steady_warmup_rounds']=16
        root['alternatives']=alternatives[1:];suite.append(root)
(BASE/'phase2/pointer-validation-suite.json').write_text(json.dumps(suite,indent=2)+'\n')
