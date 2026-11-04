"""Confirm tensor order/reuse changes against the pointer-optimized baseline."""
import json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE))
import backend
from build_pipeline import compile_one
backend.install();backend.TENSOR_ISSUE_GAP=8;backend.LDSM_READ_BARRIER=False
backend.INLINE_POINTER_INCREMENTS=True;backend.EXPERIMENTAL_RECYCLE_WITHOUT_WAIT=False
suite=[]
for dtype in ('f16','bf16'):
    for tag,m,n in (('square',4096,4096),('mlp',4096,16384),('tall',16384,4096)):
        candidates=[]
        for enabled in (False,True):
            backend.TENSOR_REUSE=enabled
            name=f'p2_schedconfirm_{tag}_{dtype}_{"column_reuse" if enabled else "row"}'
            candidates.append(compile_one(name,m,n,4096,dtype,128,128,64,64,64,compact=True,b_window=8 if enabled else 0))
        root=candidates[0];root['name']=f'p2_schedconfirm_{tag}_{dtype}';root['steady_warmup_rounds']=16
        root['alternatives']=candidates[1:];suite.append(root)
(BASE/'phase2/schedule-validation-suite.json').write_text(json.dumps(suite,indent=2)+'\n')
