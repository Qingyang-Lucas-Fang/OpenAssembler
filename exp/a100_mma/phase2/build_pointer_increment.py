"""Lower loop-carried 64-bit pointer increments directly into their homes."""
import json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE))
import backend
from build_pipeline import compile_one
if __name__=='__main__':
    backend.install();backend.TENSOR_ISSUE_GAP=8;backend.LDSM_READ_BARRIER=False
    backend.EXPERIMENTAL_RECYCLE_WITHOUT_WAIT=False
    suite=[]
    for name,enabled,wn,window in (('p2_ptr_base',False,64,0),('p2_ptr_inline',True,64,0),
                                   ('p2_ptr_w32_inline',True,32,1),('p2_ptr_w64_window4_inline',True,64,4)):
        backend.INLINE_POINTER_INCREMENTS=enabled
        suite.append(compile_one(name,4096,4096,4096,'f16',128,128,64,64,wn,compact=True,b_window=window))
    root=suite[0];root['name']='p2_pointer_paired';root['steady_warmup_rounds']=16;root['alternatives']=suite[1:]
    (BASE/'phase2/pointer-suite.json').write_text(json.dumps([root],indent=2)+'\n')
