"""Reduce fragment address registers using nvdisasm-validated LDSM offsets."""
import json
from backend import HERE, install
import backend
from build_pipeline import compile_one

if __name__=='__main__':
    install();suite=[];failed=[]
    backend.TENSOR_ISSUE_GAP=8
    backend.LDSM_READ_BARRIER=True
    for bm,bn,bk,wm,wn in ((64,64,32,32,32),(128,128,32,64,32),(128,128,32,64,64),
                           (128,128,64,64,32),(128,128,64,64,64)):
        name=f'compact_{bm}_{bn}_{bk}_{wm}_{wn}'
        try:
            suite.append(compile_one(name,4096,4096,4096,'f16',bm,bn,bk,wm,wn,compact=True))
        except RuntimeError as exc:
            failed.append({'name':name,'compile_error':str(exc)})
    (HERE/'compact-suite.json').write_text(json.dumps(suite,indent=2)+'\n')
    (HERE/'compact-compile-failures.json').write_text(json.dumps(failed,indent=2)+'\n')
