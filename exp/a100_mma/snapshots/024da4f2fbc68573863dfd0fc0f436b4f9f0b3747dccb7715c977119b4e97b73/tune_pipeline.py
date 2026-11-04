"""Full-output-checked geometry and data-movement variants; no factorization yet."""
import json
from backend import HERE, install
import backend
from build_pipeline import compile_one

if __name__=='__main__':
    install()
    suite=[];failures=[]
    backend.TENSOR_ISSUE_GAP=8
    backend.LDSM_READ_BARRIER=True
    for bm,bn,bk,wm,wn in ((64,64,32,32,32),(64,128,32,32,64),(128,128,32,64,32),
                           (128,128,32,64,64),(128,128,64,64,32),(128,128,64,64,64)):
        name=f'tuned3_{bm}_{bn}_{bk}_{wm}_{wn}'
        try:
            suite.append(compile_one(name,4096,4096,4096,'f16',bm,bn,bk,wm,wn))
        except RuntimeError as exc:
            failures.append({'name':name,'compile_error':str(exc)})
    (HERE/'tuned-suite.json').write_text(json.dumps(suite,indent=2)+'\n')
    (HERE/'tuned-compile-failures.json').write_text(json.dumps(failures,indent=2)+'\n')
