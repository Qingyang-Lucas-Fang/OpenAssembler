"""Controlled register-reuse hint experiment on otherwise identical PTX."""
import json
from backend import HERE, install
import backend
from build_pipeline import compile_one

if __name__=='__main__':
    install();suite=[]
    backend.LDSM_READ_BARRIER=False
    backend.LIST_SCHEDULE=False
    for name,reuse,gap in (('reuse_baseline',False,8),('reuse8',True,8),('reuse4',True,4)):
        backend.TENSOR_ISSUE_GAP=gap
        backend.TENSOR_REUSE=reuse
        suite.append(compile_one(name,4096,4096,4096,'f16',128,128,64,64,64,compact=True))
    (HERE/'reuse-suite.json').write_text(json.dumps(suite,indent=2)+'\n')
