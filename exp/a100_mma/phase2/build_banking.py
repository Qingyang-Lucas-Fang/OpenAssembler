"""Asymmetric A/B shared layouts tested without altering physical bank hardware."""
import json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE))
import backend
from build_pipeline import compile_one
if __name__=='__main__':
    backend.install();backend.TENSOR_ISSUE_GAP=8;backend.LDSM_READ_BARRIER=False
    suite=[]
    for amask,bmask in ((7,7),(7,3),(3,7),(0,0)):
        name=f'p2_banks_a{amask}_b{bmask}'
        suite.append(compile_one(name,4096,4096,4096,'f16',128,128,64,64,64,compact=True,
                                 swizzle_a=amask,swizzle_b=bmask))
    root=suite[0];root['name']='p2_banking_paired';root['steady_warmup_rounds']=16
    root['alternatives']=suite[1:]
    (BASE/'phase2/banking-suite.json').write_text(json.dumps([root],indent=2)+'\n')
