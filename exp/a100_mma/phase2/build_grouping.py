"""Virtual tile grouping for L2 locality, without claiming hardware clusters."""
import json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE))
import backend
from build_pipeline import compile_one
if __name__=='__main__':
    backend.install();backend.TENSOR_ISSUE_GAP=8;backend.LDSM_READ_BARRIER=False
    suite=[]
    for group in (0,2,4,8,16):
        suite.append(compile_one(f'p2_group{group}',4096,4096,4096,'f16',128,128,64,64,64,
                                 compact=True,group_m=group))
    root=suite[0];root['name']='p2_grouping_paired';root['steady_warmup_rounds']=16;root['alternatives']=suite[1:]
    (BASE/'phase2/grouping-suite.json').write_text(json.dumps([root],indent=2)+'\n')
