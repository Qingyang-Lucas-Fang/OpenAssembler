"""Test an undocumented scoreboard hypothesis; never enable by default."""
import json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE))
import backend
from build_pipeline import compile_one
if __name__=='__main__':
    backend.install();backend.TENSOR_ISSUE_GAP=8;backend.LDSM_READ_BARRIER=False
    backend.EXPERIMENTAL_RECYCLE_WITHOUT_WAIT=True
    suite=[]
    for wn in (32,64):
        suite.append(compile_one(f'p2_unsafe_recycle_w64x{wn}',4096,4096,4096,'f16',128,128,64,64,wn,compact=True))
    (BASE/'phase2/recycle-suite.json').write_text(json.dumps(suite,indent=2)+'\n')
