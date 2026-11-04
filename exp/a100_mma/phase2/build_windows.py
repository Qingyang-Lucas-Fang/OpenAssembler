"""Bound fragment lifetimes to test occupancy thresholds and operand reuse."""
import json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE))
import backend
from build_pipeline import compile_one
if __name__=='__main__':
    backend.install();backend.TENSOR_ISSUE_GAP=8;backend.LDSM_READ_BARRIER=False
    suite=[]
    for wn in (32,64):
        for window in (0,1,2,4):
            name=f'p2_window{window}_w64x{wn}'
            suite.append(compile_one(name,4096,4096,4096,'f16',128,128,64,64,wn,compact=True,b_window=window))
    (BASE/'phase2/window-suite.json').write_text(json.dumps(suite,indent=2)+'\n')
