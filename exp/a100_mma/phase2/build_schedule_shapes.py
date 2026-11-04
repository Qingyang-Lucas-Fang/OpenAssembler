"""Matched scheduling, fragment windows, reuse hints and tile-shape alternatives."""
import json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE))
import backend
from build_pipeline import compile_one
backend.install();backend.TENSOR_ISSUE_GAP=8;backend.LDSM_READ_BARRIER=False
backend.INLINE_POINTER_INCREMENTS=True
backend.EXPERIMENTAL_RECYCLE_WITHOUT_WAIT=False
suite=[]
for name,wm,wn,bk,window,reuse,listing in (
    ('schedule_row_base',64,64,64,0,False,False),
    ('schedule_column_all',64,64,64,8,False,False),
    ('schedule_column_reuse',64,64,64,8,True,False),
    ('schedule_window4_reuse',64,64,64,4,True,False),
    ('schedule_list_safe',64,64,64,0,False,True),
    ('shape_w32x64',32,64,64,1,False,False),
    ('shape_w32x32',32,32,32,1,False,False),
):
    backend.TENSOR_REUSE=reuse;backend.LIST_SCHEDULE=listing
    suite.append(compile_one('p2_'+name,4096,4096,4096,'f16',128,128,bk,wm,wn,compact=True,b_window=window))
root=suite[0];root['name']='p2_scheduling_paired';root['steady_warmup_rounds']=16;root['alternatives']=suite[1:]
(BASE/'phase2/scheduling-suite.json').write_text(json.dumps([root],indent=2)+'\n')
