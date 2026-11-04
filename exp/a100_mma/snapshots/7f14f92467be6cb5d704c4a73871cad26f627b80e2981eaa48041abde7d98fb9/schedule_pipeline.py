"""Measure scheduling alternatives on the strongest generated full-tile shape."""
import json
from backend import HERE, install
import backend
from build_pipeline import compile_one

if __name__=='__main__':
    install();suite=[]
    for name,listing,read_bar,gap in (
        ('sched_compact_list8',True,True,8),
        ('sched_compact_list4',True,True,4),
        ('sched_compact_no_read8',False,False,8),
        ('sched_compact_list_no_read8',True,False,8),
    ):
        backend.LIST_SCHEDULE=listing
        backend.LDSM_READ_BARRIER=read_bar
        backend.TENSOR_ISSUE_GAP=gap
        suite.append(compile_one(name,4096,4096,4096,'f16',128,128,64,64,64,compact=True))
    (HERE/'schedule-suite.json').write_text(json.dumps(suite,indent=2)+'\n')
