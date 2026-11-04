"""Recreate confirmation/final/Lt-control manifests from validated build suites."""
from pathlib import Path
import copy
import json

HERE=Path(__file__).resolve().parent

def main():
    old=json.loads((HERE/'search_combined_suite.json').read_text())
    stages=json.loads((HERE/'address-stage-confirm-suite.json').read_text())
    if len(old)!=6 or len(stages)!=6:raise ValueError('expected six projection cases')
    final=[];confirmation=[];lt=[]
    for baseline,stage in zip(old,stages):
        for key in ('dtype','m','n','k','tile_m','tile_n','threads','smem'):
            if baseline[key]!=stage[key]:raise ValueError(f'mismatched {key}')
        if len(stage['alternatives'])!=1:raise ValueError('expected one fixed stage schedule')
        row=copy.deepcopy(stage);row.pop('alternatives')
        root=copy.deepcopy(baseline)
        root['name']=root['name'].replace('search_combined','p3_final')
        root['alternatives']=[row,*copy.deepcopy(stage['alternatives'])]
        root['steady_warmup_rounds']=24
        final.append(root)
        confirmed=copy.deepcopy(baseline)
        confirmed['name']=confirmed['name'].replace('search_combined','p3_confirm')
        confirmed['alternatives']=[x for x in confirmed['alternatives']
            if '_row_bpair_' in x['build'] and '_role' not in x['build'] and '_gap' not in x['build']]
        if len(confirmed['alternatives'])!=1:raise ValueError('expected one fixed paired schedule')
        confirmed['steady_warmup_rounds']=24
        confirmation.append(confirmed)
        if baseline['m']==baseline['n']==baseline['k']==4096:
            lt.append(copy.deepcopy(confirmed['alternatives'][0]))
    for name,suite in [('final_suite.json',final),('confirmation_suite.json',confirmation),
                       ('lt_controls_suite.json',lt)]:
        (HERE/name).write_text(json.dumps(suite,indent=2)+'\n')
        print(name,len(suite))

if __name__=='__main__':main()
