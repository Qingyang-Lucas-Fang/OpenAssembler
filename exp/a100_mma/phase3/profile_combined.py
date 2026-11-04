"""Profile retained phase3 suite cases with the existing phase2 remote probe.

Default: FP16 square scalar baseline and row+B-pair kernel, each alongside
ptxas compiling that same PTX. No dtype, launch geometry or dynamic shared
memory is reconstructed from assumptions. --dry-run performs no GPU call.
"""
from __future__ import annotations
import base64
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
sys.path.insert(0,str(BASE))
sys.path.insert(0,str(BASE/'phase2'))
import modal
import profile_modal as phase2
from modal_run import build_case

app=modal.App('openassembler-a100-phase3-counters')
app.include(phase2.app)


def prepare(suite='phase3/search_combined_suite.json',case='search_combined_square_f16',
            variants='baseline,row_bpair'):
    path=(BASE/suite).resolve()
    if not path.is_relative_to(BASE):raise ValueError('suite must stay in exp/a100_mma')
    roots=json.loads(path.read_text())
    matches=[root for root in roots if case in (root.get('name'),root.get('build'))]
    if len(matches)!=1:raise ValueError(f'expected one suite case named {case}')
    root=matches[0]
    entries=[root,*root.get('alternatives',[])]
    labels=['baseline','row','bpair','row_bpair','row_bpair_role1','row_bpair_gap4']
    if len(entries)!=len(labels):raise ValueError('expected the six-candidate phase3 composition suite')
    selected=labels if variants=='all' else variants.split(',')
    if len(selected)!=len(set(selected)) or any(label not in labels for label in selected):
        raise ValueError(f'variants must be all or a unique comma-separated subset of {labels}')
    result=[]
    for label in selected:
        config=dict(entries[labels.index(label)])
        config.pop('alternatives',None)
        for key in ('dtype','m','n','k','threads','smem','tile_m','tile_n'):
            if config[key]!=root[key]:raise ValueError(f'unmatched suite contract for {key}')
        if config['dtype'] not in ('f16','bf16'):raise ValueError('unsupported input dtype')
        payload=build_case(config)
        if payload.get('compiled_variants'):raise AssertionError('profile one retained kernel at a time')
        result.append((label,payload))
    plan=dict(suite=str(path.relative_to(BASE)),suite_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
              case=root['name'],dtype=root['dtype'],shape_mnk=[root[k] for k in ('m','n','k')],
              grid=result[0][1]['grid'],threads=root['threads'],dynamic_shared_bytes=root['smem'],
              profile_contract='each OA kernel and ptxas on that exact same source PTX',
              remote_function='phase2/profile_modal.py:profile',
              requested_metrics=phase2.METRICS,
              candidates={label:payload['input_provenance'] for label,payload in result})
    return plan,result


def save_result(directory,result):
    directory.mkdir(parents=True,exist_ok=True)
    for method in ('ours','ptxas'):
        record=result.get(method,{})
        if 'report_base64' in record:
            (directory/f'{method}.ncu-rep').write_bytes(base64.b64decode(record.pop('report_base64')))
        (directory/f'{method}.csv').write_text(record.get('raw_csv',record.get('stdout','')))
    (directory/'profile.json').write_text(json.dumps(result,indent=2)+'\n')


@app.local_entrypoint()
def main(suite:str='phase3/search_combined_suite.json',case:str='search_combined_square_f16',
         variants:str='baseline,row_bpair',dry_run:bool=False):
    plan,payloads=prepare(suite,case,variants)
    output=HERE/'profiles'/case
    output.mkdir(parents=True,exist_ok=True)
    (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps({k:plan[k] for k in ('case','dtype','shape_mnk','grid','threads','dynamic_shared_bytes')},indent=2))
    if dry_run:
        print('Dry run: no GPU call; saved',output/'plan.json')
        return
    for label,payload in payloads:
        result=phase2.profile.remote(payload)
        result['phase3_plan']=dict(case=case,variant=label,dtype=plan['dtype'],
                                  shape_mnk=plan['shape_mnk'],dynamic_shared_bytes=plan['dynamic_shared_bytes'])
        save_result(output/label,result)
        print(json.dumps(dict(variant=label,results={name:{k:v for k,v in result.get(name,{}).items()
                     if k in ('rc','stderr','error','import_rc')} for name in ('ours','ptxas')}),indent=2))
