"""Profile arbitrary retained suite entries by index; never infer geometry.

Index0 is the root; indices1..N are its alternatives. Each selected OA
kernel is compared with ptxas on its own exact source PTX through the existing
phase2 remote function. Plans and reports stay under phase3/profiles/<build>.
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

app=modal.App('openassembler-a100-phase3-generic-counters')
app.include(phase2.app)


def prepare(suite,case,indices='0,1'):
    path=(BASE/suite).resolve()
    if not path.is_relative_to(BASE):raise ValueError('keep suite within exp/a100_mma')
    roots=json.loads(path.read_text())
    matches=[r for r in roots if case in (r.get('name'),r.get('build'))]
    if not matches and str(case).isdigit() and int(case)<len(roots):matches=[roots[int(case)]]
    if len(matches)!=1:raise ValueError(f'expected one suite case: {case}')
    root=matches[0];entries=[root,*root.get('alternatives',[])]
    selected=list(range(len(entries))) if indices=='all' else [int(i) for i in indices.split(',')]
    if (not selected or len(selected)!=len(set(selected))
            or any(i<0 or i>=len(entries) for i in selected)):
        raise ValueError(f'indices must be unique within0..{len(entries)-1}, or all')
    result=[]
    for index in selected:
        config=dict(entries[index]);config.pop('alternatives',None)
        if config.get('dtype') not in ('f16','bf16'):raise ValueError('need a GEMM input dtype f16/bf16')
        payload=build_case(config)
        if payload.get('compiled_variants'):raise AssertionError('one kernel per profile target')
        directory=(HERE/'profiles'/config['build']).resolve()
        if not directory.is_relative_to(HERE/'profiles'):raise ValueError('unsafe build path')
        plan=dict(suite=str(path.relative_to(BASE)),suite_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  suite_case=root.get('name',root['build']),index=index,config=config,
                  dtype=payload['dtype'],shape_mnk=payload['shape'],grid=payload['grid'],
                  threads=payload['threads'],dynamic_shared_bytes=payload['smem'],
                  profile_contract='OA and ptxas generated from this exact same source PTX',
                  remote_function='phase2/profile_modal.py:profile',requested_metrics=phase2.METRICS,
                  input_provenance=payload['input_provenance'])
        result.append((directory,plan,payload))
    return result


def save(directory,plan,result=None):
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    if result is None:return
    for method in ('ours','ptxas'):
        record=result.get(method,{})
        if 'report_base64' in record:
            (directory/f'{method}.ncu-rep').write_bytes(base64.b64decode(record.pop('report_base64')))
        (directory/f'{method}.csv').write_text(record.get('raw_csv',record.get('stdout','')))
    result['phase3_plan']={key:plan[key] for key in
                          ('suite','suite_case','index','dtype','shape_mnk','grid','threads','dynamic_shared_bytes')}
    (directory/'profile.json').write_text(json.dumps(result,indent=2)+'\n')


@app.local_entrypoint()
def main(suite:str='phase3/address-stage-confirm-suite.json',
         case:str='address_stage_confirm_tall_f16',indices:str='0,1',dry_run:bool=False):
    for directory,plan,payload in prepare(suite,case,indices):
        save(directory,plan)
        print(json.dumps({key:plan[key] for key in
                         ('suite_case','index','dtype','shape_mnk','grid','threads','dynamic_shared_bytes')},indent=2))
        if dry_run:
            print('Dry run: no GPU call; saved',directory/'plan.json')
            continue
        result=phase2.profile.remote(payload)
        save(directory,plan,result)
        print(json.dumps(dict(build=plan['config']['build'],results={name:{k:v for k,v in result.get(name,{}).items()
                     if k in ('rc','stderr','error','import_rc')} for name in ('ours','ptxas')}),indent=2))
