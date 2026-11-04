"""Bounded standalone A100 runner for the fused-transform proof of lowering."""
from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path

import modal

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
APP=modal.App('openassembler-a100-lowrank-staging-exp')
IMAGE=modal.Image.from_registry('pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel')
IMAGE=IMAGE.add_local_python_source('openassembler',copy=False)
IMAGE=IMAGE.add_local_file(HERE/'lowrank_worker.py','/root/lowrank_worker.py')
IMAGE=IMAGE.add_local_file(HERE/'lowrank_oracle.py','/root/lowrank_oracle.py')


@APP.function(image=IMAGE,gpu='A100-40GB',timeout=180,max_containers=1,retries=0)
def execute(payload):
    from lowrank_worker import safe_run
    return safe_run(payload)


def prepare():
    cases=json.loads((HERE/'lowrank_suite.json').read_text())
    if len(cases)!=2 or {x['scheme'] for x in cases}!={'strassen','naive'}:
        raise ValueError('require both rank7 and rank8 matched-generator controls')
    m,n=cases[0]['m'],cases[0]['n']
    payload={'m':m,'n':n,'variants':[],'input_provenance':[]}
    for case in cases:
        if (case['m'],case['n'],case['k'])!=(m,n,32):
            raise ValueError('match all control shapes')
        directory=(BASE/case['build']).resolve()
        if not directory.is_relative_to(HERE):
            raise ValueError('keep lowrank builds in phase3')
        entry=json.loads((directory/'manifest.json').read_text())['entries'][0]
        provenance={'name':case['name'],'compilation':json.loads((directory/'experiment.json').read_text())}
        for compiler,key in (('ours','cubin'),('ptxas','reference')):
            binary=(directory/entry[key]).read_bytes()
            digest=hashlib.sha256(binary).hexdigest()
            provenance[f'{compiler}_sha256']=digest
            payload['variants'].append({'name':f'{case["scheme"]}_{compiler}',
                'scheme':case['scheme'],'cubin':base64.b64encode(binary).decode(),'sha256':digest})
        payload['input_provenance'].append(provenance)
    return payload


@APP.local_entrypoint()
def main():
    result=execute.remote(prepare())
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    result['recorded_utc']=stamp
    output=HERE/f'lowrank_gpu_{stamp}.json'
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(output)
    print(json.dumps({'passed':result['passed'],'paired_speed_ratios':result.get('paired_speed_ratios'),
                      'error':result.get('error')},indent=2))
    if not result['passed']:raise SystemExit(1)
