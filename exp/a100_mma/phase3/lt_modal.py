"""Run bounded, matched cuBLASLt sweeps on a single A100, one process per dtype."""
from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path

import modal

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
APP=modal.App('openassembler-a100-cublaslt-heuristics')
IMAGE=modal.Image.from_registry('pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel')
IMAGE=IMAGE.add_local_python_source('openassembler',copy=False)
IMAGE=IMAGE.add_local_file(HERE/'lt_worker.py','/root/lt_worker.py')
IMAGE=IMAGE.add_local_file(HERE/'lt_abi.py','/root/lt_abi.py')


@APP.function(image=IMAGE,gpu='A100-40GB',timeout=600,max_containers=1,retries=0)
def execute(configs):
    import os
    import subprocess
    import sys
    import tempfile
    from pathlib import Path
    results=[]
    with tempfile.TemporaryDirectory(prefix='oa-lt-') as temporary:
        root=Path(temporary)
        for config in configs:
            source=root/f'{config["dtype"]}-input.json'
            target=root/f'{config["dtype"]}-result.json'
            source.write_text(json.dumps(config))
            try:
                process=subprocess.run([sys.executable,'/root/lt_worker.py',str(source),str(target)],
                    timeout=260,capture_output=True,text=True,
                    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
                record=json.loads(target.read_text()) if target.exists() else {
                    'passed':False,'error':'worker returned without result JSON'}
                record.update({'worker_rc':process.returncode,'worker_stdout':process.stdout,
                               'worker_stderr':process.stderr,'dtype':config['dtype']})
            except subprocess.TimeoutExpired as exc:
                record={'passed':False,'dtype':config['dtype'],'error':'260s worker watchdog',
                        'worker_stdout':str(exc.stdout),'worker_stderr':str(exc.stderr)}
            results.append(record)
            print(config['dtype'],record.get('best_by_workspace_cap',record.get('error')),flush=True)
    return results


def native_controls(path,shape):
    if not path:return {}
    source=(BASE/path).resolve()
    if not source.is_relative_to(BASE):raise ValueError('baseline suite must stay inside experiment')
    controls={}
    for config in json.loads(source.read_text()):
        if [config[x] for x in ('m','n','k')]!=shape:
            raise ValueError('native controls must match exact matrix dimensions')
        root=(BASE/'build'/config['build']).resolve()
        if not root.is_relative_to(BASE):raise ValueError('native build must stay inside experiment')
        entry=json.loads((root/'manifest.json').read_text())['entries'][0]
        binary=(root/entry['cubin']).read_bytes()
        control={'name':config.get('name','openassembler'),'grid':[shape[0]//config['tile_m'],shape[1]//config['tile_n'],1],
            'threads':config['threads'],'smem':config.get('smem',0),
            'cubin':base64.b64encode(binary).decode(),'sha256':hashlib.sha256(binary).hexdigest(),
            'build':config['build']}
        controls.setdefault(config['dtype'],[]).append(control)
    return controls


def prepare(requested=32,size=4096,dtype='both',baseline_suite=''):
    if requested not in (16,32):raise ValueError('request16 or32 heuristics')
    if not 32<=size<=8192:raise ValueError('bounded square experiment supports32..8192')
    shape=[size,size,size]
    extras=native_controls(baseline_suite,shape)
    dtypes=('f16','bf16') if dtype=='both' else (dtype,)
    if any(x not in ('f16','bf16') for x in dtypes):raise ValueError('dtype must be f16,bf16,or both')
    source_hashes={path.name:hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in (HERE/'lt_abi.py',HERE/'lt_worker.py',Path(__file__))}
    return [{'dtype':dt,'shape':shape,'requested':requested,'graph_nodes':20,'warmup_rounds':8,
             'extras':extras.get(dt,[]),'runner_source_sha256':source_hashes} for dt in dtypes]


@APP.local_entrypoint()
def main(requested:int=32,size:int=4096,dtype:str='both',baseline_suite:str=''):
    configs=prepare(requested,size,dtype,baseline_suite)
    results=execute.remote(configs)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    for result in results:result['recorded_utc']=stamp
    target=HERE/f'lt_results_{stamp}.json'
    target.write_text(json.dumps(results,indent=2)+'\n')
    print(target)
    if any(not x.get('passed') for x in results):raise SystemExit(1)
