"""Stable tanh-GELU derivative experiment transport; builds stay local and isolated.

.venv/bin/python exp/nn_round2/operator_gelu_stable.py --build
PYTHONPATH=. python3 -m modal run exp/nn_round2/operator_gelu_stable_modal.py
"""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import modal

_FILE=Path(__file__).resolve()
ROOT=_FILE.parents[2] if len(_FILE.parents)>2 else Path('/root')
HERE=ROOT/'exp/nn_round2'
app=modal.App('openassembler-nn-round2-stable-gelu')
image=(modal.Image.from_registry('pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel')
       .add_local_dir(str(ROOT/'openassembler'),'/root/openassembler',copy=True,ignore=['**/__pycache__','**/*.pyc'])
       .add_local_file(str(HERE/'operator_worker.py'),'/root/exp/nn_round2/operator_worker.py',copy=True)
       .add_local_file(str(HERE/'operator_gelu_stable.py'),'/root/exp/nn_round2/operator_gelu_stable.py',copy=True))


@app.function(image=image,gpu='A100-40GB',timeout=1800,max_containers=1,retries=0)
def execute(payload,options):
    import sys
    sys.path.insert(0,'/root')
    from exp.nn_round2.operator_gelu_stable import run
    return run(payload,**options)


@app.local_entrypoint()
def main(samples:int=19,rounds:int=2):
    payload=json.loads((HERE/'operator_gelu_stable_build/payload.json').read_text())
    options={'samples':samples,'rounds':rounds,'graph_repeats':32}
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    path=HERE/'runs'/f'operator-gelu-stable-{stamp}.json';path.parent.mkdir(parents=True,exist_ok=True)
    record={'schema':1,'phase':'operator-gelu-stable','status':'running','started_utc':datetime.now(timezone.utc).isoformat(),'options':options,'harness_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/'operator_gelu_stable.py',HERE/'operator_gelu_stable_modal.py',HERE/'operator_worker.py')},'payload_sha256':hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()}
    path.write_text(json.dumps(record,indent=2)+'\n')
    try:record.update(status='completed',result=execute.remote(payload,options))
    except BaseException as exc:record.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        record['finished_utc']=datetime.now(timezone.utc).isoformat();path.write_text(json.dumps(record,indent=2)+'\n');print(f'evidence: {path}',flush=True)
