import json,sys
from pathlib import Path
import modal
HERE=Path(__file__).resolve().parent
app=modal.App('openassembler-a100-autograd-experiment')
image=modal.Image.from_registry('pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel').add_local_python_source('openassembler',copy=False)
image=image.add_local_file(HERE/'integration_worker.py','/root/exp/integration_worker.py')
@app.function(image=image,gpu='A100-40GB',timeout=240,max_containers=1,retries=0)
def execute(cases):
    import subprocess,os
    root=Path('/root/exp');result=[]
    for i,case in enumerate(cases):
        source=root/f'integration{i}.json';dest=root/f'result{i}.json';source.write_text(json.dumps(case))
        p=subprocess.run([sys.executable,str(root/'integration_worker.py'),str(source),str(dest)],
                         capture_output=True,text=True,timeout=90,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        record=json.loads(dest.read_text()) if dest.exists() else {'error':p.stderr}
        record['name']=case['name'];result.append(record)
    return result
@app.local_entrypoint()
def main():
    result=execute.remote(json.loads((HERE/'integration-cases.json').read_text()))
    (HERE/'integration-results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if any('error' in r for r in result):raise SystemExit(1)
