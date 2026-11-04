"""Read-only SM80 profiler reconnaissance; explicit bounded Modal execution."""
import json
from pathlib import Path
import modal

HERE=Path(__file__).resolve().parent
app=modal.App('openassembler-a100-counter-research')
image=modal.Image.from_registry('pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel')

@app.function(image=image,gpu='A100-40GB',timeout=180,max_containers=1,retries=0)
def inspect():
    import subprocess,glob,shutil,os
    commands=[['nvidia-smi','-q'],['nvcc','--version'],['which','ncu'],
              ['cat','/proc/driver/nvidia/params'],['ls','/usr/local/cuda/extras/CUPTI/lib64'],
              ['bash','-c','dpkg -l | head -8']]
    result={'commands':[],'profilers':glob.glob('/opt/nvidia/nsight-compute*')+glob.glob('/usr/local/cuda*/nsight-compute*'),
            'cupti':glob.glob('/usr/local/cuda*/extras/CUPTI/include/*profiler*')}
    for command in commands:
        try:
            p=subprocess.run(command,text=True,capture_output=True,timeout=20)
            result['commands'].append({'command':command,'rc':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
        except OSError as exc:result['commands'].append({'command':command,'error':str(exc)})
    return result

@app.local_entrypoint()
def main():
    result=inspect.remote()
    (HERE/'capabilities.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
