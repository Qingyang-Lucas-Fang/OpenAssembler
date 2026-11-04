"""Read installed cuBLASLt ABI and observe every launch in matched GEMM calls."""
import json
from pathlib import Path
import modal

HERE=Path(__file__).resolve().parent
app=modal.App('openassembler-a100-cublas-dispatch')
image=modal.Image.from_registry('pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel')
image=image.add_local_file(HERE/'cublas_target.py','/root/exp/cublas_target.py')

@app.function(image=image,gpu='A100-40GB',timeout=360,max_containers=1,retries=0)
def probe():
    import glob,re,subprocess,sys,os
    out={'headers':{},'dispatch':[]}
    names=['cublasLtMatmulAlgo_t','cublasLtMatmulHeuristicResult_t',
           'CUBLASLT_MATMUL_DESC_TRANSA','CUBLASLT_MATMUL_DESC_TRANSB',
           'CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES','CUBLASLT_ALGO_CONFIG_ID',
           'CUBLASLT_ALGO_CONFIG_TILE_ID','CUBLASLT_ALGO_CONFIG_STAGES_ID',
           'CUBLASLT_ALGO_CONFIG_SPLITK_NUM','CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME']
    for path in glob.glob('/opt/conda/lib/python*/site-packages/nvidia/cublas/include/cublasLt.h')+glob.glob('/usr/local/cuda/include/cublasLt.h'):
        text=Path(path).read_text()
        out['headers'][path]={}
        for name in names:
            matches=list(re.finditer(r'\b'+name+r'\b',text))
            if matches:
                pos=matches[0].start()
                out['headers'][path][name]=text[max(0,pos-650):pos+450]
    for dtype in ('f16','bf16'):
        argv=['ncu','--section','LaunchStats','--launch-count','25',
              '--kernel-name-base','mangled','--csv','--clock-control','none',
              sys.executable,'/root/exp/cublas_target.py','--dtype',dtype]
        p=subprocess.run(argv,capture_output=True,text=True,timeout=140,
                         env=dict(os.environ,CUBLAS_LOGINFO_DBG='1',CUBLAS_LOGDEST_DBG='stderr'))
        out['dispatch'].append({'dtype':dtype,'argv':argv,'rc':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
    return out

@app.local_entrypoint()
def main():
    out=probe.remote()
    (HERE/'cublas-dispatch.json').write_text(json.dumps(out,indent=2)+'\n')
    for case in out['dispatch']:
        print(case['dtype'],case['rc'])
    print('Saved cublas-dispatch.json')
