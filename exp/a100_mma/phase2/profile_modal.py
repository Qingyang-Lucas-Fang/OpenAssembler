"""Collect hardware evidence from identical OpenAssembler/ptxas binary inputs."""
import json
import sys
from pathlib import Path
import modal

HERE=Path(__file__).resolve().parent

app=modal.App('openassembler-a100-hardware-counters')
image=modal.Image.from_registry('pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel')
image=image.add_local_python_source('openassembler',copy=False)
image=image.add_local_file(HERE/'profile_target.py','/root/exp/profile_target.py')

METRICS=[
 'gpu__time_duration.sum',
 'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active',
 'sm__warps_active.avg.pct_of_peak_sustained_active',
 'smsp__inst_executed.sum',
 'smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct',
 'smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct',
 'smsp__warp_issue_stalled_barrier_per_warp_active.pct',
 'smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct',
 'smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct',
 'smsp__warp_issue_stalled_wait_per_warp_active.pct',
 'smsp__warp_issue_stalled_not_selected_per_warp_active.pct',
 'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum',
 'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum',
 'dram__bytes.sum','lts__t_sector_hit_rate.pct',
 'smsp__inst_executed_pipe_tensor.sum',
 'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum',
 'l1tex__t_requests_pipe_lsu_mem_shared_op_ld.sum',
 'smsp__warps_eligible.avg.per_cycle_active',
]

@app.function(image=image,gpu='A100-40GB',timeout=540,max_containers=1,retries=0)
def profile(case):
    import subprocess,os,glob,time
    root=Path('/root/exp');root.mkdir(exist_ok=True)
    (root/'case.json').write_text(json.dumps(case))
    result={'provenance':case.get('input_provenance'),'commands':[]}
    for argv in (['ncu','--version'],['ncu','--query-metrics']):
        p=subprocess.run(argv,capture_output=True,text=True,timeout=30)
        result['commands'].append({'argv':argv,'rc':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
    available=result['commands'][-1]['stdout']
    selected=[m for m in METRICS if m.split('.')[0] in available]
    result['selected_metrics']=selected
    result['unavailable_requested_metrics']=[m for m in METRICS if m not in selected]
    if not selected:raise RuntimeError('Nsight metric discovery failed; refusing an empty profile')
    for variant in ('ours','ptxas'):
        dest=root/variant
        argv=['ncu','--target-processes','all','--kernel-name','regex:^gemm$',
              '--launch-skip','3','--launch-count','1','--metrics',','.join(selected),
              '--clock-control','none','--cache-control','all','--csv',
              '--export',str(dest),'--force-overwrite',
              sys.executable,str(root/'profile_target.py'),str(root/'case.json'),variant]
        try:
            p=subprocess.run(argv,capture_output=True,text=True,timeout=200,
                             env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
            result[variant]={'argv':argv,'rc':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
            report=dest.with_suffix('.ncu-rep')
            if report.exists():
                import base64
                result[variant]['report_base64']=base64.b64encode(report.read_bytes()).decode()
                parsed=subprocess.run(['ncu','--import',str(report),'--page','raw','--csv'],
                                      capture_output=True,text=True,timeout=30)
                result[variant]['raw_csv']=parsed.stdout
                result[variant]['import_rc']=parsed.returncode
        except subprocess.TimeoutExpired as exc:
            result[variant]={'error':'profile timeout 200s','stdout':str(exc.stdout),'stderr':str(exc.stderr)}
    return result

@app.local_entrypoint()
def main(build:str='validated_square_f16'):
    sys.path.insert(0,str(HERE.parent))
    from modal_run import build_case
    metadata=json.loads((HERE.parent/'build'/build/'experiment.json').read_text())
    g=metadata.get('geometry',dict(m=4096,n=4096,k=4096,bm=128,bn=128,bk=64,wm=64,wn=64))
    shared=2*g.get('stages',2)*g['bk']*(g['bm']+g['bn'])
    config=dict(build=build,m=g['m'],n=g['n'],k=g['k'],dtype='f16',tile_m=g['bm'],tile_n=g['bn'],
                threads=g['bm']//g['wm']*(g['bn']//g['wn'])*32,smem=shared if shared>49152 else 0)
    case=build_case(config)
    result=profile.remote(case)
    out=HERE/'profiles'/build;out.mkdir(parents=True,exist_ok=True)
    for name in ('ours','ptxas'):
        record=result.get(name,{})
        if 'report_base64' in record:
            import base64
            (out/f'{name}.ncu-rep').write_bytes(base64.b64decode(record.pop('report_base64')))
        (out/f'{name}.csv').write_text(record.get('raw_csv',record.get('stdout','')))
    (out/'profile.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({name:{k:v for k,v in result.get(name,{}).items() if k in ('rc','stderr','error','import_rc')}
                      for name in ('ours','ptxas')},indent=2))
