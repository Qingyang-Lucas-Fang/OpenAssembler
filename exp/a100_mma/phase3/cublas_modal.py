"""Observe actual cuBLAS dispatch, resources, counters and resident SASS.

Vendor disassembly stays in the ephemeral GPU container. We retain derived
statistics, dispatch/resource reports and local analysis, never copied vendor
kernel modules as OpenAssembler implementations.
"""
import json,sys
from pathlib import Path
import modal
HERE=Path(__file__).resolve().parent
app=modal.App('openassembler-a100-cublas-observation')
image=modal.Image.from_registry('pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel')
image=image.add_local_file(HERE/'cublas_target.py','/root/exp/cublas_target.py')
METRICS=['gpu__time_duration.sum','sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active',
 'sm__warps_active.avg.pct_of_peak_sustained_active','smsp__inst_executed.sum',
 'smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct',
 'smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct',
 'smsp__warp_issue_stalled_wait_per_warp_active.pct',
 'smsp__warp_issue_stalled_barrier_per_warp_active.pct',
 'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum','dram__bytes.sum','lts__t_sector_hit_rate.pct']

@app.function(image=image,gpu='A100-40GB',timeout=480,max_containers=1,retries=0)
def observe(cases):
    import subprocess,os,re,csv,hashlib,collections,glob
    root=Path('/root/exp');root.mkdir(exist_ok=True)
    out={'cases':[],'tools':{}}
    for tool in ('ncu','cuobjdump','nvdisasm'):
        pr=subprocess.run([tool,'--version'],capture_output=True,text=True,timeout=20)
        out['tools'][tool]=pr.stdout
    helptext=subprocess.run(['ncu','--help'],capture_output=True,text=True,timeout=20).stdout
    out['source_options']=[line for line in helptext.splitlines() if any(x in line for x in ('print-source','page','kernel-name-base'))]
    for idx,case in enumerate(cases):
        report=root/f'cublas{idx}'
        argv=['ncu','--target-processes','all','--launch-skip','3','--launch-count','1',
              '--kernel-name-base','mangled','--metrics',','.join(METRICS),
              '--clock-control','none','--cache-control','all','--export',str(report),'--force-overwrite',
              sys.executable,str(root/'cublas_target.py'),'--m',str(case['m']),'--n',str(case['n']),
              '--k',str(case['k']),'--dtype',case['dtype']]
        proc=subprocess.run(argv,capture_output=True,text=True,timeout=150,
                            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUBLAS_LOGINFO_DBG='1',CUBLAS_LOGDEST_DBG='stderr'))
        record={'configuration':case,'argv':argv,'rc':proc.returncode,'stdout':proc.stdout,'stderr':proc.stderr}
        path=report.with_suffix('.ncu-rep')
        if path.exists():
            raw=subprocess.run(['ncu','--import',str(path),'--page','raw','--csv','--kernel-name-base','mangled'],
                               capture_output=True,text=True,timeout=30)
            record['raw_csv']=raw.stdout
            lines=raw.stdout.splitlines();rows=list(csv.reader(lines))
            if len(rows)>=3:
                data=dict(zip(rows[0],rows[-1]));units=dict(zip(rows[0],rows[1]))
                record['kernel_name']=data.get('Kernel Name')
                record['metrics']={k:{'value':data[k],'unit':units[k]} for k in METRICS if k in data}
                record['launch']={k:v for k,v in data.items() if k.startswith('launch__')}
            source=subprocess.run(['ncu','--import',str(path),'--page','source','--print-source','sass',
                                   '--kernel-name-base','mangled'],capture_output=True,text=True,timeout=40)
            (root/f'cublas{idx}.sass.txt').write_text(source.stdout)
            record['source_rc']=source.returncode;record['source_stderr']=source.stderr
            record['source_length']=len(source.stdout);record['source_sha256']=hashlib.sha256(source.stdout.encode()).hexdigest()
            # Derived instruction mix and controls, with no retained vendor code.
            counts=collections.Counter()
            for line in source.stdout.splitlines():
                match=re.search(r'\b((?:HMMA|IMMA|LDSM|LDGSTS|LDGDEPBAR|DEPBAR|BAR|STG|STS|LDS|LDG|MOV|IMAD|IADD3|LEA|SHF|LOP3|F2F|FADD|FFMA|BRA|NOP|UIADD3|UIMAD|ULOP3|S2R|CS2R|ISETP|EXIT)(?:\.[A-Za-z0-9_]+)*)\s',line)
                if match:counts[match[1]]+=1
            record['static_opcode_counts']=dict(counts)
            record['reuse_annotation_count']=source.stdout.count('.reuse')
            record['source_preview']=source.stdout[:1500] if not counts else None
        out['cases'].append(record)
    libraries=glob.glob('/opt/conda/lib/python*/site-packages/nvidia/cublas/lib/libcublas*.so*')+glob.glob('/usr/local/cuda/lib64/libcublas*.so*')
    out['library_paths']=libraries
    return out

@app.local_entrypoint()
def main():
    cases=[dict(m=4096,n=4096,k=4096,dtype='f16'),dict(m=4096,n=4096,k=4096,dtype='bf16')]
    result=observe.remote(cases)
    (HERE/'cublas-observation.json').write_text(json.dumps(result,indent=2)+'\n')
    for record in result['cases']:
        print(json.dumps({k:record.get(k) for k in ('configuration','rc','kernel_name','metrics','launch','source_rc','source_length','static_opcode_counts','source_preview')},indent=2))
