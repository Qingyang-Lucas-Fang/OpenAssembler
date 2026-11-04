"""Recompute fixed-schedule comparisons from retained A100 result records."""
from pathlib import Path
import hashlib
import json
import statistics

HERE=Path(__file__).resolve().parent
BASE=HERE.parent

def record(name):
    path=BASE/'results'/f'{name}.json'
    if not path.exists():return None
    raw=path.read_bytes();data=json.loads(raw)[0]
    if data.get('error') or data.get('worker_rc')!=0:
        raise AssertionError(f'failed experiment: {name}: {data.get("error")}')
    checks=data['checks'];m,n,k=data['shape_mnk']
    if not checks or any(c['full_output_elements_checked']!=m*n for c in checks):
        raise AssertionError(f'incomplete matrix oracle: {name}')
    return data,dict(file=str(path.relative_to(BASE)),sha256=hashlib.sha256(raw).hexdigest(),
                     recorded_utc=data['recorded_utc'],check_rows=len(checks),
                     elements_per_check=m*n,device=data['device'])

def comparison(name,candidate):
    pair=record(name)
    if pair is None:return None
    data,source=pair;t=data['milliseconds']
    ratios=[b/c for b,c in zip(data['samples_ms']['ours'],data['samples_ms'][candidate])]
    return dict(source=source,shape_mnk=data['shape_mnk'],dtype=data['dtype'],candidate=candidate,
                baseline_ms=t['ours'],candidate_ms=t[candidate],cublas_gemmex_ms=t['cublas'],
                latency_reduction_pct=100*(1-t[candidate]/t['ours']),
                throughput_relative_to_cublas=t['cublas']/t[candidate],
                candidate_tflops=data['tflops'][candidate],
                within_round_speedup_median=statistics.median(ratios),
                within_round_speedup_min=min(ratios),within_round_speedup_max=max(ratios),
                registers=data['resources'][candidate]['num_regs'])

def main():
    out={'contract':'RCR full-tile dense GEMM; FP16/BF16 inputs, FP32 accumulator/output',
         'timing':'median7 shuffled rounds of20-node CUDA graphs; clocks unlocked',
         'vendor_baseline':'matched cublasGemmEx compute32F, tensor-op default; cuBLASLt separate',
         'combined':[],'confirmation':[],'stage_confirmation':[],'final':[],'address':[]}
    for dtype in ('f16','bf16'):
        for shape in ('square','mlp','tall'):
            candidate=f'search_combined_{shape}_row_bpair_{dtype}'
            for prefix,key in [('search_combined','combined'),('p3_confirm','confirmation')]:
                item=comparison(f'{prefix}_{shape}_{dtype}',candidate)
                if item:out[key].append(item)
            stage=(f'address_stagepair_square_{dtype}' if shape=='square' else
                   f'address_stagepair_confirm_{shape}_{dtype}')
            for prefix,key in [('address_stage_confirm','stage_confirmation'),('p3_final','final')]:
                item=comparison(f'{prefix}_{shape}_{dtype}',stage)
                if item:out[key].append(item)
    for name in ('address_shared_square_f16_paired','address_shared_square_bf16_paired',
                 'address_stage_square_f16_paired','address_stage_square_bf16_paired'):
        # These suites compare to the combined kernel, not the older scalar.
        pair=record(name)
        if pair is None:continue
        data,source=pair
        out['address'].append(dict(source=source,milliseconds=data['milliseconds']))
    (HERE/'evidence-summary.json').write_text(json.dumps(out,indent=2)+'\n')
    for key in ('combined','confirmation','stage_confirmation','final'):
        print(key)
        for item in out[key]:
            print(item['dtype'],item['shape_mnk'],
                  f"{item['baseline_ms']:.6f} -> {item['candidate_ms']:.6f} ms; "
                  f"{item['latency_reduction_pct']:.2f}% less time; "
                  f"{100*item['throughput_relative_to_cublas']:.2f}% cuBLAS throughput")

if __name__=='__main__':main()
