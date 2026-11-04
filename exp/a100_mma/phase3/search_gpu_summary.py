"""Summarize retained search GPU evidence without invoking remote execution."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
BASE=HERE.parent


def summarize():
    summary=[]
    for scale in ('smoke','large'):
        for dtype in ('f16','bf16'):
            path=BASE/'results'/f'search_{scale}_{dtype}.json'
            if not path.exists():continue
            result=json.loads(path.read_text())[0]
            baseline=result['milliseconds']['ours']
            checks=result['checks']
            summary.append(dict(scale=scale,dtype=dtype,shape=result['shape_mnk'],
                 recorded_utc=result['recorded_utc'],device=result['device'],worker_rc=result['worker_rc'],
                 source=str(path.relative_to(BASE)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 check_rows=len(checks),full_output_elements=min(c['full_output_elements_checked'] for c in checks),
                 zero_integer_all_exact=all(c['exact'] for c in checks if c['pattern'] in ('zero','integer')),
                 maximum_full_output_relative_l2=max(c['full_output_relative_l2_vs_cublas'] for c in checks),
                 milliseconds=result['milliseconds'],samples_ms=result['samples_ms'],
                 latency_reduction_percent={name:100*(baseline-ms)/baseline for name,ms in result['milliseconds'].items()}))
    (HERE/'search_gpu_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


if __name__=='__main__':
    rows=summarize()
    for row in rows:
        print(row['scale'],row['dtype'],row['worker_rc'],row['check_rows'],
              row['full_output_elements'],row['maximum_full_output_relative_l2'])
