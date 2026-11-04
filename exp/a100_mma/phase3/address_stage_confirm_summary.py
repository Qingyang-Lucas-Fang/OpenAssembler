"""Summarize retained stage-pair confirmation results without GPU execution."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
BASE=HERE.parent


if __name__=='__main__':
    rows=[]
    for path in sorted((BASE/'results').glob('address_stage_confirm*.json')):
        result=json.loads(path.read_text())[0]
        times=result['milliseconds']
        candidates=[key for key in times if key.startswith('address_stagepair')]
        assert len(candidates)==1 and result['worker_rc']==0
        candidate=candidates[0]
        rows.append(dict(name=result['name'],dtype=result['dtype'],shape_mnk=result['shape_mnk'],
            source=str(path.relative_to(BASE)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            check_rows=len(result['checks']),
            full_output_elements=min(row['full_output_elements_checked'] for row in result['checks']),
            maximum_full_output_relative_l2=max(row['full_output_relative_l2_vs_cublas'] for row in result['checks']),
            row_bpair_ms=times['ours'],stage_pair_ms=times[candidate],cublas_ms=times['cublas'],
            additional_latency_reduction_percent=100*(1-times[candidate]/times['ours']),
            latency_gap_to_cublas_percent=100*(times[candidate]/times['cublas']-1),
            samples_ms=result['samples_ms']))
    assert len(rows)==6
    (HERE/'address-stage-confirm-summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    for row in rows:print(row['name'],row['additional_latency_reduction_percent'])
