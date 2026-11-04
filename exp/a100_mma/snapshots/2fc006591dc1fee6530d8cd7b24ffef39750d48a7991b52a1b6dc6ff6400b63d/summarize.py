"""Aggregate evidence without converting failed runs into performance claims."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
rows=[]
for path in sorted((HERE/'results').glob('*.json')):
    for record in json.loads(path.read_text()):
        row={'name':record.get('name',path.stem),'path':str(path.relative_to(HERE))}
        if 'error' in record:
            row.update(status='failed',error=record['error'].strip().splitlines()[-1])
        elif 'passed' in record:
            row.update(status='passed' if record['passed'] else 'failed',family=record.get('family'),
                       encoded_gap=record.get('encoded_gap'),mismatches=record.get('mismatches'))
        else:
            times=record['milliseconds'];checks=record['checks']
            row.update(status='passed',shape_mnk=record['shape_mnk'],dtype=record['dtype'],
                       output_dtype=record['output_dtype'],milliseconds=times,
                       ours_vs_cublas=times['cublas']/times['ours'],
                       ours_vs_ptxas=times['ptxas']/times['ours'],
                       registers=record['resources']['ours']['num_regs'],
                       shared_bytes=record['resources']['ours']['shared_bytes'],
                       full_output_checked=all('full_output_elements_checked' in c for c in checks),
                       max_reference_relative_l2=max(c['relative_l2'] for c in checks))
        rows.append(row)
(HERE/'evidence-summary.json').write_text(json.dumps(rows,indent=2)+'\n')
for row in rows:
    if row['status']=='passed' and row.get('full_output_checked'):
        print(row['name'],row['shape_mnk'],row['dtype'],
              {k:round(v,4) for k,v in row['milliseconds'].items() if k in ('ours','ptxas','cublas')},
              f"cuBLAS speed fraction {row['ours_vs_cublas']:.3f}")
