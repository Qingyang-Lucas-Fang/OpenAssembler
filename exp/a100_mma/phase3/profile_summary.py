"""Parse retained phase3 Nsight profiles into auditable numerical summaries."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def number(text):
    try:return float(text.replace(',',''))
    except (ValueError,AttributeError):return text


def summarize(case='search_combined_square_f16'):
    directory=HERE/'profiles'/case
    plan=json.loads((directory/'plan.json').read_text())
    result=dict(case=case,plan=plan,records=[],derived={})
    for variant in ('baseline','row_bpair'):
        path=directory/variant/'profile.json';record=json.loads(path.read_text())
        assert record['phase3_plan']['dtype']==plan['dtype']
        for method in ('ours','ptxas'):
            data=record[method]
            assert data['rc']==0 and data['import_rc']==0
            rows=list(csv.reader(line for line in data['raw_csv'].splitlines() if line.startswith('"')))
            assert len(rows)==3
            values=dict(zip(rows[0],rows[2]));units=dict(zip(rows[0],rows[1]))
            metrics={key:dict(value=number(values[key]),unit=units[key])
                     for key in record['selected_metrics']}
            launch={key:values[key] for key in values if key.startswith('launch__')}
            result['records'].append(dict(variant=variant,method=method,
                  source=str(path.relative_to(HERE)),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  input_provenance=record['provenance'],metrics=metrics,launch=launch,
                  unavailable_requested_metrics=record['unavailable_requested_metrics']))
    a,b=[r for r in result['records'] if r['method']=='ours']
    assert a['variant']=='baseline' and b['variant']=='row_bpair'
    result['derived']['ours_baseline_to_row_bpair']={key:dict(
          baseline=a['metrics'][key]['value'],row_bpair=b['metrics'][key]['value'],unit=a['metrics'][key]['unit'],
          difference=b['metrics'][key]['value']-a['metrics'][key]['value'],
          relative_change_percent=100*(b['metrics'][key]['value']/a['metrics'][key]['value']-1)
              if a['metrics'][key]['value'] else None)
          for key in a['metrics'] if isinstance(a['metrics'][key]['value'],(int,float))}
    tensor='smsp__inst_executed_pipe_tensor.sum'
    assert len({r['metrics'][tensor]['value'] for r in result['records']})==1
    output=directory/'derived-summary.json';output.write_text(json.dumps(result,indent=2)+'\n')
    return result,output


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case',default='search_combined_square_f16')
    args=parser.parse_args();result,output=summarize(args.case)
    for row in result['records']:
        print(row['variant'],row['method'],
              row['metrics']['sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active']['value'],
              row['metrics']['smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct']['value'])
    print(output)
