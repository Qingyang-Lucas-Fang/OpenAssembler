"""Audit saved Lt runs without rerunning GPUs or altering original evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

HERE=Path(__file__).resolve().parent


def paired(samples,numerator,denominator):
    ratios=[x/y for x,y in zip(samples[numerator],samples[denominator])]
    return {'ratio_of_medians':statistics.median(samples[numerator])/statistics.median(samples[denominator]),
            'median_of_round_ratios':statistics.median(ratios),'round_ratios':ratios,
            'min_round_ratio':min(ratios),'max_round_ratio':max(ratios),
            'rounds_numerator_slower':sum(x>1 for x in ratios),'rounds':len(ratios)}


def audit(record):
    times=record['milliseconds'];samples=record['samples_ms'];candidates=record['candidates']
    checks=record['checks'];best=record['best_lt']
    native=[x for x in times if x not in ('gemmex',) and not x.startswith('lt_')]
    structural={
        'worker_success':record['worker_rc']==0 and record['passed'],
        'empty_worker_stderr':not record.get('worker_stderr'),
        'all_checks_pass':all(x['passed'] for x in checks),
        'guards_and_finite':all(x['guard_regions_passed'] and x['finite'] for x in checks),
        'zero_integer_exact':all(x['exact'] for x in checks if x['pattern'] in ('zero','integer')),
        'all_timed_methods_four_checks':all(sum(x['name']==name for x in checks)==4 for name in times),
        'all_timed_methods_seven_samples':all(len(v)==7 for v in samples.values()),
        'all_round_orders_cover_timed_methods':all(set(order)==set(times) and len(order)==len(times) for order in record['paired_round_orders']),
        'no_graph_rejections':not record.get('graph_rejections'),
        'all_attribute_reads_succeeded':all(v['status']==0 and v['bytes_written']==4 for c in candidates for v in c['config'].values() if isinstance(v,dict)),
        'heuristic_abi_validated':record['header_validation']['ctypes_algo_bytes']==64 and record['header_validation']['ctypes_heuristic_bytes']==96,
    }
    per_cap={}
    for cap in (0,32*1024*1024):
        returned={c['name'] for c in candidates if c['workspace_cap_bytes']==cap and c['name'] in times}
        feasible={c['name'] for c in candidates if c['name'] in times and not c.get('rejected')
                  and max(c['workspace_required_bytes'],c['algo_check']['workspace_required_bytes'])<=cap}
        selected=min(returned,key=times.get,default=None)
        best_known=min(feasible,key=times.get,default=None)
        per_cap[str(cap)]={'best_returned':selected,'best_returned_ms':times.get(selected),
            'best_known_feasible_from_both_queries':best_known,'best_known_feasible_ms':times.get(best_known),
            'returned_names':sorted(returned),'known_feasible_names':sorted(feasible)}
    winner=next(c for c in candidates if c['name']==best)
    return {'dtype':record['dtype'],'structural_checks':structural,'valid':all(structural.values()),
        'cublas_version':record['cublas_version'],'query_results':record['heuristic_queries'],
        'unique_algorithms':record['unique_heuristics'],'full_output_check_records':len(checks),
        'full_output_element_comparisons':sum(x['full_output_elements_checked'] for x in checks),
        'max_full_relative_l2':max(x['full_relative_l2_vs_gemmex'] for x in checks),
        'max_cpu_float64_sample_relative_l2':max(x['sample_relative_l2'] for x in checks),
        'best':{'name':best,'milliseconds':times[best],'config':winner['config'],
                'workspace_required_bytes':winner['workspace_required_bytes']},
        'workspace_queries_vs_feasibility':per_cap,
        'gemmex_over_best':paired(samples,'gemmex',best),
        'native_over_best':{name:paired(samples,name,best) for name in native},
        'native_over_gemmex':{name:paired(samples,name,'gemmex') for name in native},
        'confidence_limit':'Seven shuffled rounds are used both to select and assess the minimum across candidates. Ratios are descriptive; no independent confirmation or statistical confidence interval is claimed.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('source',nargs='?',default='lt_results_20261103T235453.187530Z.json')
    args=parser.parse_args();source=HERE/args.source
    rows=[audit(row) for row in json.loads(source.read_text())]
    result={'source':source.name,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'valid':all(row['valid'] for row in rows),'dtypes':rows}
    (HERE/'lt_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'valid':result['valid'],'dtypes':[(x['dtype'],x['unique_algorithms'],x['full_output_check_records']) for x in rows]}))
    if not result['valid']:raise SystemExit(1)
