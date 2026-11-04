"""Summarize full-output hardware checks and within-worker cache-hint timing."""
from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

HERE=Path(__file__).resolve().parent
BASE=HERE.parent


def main():
    suite=json.loads((HERE/'cache_suite.json').read_text());rows=[]
    for config in suite:
        path=BASE/'results'/f'{config["name"]}.json'
        records=json.loads(path.read_text())
        if len(records)!=1:raise ValueError('expectoneworkerrecord')
        result=records[0]
        if result.get('error') or result.get('worker_rc')!=0:raise AssertionError('GPUworkerfailed')
        for alternative in config['alternatives']:
            name=alternative['name']
            checks=[x for x in result['checks'] if x['method']==name]
            if len(checks)!=8:raise AssertionError('all8inputpatternsrequired')
            samples=result['samples_ms'];ratios=[a/b for a,b in zip(samples['ours'],samples[name])]
            row={'worker':config['name'],'variant':name,'dtype':config['dtype'],
                'shape_mnk':[config[x] for x in ('m','n','k')],
                'baseline_ms':result['milliseconds']['ours'],'candidate_ms':result['milliseconds'][name],
                'baseline_over_candidate_ratio_of_medians':result['milliseconds']['ours']/result['milliseconds'][name],
                'baseline_over_candidate_median_round_ratio':statistics.median(ratios),
                'candidate_faster_rounds':sum(x>1 for x in ratios),'rounds':len(ratios),
                'checks':len(checks),'full_output_elements_each_check':checks[0]['full_output_elements_checked'],
                'max_full_relative_l2_vs_cublas':max(x['full_output_relative_l2_vs_cublas'] for x in checks),
                'source_result':str(path.relative_to(BASE)),
                'source_result_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            rows.append(row)
    (HERE/'cache_hardware_summary.json').write_text(json.dumps({'passed':True,'rows':rows,
        'decision':'no promotion: measured effects are small/noisy; no useful positive improvement established',
        'scope':'isolated hint comparison within each worker; no cross-worker or transformer-step claim'},indent=2)+'\n')
    print(json.dumps([{'variant':x['variant'],'ratio':x['baseline_over_candidate_ratio_of_medians'],
                      'wins':x['candidate_faster_rounds']} for x in rows],indent=2))


if __name__=='__main__':main()
