"""Audit complete-output fragment-window GPU checks and retained timing."""
from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

HERE=Path(__file__).resolve().parent
BASE=HERE.parent


def main():
    suite=json.loads((HERE/'fragment_suite.json').read_text());rows=[]
    for config in suite:
        path=BASE/'results'/f'{config["name"]}.json'
        saved=json.loads(path.read_text())
        if len(saved)!=1:raise AssertionError('oneworkerrequired')
        record=saved[0]
        if record.get('error') or record.get('worker_rc')!=0:raise AssertionError('workerfailed')
        for variant in config['alternatives']:
            name=variant['name'];checks=[x for x in record['checks'] if x['method']==name]
            if len(checks)!=8:raise AssertionError('expect8inputchecks')
            if not all(x['full_output_elements_checked']==4096**2 and x['full_output_relative_l2_vs_cublas']<=2e-5 for x in checks):
                raise AssertionError('fulloutputfailed')
            if not all(x['exact'] for x in checks if x['pattern'] in ('integer','zero')):
                raise AssertionError('exactcasefailed')
            samples=record['samples_ms'];ratios=[x/y for x,y in zip(samples['ours'],samples[name])]
            rows.append({'variant':name,'dtype':config['dtype'],
                'baseline_ms':record['milliseconds']['ours'],'candidate_ms':record['milliseconds'][name],
                'baseline_over_candidate_ratio_of_medians':record['milliseconds']['ours']/record['milliseconds'][name],
                'baseline_over_candidate_median_paired_ratio':statistics.median(ratios),
                'candidate_faster_rounds':sum(x>1 for x in ratios),'timing_rounds':len(ratios),
                'checks':len(checks),'full_output_elements_each':4096**2,
                'maximum_full_relative_l2_vs_cublas':max(x['full_output_relative_l2_vs_cublas'] for x in checks),
                'result_source':str(path.relative_to(BASE)),
                'result_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    result={'passed_correctness':True,'promote':False,'decision':'all six bounded-B windows lose to the stage-pair baseline under the validated synchronization model',
            'causal_limit':'The experiment changes fragment lifetimes, issue bursts, and tensor epoch boundaries together; counters and isolated drain experiments are required to assign causes.',
            'rows':rows}
    (HERE/'fragment_hardware_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
