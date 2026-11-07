"""Analyze local-FMA candidates without claiming a general compiler win."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys


def analyze(path):
    raw=json.loads(path.read_text());result=raw.get('result',raw);cases=result['cases'];groups=defaultdict(dict)
    for c in cases:
        if c['status']=='ok':groups[c['round'],c['op'],tuple(c['shape']),c['threads']][c['candidate']]=c
    pairs=[];consistent=defaultdict(list)
    for (ri,op,shape,threads),rows in groups.items():
        if set(rows)!= {'baseline','fma'}:continue
        assert rows['baseline']['input_seed']==rows['fma']['input_seed']
        ratios={m:rows['baseline']['graphs'][m]['median_ms']/rows['fma']['graphs'][m]['median_ms'] for m in ('oa','ptxas')}
        row={'round':ri,'op':op,'shape':list(shape),'threads':threads,'input_seed':rows['fma']['input_seed'],'baseline_over_fma':ratios,'median_us':{candidate:{method:c['graphs'][method]['median_ms']*1000 for method in ('oa','ptxas')} for candidate,c in rows.items()},'resources':{candidate:c['resources'] for candidate,c in rows.items()},'fma_ptxas_over_oa':rows['fma']['ptxas_over_oa']}
        pairs.append(row);consistent[op,shape,threads].append(row)
    settings=[]
    for (op,shape,threads),rows in consistent.items():
        ratios=[c['baseline_over_fma']['oa'] for c in rows]
        settings.append({'op':op,'shape':list(shape),'threads':threads,'rounds':len(rows),'min_oa_baseline_over_fma':min(ratios),'max_oa_baseline_over_fma':max(ratios),'median_oa_baseline_over_fma':statistics.median(ratios),'decision':'selective_gain_candidate' if len(rows)>=2 and min(ratios)>1.03 else 'regression_or_neutral_do_not_globally_select','scope':'two reverse-order rounds in one GPU worker; requires independent-process confirmation and whole-workload admission'})
    byop=defaultdict(list)
    for p in pairs:byop[p['op']].append(p['baseline_over_fma']['oa'])
    return {'schema':1,'generated_utc':datetime.now(timezone.utc).isoformat(),'source_file':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'runtime':result['runtime'],'total_cases':len(cases),'statuses':dict(Counter(c['status'] for c in cases)),'paired_settings':len(pairs),'same_inputs_verified_by_seed':True,'precision':'original unchanged per-op FP64 gates; explicit FMA changes local rounding, ptxas previously may contract unconstrained operations','pairs':pairs,'settings':sorted(settings,key=lambda c:c['min_oa_baseline_over_fma'],reverse=True),'op_summary':[{'op':op,'pairs':len(v),'min':min(v),'median':statistics.median(v),'max':max(v)} for op,v in sorted(byop.items())],'interpretation':['Large width513 backward reductions benefit selectively, especially RMSNorm and softmax backward at4096rows.','Forward norms mostly neutral or regress; lower register/instruction counts alone do not justify selection.','Every FMA candidate remains behind matched ptxas in this measured set. This closes part of the OA gap but does not establish the user target.','No synchronization is removed, no hidden cross-thread dependencies are weakened, and no production source is edited.','Local accumulation fusion and reduced live products change both instruction choice and allocation; counters are needed to distinguish the hardware cause.','Comparisons are paired by setting across reverse-order blocks; within each kernel the OA/PTXAS event order is shuffled. Baseline/FMA themselves are not alternated within each individual sample.']}


if __name__=='__main__':
    path=Path(sys.argv[1]);out=Path(sys.argv[2]) if len(sys.argv)>2 else path.parent.parent/'operator_fma_analysis.json'
    value=analyze(path);out.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps({'cases':value['total_cases'],'statuses':value['statuses'],'selective_settings':[s for s in value['settings'] if s['decision']=='selective_gain_candidate']},indent=2));print(out)
