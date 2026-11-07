"""Summarize stable tanh-GELU derivative evidence without admitting failed rows."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys


def analyze(path):
    raw=json.loads(path.read_text());result=raw.get('result',raw);groups=defaultdict(list)
    for c in result['cases']:groups[c['candidate']].append(c)
    return {'schema':1,'generated_utc':datetime.now(timezone.utc).isoformat(),'source_file':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'runtime':result['runtime'],'total_cases':len(result['cases']),'by_candidate':{name:{'statuses':dict(Counter(c['status'] for c in cs)),'max_oa_absolute_error_all_patterns':max(o['max_abs'] for c in cs for p in c.get('patterns',[]) for o in p['checks']['oa']['outputs']),'full_checks':sum(o['elements_checked'] for c in cs for p in c.get('patterns',[]) for m in p['checks'].values() for o in m['outputs'])} for name,cs in groups.items()},'diagnostics':result['diagnostics'],'admitted_large_timings':[{'candidate':c['candidate'],'round':c['round'],'mode':c['mode'],'threads':c['threads'],'shape':c['shape'],'median_us':{m:v['median_ms']*1000 for m,v in c['graphs'].items()},'ptxas_over_oa':c['ptxas_over_oa']} for c in result['cases'] if c['status']=='ok' and c['shape']==[1048576]],'interpretation':['The analytic tanh-GELU derivative and original8e-5/2e-5 gate remain unchanged.','Direct MUFU tanh-derived backward has rare errors amplified near negative saturation; both compilers fail original gates.','The exponential sigmoid/sech-squared formulation passes all measured candidate cases and a bounded stress grid; this is a numerical repair, not a proof of general IEEE semantics.','The stable vector4 OA candidate remains slower than same-source ptxas; failed old-form cases are excluded from timing.','Candidate is isolated in exp and is not a production backward operator or automatic model rewrite.']}


if __name__=='__main__':
    path=Path(sys.argv[1]);out=Path(sys.argv[2]) if len(sys.argv)>2 else path.parent.parent/'operator_gelu_analysis.json'
    value=analyze(path);out.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps({'total_cases':value['total_cases'],'by_candidate':value['by_candidate']},indent=2));print(out)
