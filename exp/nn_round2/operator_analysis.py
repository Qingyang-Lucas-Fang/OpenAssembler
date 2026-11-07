"""Mechanical arithmetic-campaign evidence summary, with attribution limits."""
from __future__ import annotations
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys


def analyze(path):
    raw=json.loads(path.read_text());result=raw.get('result',raw);cases=result['cases']
    ok=[c for c in cases if c['status']=='ok'];failed=[c for c in cases if c['status']!='ok']
    best=defaultdict(list);mode=defaultdict(list);geometry=defaultdict(dict)
    for c in ok:
        mode[c['op'],c['mode']].append(c['ptxas_over_oa'])
        best[c['op'],tuple(c['shape'])].append(c)
        geometry[c['op'],tuple(c['shape']),c['threads']][c['mode']]=c
    best_rows=[]
    for (op,shape),rows in best.items():
        selected={m:min(rows,key=lambda c:c['graphs'][m]['median_ms']) for m in ('oa','ptxas')}
        best_rows.append({'op':op,'shape':list(shape),'best_observed':{m:{'kernel':c['kernel'],'mode':c['mode'],'threads':c['threads'],'median_us':1000*c['graphs'][m]['median_ms']} for m,c in selected.items()},'best_observed_ptxas_over_oa':selected['ptxas']['graphs']['ptxas']['median_ms']/selected['oa']['graphs']['oa']['median_ms'],'scope':'within this tested source/layout/launch set, not exhaustive autotuning; original cross-layout RNG seeds differ'})
    hypotheses=[]
    for (op,shape,threads),rows in geometry.items():
        if 'scalar' not in rows:continue
        base=rows['scalar']
        for variant in ('scalar4','vector4'):
            if variant not in rows:continue
            c=rows[variant]
            hypotheses.append({'op':op,'shape':list(shape),'threads':threads,'variant':variant,'oa_scalar_over_variant':base['graphs']['oa']['median_ms']/c['graphs']['oa']['median_ms'],'ptxas_scalar_over_variant':base['graphs']['ptxas']['median_ms']/c['graphs']['ptxas']['median_ms'],'scope':'exploratory: random inputs differ across generated kernels; changes ownership, ILP, memory width and scheduler decisions together'})
    summary={'schema':1,'generated_utc':datetime.now(timezone.utc).isoformat(),'source_file':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'runtime':result['runtime'],'total_cases':len(cases),'statuses':dict(Counter(c['status'] for c in cases)),'operator_count':len(set(c['op'] for c in cases)),'correctness_checked_output_elements':sum(o['elements_checked'] for c in cases for p in c.get('patterns',[]) for m in p['checks'].values() for o in m['outputs']),'failed_cases':[{'kernel':c['kernel'],'shape':c['shape'],'threads':c['threads'],'status':c['status'],'error':c.get('error'),'patterns':c.get('patterns')} for c in failed],'same_ptx_screening':{'above_1_03':sum(c['ptxas_over_oa']>1.03 for c in ok),'below_0_97':sum(c['ptxas_over_oa']<.97 for c in ok),'between':sum(.97<=c['ptxas_over_oa']<=1.03 for c in ok),'threshold_scope':'descriptive screening, no multiple-comparison significance or validated selection claim'},'top_same_ptx_rows':[{'kernel':c['kernel'],'shape':c['shape'],'threads':c['threads'],'ptxas_over_oa':c['ptxas_over_oa'],'median_us':{m:v['median_ms']*1000 for m,v in c['graphs'].items()},'paired_sample_ratios':[p/o for p,o in zip(c['graphs']['ptxas']['samples_ms'],c['graphs']['oa']['samples_ms'])],'resources':c['resources']} for c in sorted(ok,key=lambda c:c['ptxas_over_oa'],reverse=True)[:20]],'op_mode_ratios':[{'op':op,'mode':m,'cases':len(values),'min':min(values),'median':statistics.median(values),'max':max(values)} for (op,m),values in sorted(mode.items())],'best_observed_layouts':best_rows,'layout_hypotheses':hypotheses,'limitations':['The matched compiler comparison is native OA versus ptxas -O3 on identical PTX; it is not an exhaustive cuBLAS/cuDNN/compiled-framework benchmark.','The initial variant inputs use seed+kernel_index: OA/ptxas within one row share exact inputs; cross-layout seed is not matched. Followup fixes this.','A same-PTX scalar4 win can disappear when both compilers use a better vectorized layout; best-observed table must accompany these wins.','Tanh-GELU backward fails the unchanged original analytic FP64 gate for both compilers in ten cases. No speed is admitted from failed rows.','All pointwise kernels use FP32 or unsigned32 arithmetic; no BF16/FP16 backward arithmetic admission follows.','Cross entropy fixes label zero, returns per-row loss and unnormalized gradient; dropout consumes a provided mask; norms have no learned affine terms.','Graph timing excludes launch/submission overhead and keeps data resident; graph/event resolution matters near2–3us.','There is no complete-transformer or tokens/sec claim in this operator file.']}
    return summary


if __name__=='__main__':
    path=Path(sys.argv[1]);out=Path(sys.argv[2]) if len(sys.argv)>2 else path.parent.parent/'operator_analysis.json'
    summary=analyze(path);out.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ('total_cases','statuses','operator_count','same_ptx_screening')},indent=2));print(out)
