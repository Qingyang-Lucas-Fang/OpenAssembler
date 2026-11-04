"""Compact, provenance-preserving summary for counter and mutation evidence."""
import csv,json,statistics
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
summary={'profiles':{},'clocked':{},'sync':[],'paired':{},'integration':[],'native_sparse':[]}
for folder in (HERE/'profiles').iterdir():
    result={}
    for name in ('ours','ptxas'):
        path=folder/f'{name}.csv'
        if not path.exists():continue
        rows=list(csv.reader(path.read_text().splitlines()))
        if len(rows)<3:continue
        vals=dict(zip(rows[0],rows[-1]));units=dict(zip(rows[0],rows[1]))
        metrics=json.loads((folder/'profile.json').read_text())['selected_metrics']
        result[name]={k:{'value':vals[k],'unit':units[k]} for k in metrics if k in vals}
    summary['profiles'][folder.name]=result
    (folder/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
clock=json.loads((HERE/'clocked-exact/results-clean.json').read_text())
for family in ('dependent','independent','ldsm'):
    records=[r for r in clock if r.get('family')==family]
    assert records and all(r.get('checksum_ok') for r in records)
    gaps=sorted(set(r['encoded_issue_gap'] for r in records))
    for gap in gaps:
        group=sorted([r for r in records if r['encoded_issue_gap']==gap],key=lambda r:r['steps'])
        slope=(group[-1]['median_cycles']-group[0]['median_cycles'])/(group[-1]['steps']-group[0]['steps'])
        summary['clocked'][f'{family}_gap{gap}']={'observed_cycle_slope':slope,
                'scope':'one warp/CTA, exact SASS; LDSM includes dependent LOP3 and IADD3, not pure load latency',
                'records':[{'n':r['steps'],'cycles':r['median_cycles']} for r in group]}
for path in sorted((BASE/'results').glob('p2_sync*json')):
    r=json.loads(path.read_text())[0]
    summary['sync'].append({'name':r['name'],'passed':'error'not in r,'error':r.get('error','').splitlines()[-1:],
                            'milliseconds':r.get('milliseconds')})
for name in ('p2_paired_windows','p2_banking_paired','p2_grouping_paired','p2_pointer_paired','p2_scheduling_paired'):
    r=json.loads((BASE/'results'/f'{name}.json').read_text())[0]
    assert 'error'not in r
    summary['paired'][name]={'milliseconds':r['milliseconds'],'resources':r['resources'],
                            'checks':len(r['checks']),'provenance':r['input_provenance']}
for path in sorted(list((BASE/'results').glob('p2_confirm_*json'))+list((BASE/'results').glob('p2_schedconfirm_*json'))):
    r=json.loads(path.read_text())[0]
    summary['paired'][r['name']]={'milliseconds':r.get('milliseconds'),'error':r.get('error'),
                                 'checks':len(r.get('checks',[])),'provenance':r.get('input_provenance')}
summary['integration']=json.loads((HERE/'integration-results.json').read_text())
summary['native_sparse']=json.loads((HERE/'sparse/results-validated.json').read_text())
summary['numerical']=json.loads((HERE/'numeric/results.json').read_text())
(HERE/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'clocked':summary['clocked'],'sync':[(r['name'],r['passed']) for r in summary['sync']],
                  'integration':[{k:r[k] for k in ('name','step_wall_ms')} for r in summary['integration']]},indent=2))
