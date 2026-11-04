"""Derive stage/occupancy counter evidence from retained profiles, no GPU."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
BASE=HERE.parent


def hashed_json(path):
    raw=path.read_bytes()
    return json.loads(raw),dict(file=str(path.relative_to(BASE)),sha256=hashlib.sha256(raw).hexdigest())


def number(text):
    try:return float(text.replace(',',''))
    except (AttributeError,ValueError):return text


def parse(build):
    directory=HERE/'profiles'/build
    profile,source=hashed_json(directory/'profile.json')
    plan,plan_source=hashed_json(directory/'plan.json')
    result=[]
    for method in ('ours','ptxas'):
        record=profile[method]
        assert record['rc']==0 and record['import_rc']==0
        rows=list(csv.reader(line for line in record['raw_csv'].splitlines() if line.startswith('"')))
        assert len(rows)==3
        values=dict(zip(rows[0],rows[2]));units=dict(zip(rows[0],rows[1]))
        expected_grid=plan['grid'];expected_threads=plan['threads']
        assert [int(v) for v in re.findall(r'\d+',values['Grid Size'])]==expected_grid
        assert [int(v) for v in re.findall(r'\d+',values['Block Size'])]==[expected_threads,1,1]
        assert float(values['launch__shared_mem_per_block_dynamic'])*1000==plan['dynamic_shared_bytes']
        assert values['Kernel Name']=='gemm'
        metrics={key:dict(value=number(values[key]),unit=units[key]) for key in profile['selected_metrics']}
        duration=metrics['gpu__time_duration.sum']
        dram=metrics['dram__bytes.sum']
        normalized=dict(profile_duration_us=duration['value']*{'us':1,'ms':1000,'ns':.001}[duration['unit']],
                        dram_bytes=dram['value']*{'byte':1,'Kbyte':1000,'Mbyte':1000000,'Gbyte':1000000000}[dram['unit']])
        hardware={key:number(value) for key,value in values.items() if key.startswith('launch__')
                  or key in ('device__attribute_max_warps_per_multiprocessor',
                             'device__attribute_max_registers_per_multiprocessor',
                             'device__attribute_warp_size')}
        active_pct=metrics['sm__warps_active.avg.pct_of_peak_sustained_active']['value']
        normalized['active_warps_from_occupancy']=active_pct/100*hardware['device__attribute_max_warps_per_multiprocessor']
        result.append(dict(build=build,method=method,source=source,plan_source=plan_source,
                           contract={key:plan[key] for key in ('dtype','shape_mnk','grid','threads','dynamic_shared_bytes')},
                           input_provenance=profile['provenance'],metrics=metrics,hardware=hardware,
                           normalized=normalized,unavailable_requested_metrics=profile['unavailable_requested_metrics']))
    return result


def compare(before,after):
    assert before['contract']==after['contract']
    tensor='smsp__inst_executed_pipe_tensor.sum'
    assert before['metrics'][tensor]['value']==after['metrics'][tensor]['value']
    changes={}
    for key,a in before['metrics'].items():
        b=after['metrics'][key]
        if isinstance(a['value'],(int,float)) and a['unit']==b['unit']:
            changes[key]=dict(before=a['value'],after=b['value'],unit=a['unit'],
                         difference=b['value']-a['value'],
                         relative_change_percent=100*(b['value']/a['value']-1) if a['value'] else None)
    return dict(before=before['build'],after=after['build'],changes=changes)


def summarize():
    builds=['search_combined_tall_row_bpair_f16','address_stagepair_confirm_tall_f16',
            'address_occupancy_large_f16_w64x32_unfolded','address_occupancy_large_f16_w64x32_folded']
    records=[row for build in builds for row in parse(build)]
    ours={row['build']:row for row in records if row['method']=='ours'}
    result=dict(records=records,stage_tall_ours=compare(ours[builds[0]],ours[builds[1]]),
                occupancy_ours=compare(ours[builds[2]],ours[builds[3]]),occupancy_benchmarks=[])
    for dtype in ('f16','bf16'):
        rows,source=hashed_json(BASE/'results'/f'address_occupancy_large_{dtype}.json')
        row=rows[0];assert row['worker_rc']==0
        times=row['milliseconds'];baseline=times['ours']
        families=[]
        for shape in ('w64x32','w32x64'):
            unfolded=times[f'address_occupancy_large_{dtype}_{shape}_unfolded']
            folded=times[f'address_occupancy_large_{dtype}_{shape}_folded']
            families.append(dict(warp_shape=shape,unfolded_ms=unfolded,folded_ms=folded,
                  folding_latency_reduction_percent=100*(1-folded/unfolded),
                  folded_latency_gap_to_w64x64_percent=100*(folded/baseline-1)))
        result['occupancy_benchmarks'].append(dict(dtype=dtype,source=source,
                   check_rows=len(row['checks']),w64x64_row_bpair_ms=baseline,families=families))
    evidence,source=hashed_json(HERE/'evidence-summary.json')
    result['final_same_worker_evidence']=dict(source=source,rows=evidence['final'])
    output=HERE/'profile-followup-summary.json'
    output.write_text(json.dumps(result,indent=2)+'\n')
    return result,output


if __name__=='__main__':
    result,output=summarize()
    for row in result['records']:
        print(row['build'],row['method'],row['normalized'])
    print(output)
