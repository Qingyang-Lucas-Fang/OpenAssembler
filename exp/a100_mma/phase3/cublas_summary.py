"""Derive launch/resource facts from retained cuBLAS reports; no GPU calls."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def target_record(stdout):
    records=[json.loads(line) for line in stdout.splitlines() if line.startswith('{')]
    if len(records)!=1:raise ValueError('expected exactly one target contract record')
    return records[0]


def triplet(text):
    values=[int(x) for x in re.findall(r'\d+',text)]
    if len(values)!=3:raise ValueError(f'expected CUDA launch triplet: {text}')
    return values


def category(name):
    if 's16816gemm' in name:return 'gemm'
    if 'distribution_elementwise' in name:return 'random_input'
    if 'NormTwoOps' in name:return 'output_norm'
    return 'finite_output_diagnostic'


def dispatch_records(case):
    rows=csv.DictReader([line for line in case['stdout'].splitlines() if line.startswith('"')])
    launches={}
    for row in rows:
        idx=int(row['ID'])
        record=launches.setdefault(idx,dict(id=idx,name=row['Kernel Name'],
                  category=category(row['Kernel Name']),grid=triplet(row['Grid Size']),
                  block=triplet(row['Block Size']),metrics={}))
        metric=row['Metric Name']
        if metric:record['metrics'][metric]=dict(value=row['Metric Value'],unit=row['Metric Unit'])
    ordered=[launches[key] for key in sorted(launches)]
    gemm=[record for record in ordered if record['category']=='gemm']
    apis=len(re.findall(r'function cublasStatus_t cublasGemmEx\(',case['stderr']))
    fields={}
    for name in ('transa','transb','Atype','Btype','Ctype','computeType','algo'):
        values=set(re.findall(r'\b'+name+r': type=[^;]+; val=([^\n]+)',case['stderr']))
        if len(values)!=1:raise ValueError(f'expected one consistent API field: {name}')
        fields[name]=values.pop()
    assert apis==6 and len(gemm)==6
    assert [r['id'] for r in gemm]==list(range(2,8))
    assert len(ordered)==14 and [r['category'] for r in ordered[:2]]==['random_input']*2
    assert len({r['name'] for r in gemm})==1
    return dict(dtype=case['dtype'],rc=case['rc'],contract=target_record(case['stdout']),
                cublas_api_fields=fields,cublas_api_calls=apis,launch_count=len(ordered),
                gemm_launch_ids=[r['id'] for r in gemm],
                separate_gemm_auxiliary_kernel_observed=False,launches=ordered)


def summarize():
    dispatch_path=HERE/'cublas-dispatch.json';observation_path=HERE/'cublas-observation.json'
    dispatch=json.loads(dispatch_path.read_text());observation=json.loads(observation_path.read_text())
    result=dict(inputs={p.name:sha(p) for p in (dispatch_path,observation_path,HERE/'cublas_target.py')},
                evidence_scope='derived reports and counters; no vendor kernel copied or imported as OA code',
                public_swizzle_reference=dict(
                    url='https://github.com/NVIDIA/cutlass/blob/418ba45013423731194f8c487348a5da2511298a/include/cutlass/gemm/threadblock/threadblock_swizzle.h',
                    source_sha256='a8c1731909ecb877be8933bedfd645c5ebb792bd5d45d542d1aa04d7692ff4fe',
                    interpretation='public CUTLASS identity swizzle maps logical K to grid.z; cuBLAS private specialization is unproven'),
                tool_versions=observation['tools'],library_paths_observed=observation['library_paths'],cases=[])
    by_dtype={c['dtype']:dispatch_records(c) for c in dispatch['dispatch']}
    for measured in observation['cases']:
        dtype=measured['configuration']['dtype'];record=by_dtype[dtype]
        contract=target_record(measured['stdout']);assert contract==record['contract']
        launch=measured['launch']
        grid=[int(launch[f'launch__grid_dim_{axis}']) for axis in 'xyz']
        block=[int(launch[f'launch__block_dim_{axis}']) for axis in 'xyz']
        for entry in record['launches'][2:8]:assert entry['grid']==grid and entry['block']==block
        tile=(128,128,32,5) if dtype=='f16' else (128,256,64,3)
        m,n,k=contract['shape_mnk'];tm,tn,tk,stages=tile
        nominal_tiles=((m+tm-1)//tm)*((n+tn-1)//tn)
        shared=2*stages*tk*(tm+tn)
        dynamic=int(float(launch['launch__shared_mem_per_block_dynamic'])*1000)
        static=int(float(launch['launch__shared_mem_per_block_static'])*1000)
        assert shared==dynamic+static
        record['profile']=dict(kernel_name=measured['kernel_name'],grid=grid,block=block,
                    grid_product=grid[0]*grid[1]*grid[2],nominal_spatial_tiles=nominal_tiles,
                    grid_xy_product=grid[0]*grid[1],
                    named_tile_and_stages=dict(m=tm,n=tn,k=tk,stages=stages),
                    registers_per_thread=int(launch['launch__registers_per_thread']),
                    allocated_registers_per_thread=float(launch['launch__registers_per_thread_allocated']),
                    shared_operand_bytes=shared,dynamic_shared_bytes=dynamic,static_shared_bytes=static,
                    driver_shared_bytes=int(float(launch['launch__shared_mem_per_block_driver'])*1000),
                    resident_ctas_register_limit=float(launch['launch__occupancy_limit_registers']),
                    resident_ctas_shared_limit=float(launch['launch__occupancy_limit_shared_mem']),
                    metrics=measured['metrics'],static_opcode_counts=measured['static_opcode_counts'],
                    static_source_sha256=measured['source_sha256'],
                    reuse_annotation_count=measured['reuse_annotation_count'])
        record['split_k_assessment']=(dict(observed_grid_z=2,nominal_cta_ratio=2,
              conclusion='strong evidence consistent with two K partitions; not direct cuBLAS algorithm-attribute proof',
              separate_reduction_kernel=False,
              reduction_mechanism='unknown; trace cannot distinguish semaphore/serial/atomic/internal reduction')
              if dtype=='f16' else dict(observed_grid_z=1,nominal_cta_ratio=1,
              conclusion='no launch-level evidence for K partitioning in this call'))
        result['cases'].append(record)
    (HERE/'cublas-derived-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    for case in summarize()['cases']:
        p=case['profile']
        print(case['dtype'],case['contract']['cublas_version'],case['launch_count'],
              p['grid'],p['block'],p['shared_operand_bytes'],case['split_k_assessment']['conclusion'])
