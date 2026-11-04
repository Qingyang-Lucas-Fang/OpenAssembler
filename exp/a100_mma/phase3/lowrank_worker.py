"""GPU execution with full output, guards, matched numerical oracles, and pairing."""
from __future__ import annotations

import base64
import ctypes as C
import dataclasses
import random
import statistics
import traceback


def run(payload):
    import torch
    from lowrank_oracle import compare,direct_reference,inputs,reference
    from openassembler.gpu.cuda_driver import Driver
    torch.cuda.init()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    drv=Driver()
    m,n=payload['m'],payload['n']
    pad=256
    storage=torch.full((m*n+pad*2,),-12345.,device='cuda',dtype=torch.float32)
    output=storage[pad:-pad].view(m,n)
    a=torch.empty((m,32),device='cuda',dtype=torch.float16)
    b=torch.empty((n,32),device='cuda',dtype=torch.float16)
    pointers=[C.c_uint64(t.data_ptr()) for t in (a,b,output)]
    params=(C.c_void_p*3)(*[C.cast(C.pointer(p),C.c_void_p) for p in pointers])
    modules=[];launches={};resources={}
    for variant in payload['variants']:
        module=drv.load(base64.b64decode(variant['cubin']));modules.append(module)
        fn=drv.function(module,'gemm')
        resources[variant['name']]=drv.func_attrs(fn)
        def launch(fn=fn):
            stream=C.c_void_p(torch.cuda.current_stream().cuda_stream)
            drv.ck(drv.cu.cuLaunchKernel(fn,m//32,n//32,1,32,1,1,0,stream,params,None),'lowrank launch')
        launches[variant['name']]=launch
    records=[]
    for family in ('integer','floating','cancellation'):
        av,bv=inputs(m,n,family)
        a.copy_(torch.tensor(av,dtype=torch.float16).view(m,32))
        b.copy_(torch.tensor(bv,dtype=torch.float16).view(n,32))
        direct=direct_reference(av,bv,m,n)
        expected={s:reference(av,bv,m,n,s) for s in ('strassen','naive')}
        for variant in payload['variants']:
            output.fill_(float('nan'))
            launches[variant['name']]()
            torch.cuda.synchronize()
            actual=output.cpu().flatten().tolist()
            guard_ok=bool((storage[:pad]==-12345.).all() and (storage[-pad:]==-12345.).all())
            check=compare(actual,expected[variant['scheme']],family=='integer')
            records.append({'variant':variant['name'],'family':family,'matched_policy':check,
                'guard_regions_passed':guard_ok,
                'actual_vs_untransformed_direct_max_abs':max(abs(x-y) for x,y in zip(actual,direct)),
                'matched_oracle_vs_untransformed_direct_max_abs':max(abs(x-y) for x,y in zip(expected[variant['scheme']],direct))})
    passed=all(x['matched_policy']['passed'] and x['guard_regions_passed'] for x in records)
    timing={};ratios={}
    if passed:
        # Warm the same floating inputs before all kernels, capture identical
        # node counts, and shuffle graph order within every paired round.
        av,bv=inputs(m,n,'floating')
        a.copy_(torch.tensor(av,dtype=torch.float16).view(m,32))
        b.copy_(torch.tensor(bv,dtype=torch.float16).view(n,32))
        graph_nodes=32
        graphs={}
        for name,launch in launches.items():
            for _ in range(20):launch()
            torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(graph_nodes):launch()
            graphs[name]=graph
            for _ in range(10):graph.replay()
        torch.cuda.synchronize()
        samples={name:[] for name in launches}
        rng=random.Random(41)
        for _ in range(7):
            order=list(graphs);rng.shuffle(order)
            for name in order:
                begin,end=(torch.cuda.Event(enable_timing=True) for _ in range(2))
                begin.record();graphs[name].replay();end.record();end.synchronize()
                samples[name].append(begin.elapsed_time(end)/graph_nodes)
        timing={name:{'median_ms':statistics.median(values),'samples_ms':values}
                for name,values in samples.items()}
        for compiler in ('ours','ptxas'):
            ratios[f'{compiler}_naive_over_strassen']=timing[f'naive_{compiler}']['median_ms']/timing[f'strassen_{compiler}']['median_ms']
    return {'passed':passed,'device':dataclasses.asdict(drv.info()),'shape':[m,n,32],
            'checks':records,'resources':resources,'timing':timing,'paired_speed_ratios':ratios,
            'input_provenance':payload['input_provenance'],
            'precision_policy':'FP16 base inputs; at most two-term transforms RN-quantized once to FP16; FP32 HMMA and ordered FP32 reconstruction/output',
            'timing_scope':'same-worker microprototype,32x32x32 output CTAs,32-node graphs,7 shuffled rounds; not competitive large GEMM or transformer-step timing',
            'oracle_scope':'integer exact; finite floating oracle rounds FP64 dot16 to FP32, permits stated tolerances for hardware HMMA internal arithmetic'}


def safe_run(payload):
    try:return run(payload)
    except Exception as exc:
        return {'passed':False,'error':str(exc),'traceback':traceback.format_exc()}
