"""Full-output verified cuBLASLt heuristic sweep against matched GemmEx.

Each dtype runs in a fresh GPU process. Accepted candidates are timed together
on the same quantized inputs, stream, output mode, and randomized graph rounds.
"""
from __future__ import annotations

import base64
import ctypes as C
import dataclasses
import hashlib
import json
import random
import statistics
import sys
import traceback
from pathlib import Path

from lt_abi import Algo,Heuristic,algorithm_config,bind,check,load_library,validate_header


def run(config,result):
    import torch
    from openassembler.gpu.cuda_driver import Driver
    header=validate_header()
    result.update({'header_validation':header,'configuration':config,
                   'precision_policy':'quantized FP16/BF16 inputs; CUBLAS_COMPUTE_32F=68; FP32 accumulation/output; alpha=1,beta=0; default math mode',
                   'layout':'C^T[N,M] column-major; Lt A=Bstorage[K,N],transA=T; Lt B=Astorage[K,M],transB=N',
                   'heuristic_queries':[],'candidates':[],'checks':[]})
    torch.cuda.init();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    drv=Driver()
    result['device']=dataclasses.asdict(drv.info());result['torch']=torch.__version__
    lt,lt_path=load_library('libcublasLt');blas,blas_path=load_library('libcublas')
    bind(lt,blas)
    result['libraries']={'lt':lt_path,'blas':blas_path}
    m,n,k=config.get('shape',[4096,4096,4096])
    dtype_name=config['dtype'];dtype_id=2 if dtype_name=='f16' else 14
    dtype=torch.float16 if dtype_name=='f16' else torch.bfloat16
    requested=config.get('requested',32)
    if requested not in (16,32):raise ValueError('bounded heuristic request must be16 or32')
    shape=(m,n,k);result['shape_mnk']=shape
    a=torch.empty((m,k),device='cuda',dtype=dtype)
    b=torch.empty((n,k),device='cuda',dtype=dtype)
    pad=256
    storage=torch.full((m*n+pad*2,),-12345.,device='cuda',dtype=torch.float32)
    output=storage[pad:-pad].view(m,n)
    workspace=torch.empty((32*1024*1024,),device='cuda',dtype=torch.uint8)
    one,zero=C.c_float(1),C.c_float(0)
    handle=C.c_void_p();lt_handle=C.c_void_p();desc=C.c_void_p();preference=C.c_void_p()
    layouts=[];modules=[]
    check(blas.cublasCreate_v2(C.byref(handle)),'cublasCreate')
    check(blas.cublasSetMathMode(handle,0),'default math mode')
    version=C.c_int();mode=C.c_int()
    check(blas.cublasGetVersion_v2(handle,C.byref(version)),'cublasGetVersion')
    check(blas.cublasGetMathMode(handle,C.byref(mode)),'cublasGetMathMode')
    result['cublas_version']=version.value;result['math_mode']=mode.value
    check(lt.cublasLtCreate(C.byref(lt_handle)),'cublasLtCreate')
    check(lt.cublasLtMatmulDescCreate(C.byref(desc),68,0),'FP32 compute/scale descriptor')
    transa,transb=C.c_int(1),C.c_int(0)
    check(lt.cublasLtMatmulDescSetAttribute(desc,3,C.byref(transa),C.sizeof(transa)),'Lt transA')
    check(lt.cublasLtMatmulDescSetAttribute(desc,4,C.byref(transb),C.sizeof(transb)),'Lt transB')
    for typ,rows,cols,ld in ((dtype_id,k,n,k),(dtype_id,k,m,k),(0,n,m,n),(0,n,m,n)):
        layout=C.c_void_p()
        check(lt.cublasLtMatrixLayoutCreate(C.byref(layout),typ,rows,cols,ld),'column-major layout')
        layouts.append(layout)
    ad,bd,cd,dd=layouts
    check(lt.cublasLtMatmulPreferenceCreate(C.byref(preference)),'Lt preference')

    def gemmex():
        stream=C.c_void_p(torch.cuda.current_stream().cuda_stream)
        check(blas.cublasSetStream_v2(handle,stream),'GemmEx stream')
        check(blas.cublasGemmEx(handle,1,0,n,m,k,C.byref(one),b.data_ptr(),dtype_id,k,
            a.data_ptr(),dtype_id,k,C.byref(zero),output.data_ptr(),0,n,68,99),'matched GemmEx')

    launches={'gemmex':gemmex};algorithms={};records={}
    for cap in (0,32*1024*1024):
        limit=C.c_uint64(cap)
        check(lt.cublasLtMatmulPreferenceSetAttribute(preference,1,C.byref(limit),C.sizeof(limit)),'Lt workspace preference')
        found=(Heuristic*requested)();count=C.c_int()
        status=lt.cublasLtMatmulAlgoGetHeuristic(lt_handle,desc,ad,bd,cd,dd,preference,
                                              requested,found,C.byref(count))
        query={'workspace_cap_bytes':cap,'requested':requested,'returned':count.value,'status':status}
        result['heuristic_queries'].append(query)
        if status:continue
        if not 0<=count.value<=requested:raise RuntimeError('invalid heuristic count')
        for rank in range(count.value):
            h=found[rank]
            algo=Algo.from_buffer_copy(bytes(h.algo))
            key=hashlib.sha256(bytes(algo)).hexdigest()
            name=f'lt_{key[:12]}'
            record={'name':name,'heuristic_rank':rank,'workspace_cap_bytes':cap,
                'workspace_required_bytes':h.workspaceSize,'heuristic_state':h.state,
                'waves_count':h.wavesCount,'algo_bytes_hex':bytes(algo).hex(),
                'config':algorithm_config(lt,algo,header)}
            result['candidates'].append(record)
            if h.state or h.workspaceSize>cap:
                record['rejected']='heuristic state or workspace exceeds cap';continue
            checked=Heuristic()
            state=lt.cublasLtMatmulAlgoCheck(lt_handle,desc,ad,bd,cd,dd,C.byref(algo),C.byref(checked))
            record['algo_check']={'return_status':state,'state':checked.state,
                                  'workspace_required_bytes':checked.workspaceSize}
            if state or checked.state or checked.workspaceSize>cap:
                record['rejected']='AlgoCheck rejected or workspace exceeds cap';continue
            if name in launches:
                record['duplicate_of']=name;continue
            required=max(h.workspaceSize,checked.workspaceSize)
            algorithms[name]=algo
            records[name]=record
            def launch(algo=algo,required=required):
                stream=C.c_void_p(torch.cuda.current_stream().cuda_stream)
                check(lt.cublasLtMatmul(lt_handle,desc,C.byref(one),b.data_ptr(),ad,
                    a.data_ptr(),bd,C.byref(zero),output.data_ptr(),cd,output.data_ptr(),dd,
                    C.byref(algo),workspace.data_ptr() if required else None,required,stream),'Lt matmul')
            launches[name]=launch

    # Optional native OpenAssembler controls use the same three-pointer ABI and
    # are subjected to the identical complete-output checks and timing rounds.
    pointers=[C.c_uint64(t.data_ptr()) for t in (a,b,output)]
    params=(C.c_void_p*3)(*[C.cast(C.pointer(p),C.c_void_p) for p in pointers])
    for extra in config.get('extras',[]):
        mod=drv.load(base64.b64decode(extra['cubin']));modules.append(mod)
        fn=drv.function(mod,'gemm');shared=extra.get('smem',0)
        if shared:
            drv.cu.cuFuncSetAttribute.argtypes=[C.c_void_p,C.c_int,C.c_int]
            drv.ck(drv.cu.cuFuncSetAttribute(fn,8,shared),'native dynamic shared')
        extra['resources']=drv.func_attrs(fn)
        def native(fn=fn,extra=extra,shared=shared):
            stream=C.c_void_p(torch.cuda.current_stream().cuda_stream)
            drv.ck(drv.cu.cuLaunchKernel(fn,*extra['grid'],extra['threads'],1,1,shared,stream,params,None),'native paired control')
        launches[extra['name']]=native

    # Check all outputs on every accepted unique algorithm. Integer/zero cases
    # are exact; floating checks allow different legal FP32 reduction orders.
    enabled=set(launches)
    nr,nc=min(m,32),min(n,32)
    rows=torch.arange(nr,device='cuda')*max(1,m//nr)
    cols=torch.arange(nc,device='cuda')*max(1,n//nc)
    for pattern in ('zero','integer','normal','scaled'):
        torch.manual_seed(817 if pattern!='scaled' else 829)
        if pattern=='zero':a.zero_();b.normal_()
        elif pattern=='integer':
            a.copy_(torch.randint(-3,4,(m,k),device='cuda').to(dtype))
            b.copy_(torch.randint(-3,4,(n,k),device='cuda').to(dtype))
        else:
            a.normal_();b.normal_()
            if pattern=='scaled':a.mul_(0.03125);b.mul_(32.)
        truth=a[rows].cpu().double()@b[cols].cpu().double().T
        baseline=None
        for name in ['gemmex']+[x for x in launches if x!='gemmex' and x in enabled]:
            storage.fill_(-12345.);output.fill_(float('nan'))
            try:
                launches[name]();torch.cuda.synchronize()
            except Exception as exc:
                result['checks'].append({'name':name,'pattern':pattern,'passed':False,'launch_error':str(exc)})
                if name=='gemmex':raise
                enabled.discard(name)
                continue
            guard=bool((storage[:pad]==-12345.).all() and (storage[-pad:]==-12345.).all())
            finite=bool(torch.isfinite(output).all())
            if name=='gemmex':baseline=output.clone()
            difference=output-baseline
            full_rel=float(torch.linalg.vector_norm(difference)/torch.linalg.vector_norm(baseline).clamp_min(1e-30))
            full_max=float(difference.abs().max())
            sample=output[rows[:,None],cols[None,:]].cpu().double()
            error=sample-truth
            sample_rel=float(torch.linalg.vector_norm(error)/torch.linalg.vector_norm(truth).clamp_min(1e-30))
            exact=bool(torch.equal(output,baseline)) and bool(torch.equal(sample,truth))
            passed=guard and finite and full_rel<=2e-5 and sample_rel<=2e-5 and (exact or pattern not in ('zero','integer'))
            result['checks'].append({'name':name,'pattern':pattern,'passed':passed,
                'full_output_elements_checked':m*n,'guard_regions_passed':guard,'finite':finite,
                'full_relative_l2_vs_gemmex':full_rel,'full_max_abs_vs_gemmex':full_max,
                'cpu_float64_oracle_sample_shape':[nr,nc],'sample_relative_l2':sample_rel,
                'sample_max_abs':float(error.abs().max()),'exact':exact})
            if not passed:
                if name=='gemmex':raise AssertionError('GemmEx baseline failed oracle')
                enabled.discard(name)

    graphs={};nodes=config.get('graph_nodes',20)
    # normal distribution for all timing; quantized identical operands remain
    # fixed across every candidate and baseline graph.
    torch.manual_seed(841);a.normal_();b.normal_()
    for name,launch in launches.items():
        if name not in enabled:continue
        try:
            for _ in range(5):launch()
            torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(nodes):launch()
            graph.replay();torch.cuda.synchronize();graphs[name]=graph
        except Exception as exc:
            result.setdefault('graph_rejections',[]).append({'name':name,'error':str(exc)})
            if name=='gemmex':raise
    for _ in range(config.get('warmup_rounds',8)):
        for graph in graphs.values():graph.replay()
    torch.cuda.synchronize()
    samples={name:[] for name in graphs};rng=random.Random(43)
    order_records=[]
    for _ in range(7):
        order=list(graphs);rng.shuffle(order);order_records.append(order)
        for name in order:
            begin,end=(torch.cuda.Event(enable_timing=True) for _ in range(2))
            begin.record();graphs[name].replay();end.record();end.synchronize()
            samples[name].append(begin.elapsed_time(end)/nodes)
    medians={name:statistics.median(values) for name,values in samples.items()}
    best=min((name for name in medians if name.startswith('lt_')),key=medians.get,default=None)
    best_by_cap={}
    for cap in (0,32*1024*1024):
        choices={x['name'] for x in result['candidates'] if x['workspace_cap_bytes']==cap and 'rejected' not in x and x['name'] in medians}
        winner=min(choices,key=medians.get,default=None)
        best_by_cap[str(cap)]={'name':winner,'milliseconds':medians.get(winner),
            'gemmex_over_best':medians['gemmex']/medians[winner] if winner else None}
    result.update({'passed':bool(best),'milliseconds':medians,'samples_ms':samples,
        'paired_round_orders':order_records,'best_lt':best,'best_by_workspace_cap':best_by_cap,
        'unique_heuristics':len(algorithms),'correctness_accepted':sorted(enabled),
        'all_executed_candidate_checks_passed':all(x['passed'] for x in result['checks']),
        'failed_checks':sum(not x['passed'] for x in result['checks']),
        'graph_nodes':nodes,'warmup_rounds':config.get('warmup_rounds',8),
        'tflops':{name:2*m*n*k/(ms*1e9) for name,ms in medians.items()},
        'scope':'public cuBLASLt heuristic candidates, not exhaustive autotuning or extraction of proprietary kernels; same-worker FP32-output baseline comparison'})
    check(lt.cublasLtMatmulPreferenceDestroy(preference),'preference destroy')
    for layout in layouts:check(lt.cublasLtMatrixLayoutDestroy(layout),'layout destroy')
    check(lt.cublasLtMatmulDescDestroy(desc),'descriptor destroy')
    check(lt.cublasLtDestroy(lt_handle),'Lt destroy')
    check(blas.cublasDestroy_v2(handle),'GemmEx destroy')
    for mod in modules:drv.unload(mod)
    return result


def safe_run(config):
    result={}
    try:return run(config,result)
    except Exception as exc:
        result.update({'passed':False,'error':str(exc),'traceback':traceback.format_exc()})
        return result


if __name__=='__main__':
    config_path,result_path=map(Path,sys.argv[1:3])
    result=safe_run(json.loads(config_path.read_text()))
    # Binary payloads are already hashed in the local runner; retain hashes
    # and configuration without needlessly duplicating base64 in result files.
    for extra in result.get('configuration',{}).get('extras',[]):extra.pop('cubin',None)
    result_path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'dtype':result.get('configuration',{}).get('dtype'),
        'passed':result.get('passed'),'best':result.get('best_by_workspace_cap'),
        'error':result.get('error')},indent=2),flush=True)
