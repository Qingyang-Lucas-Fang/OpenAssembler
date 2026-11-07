"""Precision repair candidate for tanh-GELU backward, no tolerance relaxation.

The tanh approximation's small error is amplified by 1-tanh(u)^2 in the
ordinary derivative. Compute sigmoid(2u) and sech(u)^2 from exp(-2|u|)
instead. This is the same analytic tanh-GELU derivative, not erf-GELU.
The tested inputs are bounded finite FP32; general exceptional values remain
outside this experiment's contract.
"""
from __future__ import annotations
import base64
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parents)>2 else Path.cwd()
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
HERE=ROOT/'exp/nn_round2';BUILD=HERE/'operator_gelu_stable_build'


def stable_arithmetic(j):
    from exp.nn_round2.operator_suite import fp
    x,z,y=f'%f{j}',f'%f{32+j}',f'%f{128+j}'
    a,b,c,d,e,t=(f'%f{224+8*j+k}' for k in range(6))
    return [f'mul.f32 {a},{x},{x};',f'mul.f32 {b},{a},{fp(.044715)};',f'add.f32 {b},{b},{fp(1)};',f'mul.f32 {b},{b},{x};',f'mul.f32 {b},{b},{fp(.7978845608028654)};',f'abs.f32 {c},{b};',f'mul.f32 {c},{c},{fp(-2.8853900817779268)};',f'ex2.approx.ftz.f32 {c},{c};',f'add.f32 {d},{c},{fp(1)};',f'rcp.approx.ftz.f32 {d},{d};',f'mul.f32 {e},{c},{d};',f'setp.ge.f32 %p1,{b},{fp(0)};',f'selp.f32 {e},{d},{e},%p1;',f'mul.f32 {a},{a},{fp(3*.044715)};',f'add.f32 {a},{a},{fp(1)};',f'mul.f32 {a},{a},{fp(.7978845608028654)};',f'mul.f32 {t},{d},{d};',f'mul.f32 {t},{t},{c};',f'mul.f32 {t},{t},{x};',f'mul.f32 {t},{t},{fp(2)};',f'mul.f32 {t},{t},{a};',f'add.f32 {t},{t},{e};',f'mul.f32 {y},{t},{z};']


def generate_module():
    from exp.nn_round2.operator_suite import POINTWISE,arithmetic,pointwise_source
    sources=[];kernels=[];op='gelu_tanh_backward'
    for mode in ('scalar','scalar4','vector4'):
        name,original=pointwise_source(op,mode);candidate=original
        ept=1 if mode=='scalar' else 4
        for j in range(ept):
            old='\n'.join(arithmetic(op,j))
            if candidate.count(old)!=1:raise ValueError('GELU derivative block not unique')
            candidate=candidate.replace(old,'\n'.join(stable_arithmetic(j)),1)
        for variant,body in (('tanh',original),('stable_ex2',candidate)):
            kname=name+'_'+variant;body=body.replace('.entry '+name+'(','.entry '+kname+'(',1);sources.append(body)
            kernels.append({'kernel':kname,'op':op,'mode':mode,'candidate':variant,'kind':'pointwise','elements_per_thread':ept,'inputs':2,'outputs':1,'dtype':'float','rtol':POINTWISE[op][3][0],'atol':POINTWISE[op][3][1]})
    return '.version 8.7\n.target sm_80\n.address_size 64\n'+'\n'.join(sources),kernels


def build_payload():
    from openassembler.build import Options,compile_paths
    BUILD.mkdir(exist_ok=True,parents=True);source,kernels=generate_module();path=BUILD/'gelu_stable.ptx';path.write_text(source);out=BUILD/'compiled'
    result=compile_paths(path,out,Options(),o_name='gelu_stable')
    if result.rc:raise RuntimeError('stable GELU backward build failed')
    manifest=json.loads((out/'manifest.json').read_text());entry=manifest['entries'][0]
    if any(k['compiled_by']!='openassembler' for k in entry['kernels']):raise RuntimeError('native-only comparison required')
    payload={'schema':1,'kernels':kernels,'manifest':manifest,'source':source,'source_sha256':hashlib.sha256(source.encode()).hexdigest()}
    for method,key in (('oa','cubin'),('ptxas','reference')):
        data=(out/entry[key]).read_bytes();payload[method]=base64.b64encode(data).decode();payload[method+'_sha256']=hashlib.sha256(data).hexdigest()
    payload['source_identities']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ('exp/nn_round2/operator_suite.py','exp/nn_round2/operator_worker.py','exp/nn_round2/operator_gelu_stable.py','openassembler/backend/ra.py','openassembler/backend/compile.py','openassembler/backend/sched.py')}
    (BUILD/'payload.json').write_text(json.dumps(payload));return payload



def diagnose(payload):
    """Retain exact failing values and a wider bounded derivative stress grid."""
    import ctypes as C
    import torch
    from exp.nn_round2 import operator_worker
    from openassembler.gpu.cuda_driver import Driver
    driver=Driver();modules={m:driver.load(base64.b64decode(payload[m])) for m in ('oa','ptxas')};results=[]
    points=torch.tensor([-12.,-8.,-4.,-3.,-2.,-.752,-.75,-.748,-1e-6,0.,1e-6,.75,2.,3.,4.,8.,12.],device='cuda')
    gradients=torch.tensor([-100.,-1.,0.,1.,100.],device='cuda')
    xx=points.repeat_interleave(len(gradients));zz=gradients.repeat(len(points));stress=[xx,zz,torch.zeros_like(xx),torch.zeros_like(xx)]
    inputs_by_pattern={'normal_1m':operator_worker.make_inputs('gelu_tanh_backward',(1048576,),'normal',20261107),'bounded_stress':stress}
    try:
        for pattern,inputs in inputs_by_pattern.items():
            expected=operator_worker.reference('gelu_tanh_backward',inputs)[0]
            for k in payload['kernels']:
                if k['mode']!='scalar':continue
                for method,mod in modules.items():
                    storage=torch.full((inputs[0].numel()+512,),-12345.,device='cuda');output=storage[256:-256];output.fill_(float('nan'))
                    fn=driver.function(mod,k['kernel']);tensors=inputs+[output,output,output]
                    args=[C.c_uint64(t.data_ptr()) for t in tensors]+[C.c_uint32(output.numel())]
                    params=(C.c_void_p*len(args))(*[C.cast(C.pointer(a),C.c_void_p) for a in args])
                    driver.ck(driver.cu.cuLaunchKernel(fn,(output.numel()+127)//128,1,1,128,1,1,0,C.c_void_p(torch.cuda.current_stream().cuda_stream),params,None),'diagnostic derivative');torch.cuda.synchronize()
                    close=torch.isclose(output.double(),expected,rtol=k['rtol'],atol=k['atol']);indices=torch.nonzero(~close).flatten()[:32]
                    results.append({'pattern':pattern,'kernel':k['kernel'],'candidate':k['candidate'],'method':method,'checked_elements':output.numel(),'mismatched_elements':int((~close).sum()),'max_abs':float((output.double()-expected).abs().max()),'canaries':bool((storage[:256]==-12345).all() and (storage[-256:]==-12345).all()),'failing_values':[{'index':int(i),'x':float(inputs[0][i]),'dy':float(inputs[1][i]),'actual':float(output[i]),'expected_fp64':float(expected[i]),'absolute_error':float((output[i].double()-expected[i]).abs()),'allowed_error':k['atol']+k['rtol']*abs(float(expected[i]))} for i in indices]})
    finally:
        for mod in modules.values():driver.unload(mod)
    return results


def run(payload,samples=19,graph_repeats=32,rounds=2):
    from exp.nn_round2 import operator_worker
    results=[];start=time.monotonic()
    for ri in range(rounds):
        order=list(payload['kernels'])
        if ri%2:order.reverse()
        for k in order:
            # All ownership modes and candidates share the same exact inputs.
            result=operator_worker.run({**payload,'kernels':[k]},samples=samples,graph_repeats=graph_repeats,sizes=(65536,1048576),launch_threads=(128,256),seed=20261107)
            for case in result['cases']:case.update(round=ri,input_seed=20261107,candidate=k['candidate'])
            results.extend(result['cases'])
    return {'schema':1,'cases':results,'runtime':result['runtime'],'methodology':{**result['methodology'],'rounds':rounds,'input_seeding':'identical20261107 across kernels, variants, geometries and rounds','candidate':'stable sigmoid(2u)/sech² derivative of same tanh-GELU; original tolerances unchanged'},'source_sha256':payload['source_sha256'],'source_identities':payload['source_identities'],'oa_sha256':payload['oa_sha256'],'ptxas_sha256':payload['ptxas_sha256'],'diagnostics':diagnose(payload),'elapsed_seconds':time.monotonic()-start}


if __name__=='__main__':
    p=build_payload();print(json.dumps({'kernels':len(p['kernels']),'payload':str(BUILD/'payload.json'),'source_sha256':p['source_sha256']},indent=2))
