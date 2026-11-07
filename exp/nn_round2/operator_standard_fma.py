"""Test local-FMA backward reductions at standard model widths, experimentally."""
from __future__ import annotations
import base64
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parents)>2 else Path.cwd()
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
HERE=ROOT/'exp/nn_round2';BUILD=HERE/'operator_standard_fma_build'


def generate_module():
    from exp.nn_round2.operator_candidates import fma_local
    from exp.nn_round2.operator_suite import ROWWISE,rowwise_source
    sources=[];kernels=[]
    for width in (512,1024):
        for op in ('softmax_backward','rmsnorm_backward'):
            for warps in (4,8):
                original_name,original=rowwise_source(op,width,warps)
                candidate,replacements=fma_local(original,op,width)
                for mode,body in (('baseline',original),('fma',candidate)):
                    name=original_name+'_'+mode;body=body.replace('.entry '+original_name+'(','.entry '+name+'(',1);sources.append(body)
                    kernels.append({'kernel':name,'op':op,'mode':f'{mode}_warp{warps}','candidate':mode,'kind':'rowwise','width':width,'warps':warps,'inputs':2,'outputs':1,'dtype':'float','rtol':ROWWISE[op][2][0],'atol':ROWWISE[op][2][1],'changed_local_reductions':replacements if mode=='fma' else [],'rounding_policy':'explicit local FMA' if mode=='fma' else 'unconstrained PTX mul/add; ptxas may contract'})
    return '.version 8.7\n.target sm_80\n.address_size 64\n'+'\n'.join(sources),kernels


def build_payload():
    from openassembler.build import Options,compile_paths
    BUILD.mkdir(parents=True,exist_ok=True);source,kernels=generate_module();path=BUILD/'standard_fma.ptx';path.write_text(source);out=BUILD/'compiled'
    result=compile_paths(path,out,Options(),o_name='standard_fma')
    if result.rc:raise RuntimeError('standard-width FMA build failed')
    manifest=json.loads((out/'manifest.json').read_text());entry=manifest['entries'][0]
    if len(entry['kernels'])!=len(kernels) or any(k['compiled_by']!='openassembler' for k in entry['kernels']):raise RuntimeError('native-only compilation required')
    payload={'schema':1,'kernels':kernels,'manifest':manifest,'source':source,'source_sha256':hashlib.sha256(source.encode()).hexdigest()}
    for method,key in (('oa','cubin'),('ptxas','reference')):
        data=(out/entry[key]).read_bytes();payload[method]=base64.b64encode(data).decode();payload[method+'_sha256']=hashlib.sha256(data).hexdigest()
    payload['source_identities']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ('exp/nn_round2/operator_standard_fma.py','exp/nn_round2/operator_suite.py','exp/nn_round2/operator_candidates.py','exp/nn_round2/operator_worker.py','openassembler/backend/ra.py','openassembler/backend/compile.py','openassembler/backend/sched.py')}
    (BUILD/'payload.json').write_text(json.dumps(payload));return payload


if __name__=='__main__':
    p=build_payload();print(json.dumps({'kernels':len(p['kernels']),'source_sha256':p['source_sha256'],'payload':str(BUILD/'payload.json')},indent=2))
