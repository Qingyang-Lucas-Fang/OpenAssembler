"""Precision/dependency tests for local FMA reduction candidates."""
import math
import pytest
from exp.nn_round2.operator_candidates import OPS,fma_local,generate_module
from exp.nn_round2.operator_suite import ROWWISE,rowwise_source
from exp.nn_round2.test_operator_suite import _emulate,_f32
from openassembler.ptx.parse import parse_module


def test_fma_candidates_have_same_abi_masks_and_sync_count():
    source,kernels=generate_module();module=parse_module(source)
    assert len(kernels)==len(module.kernels)==60
    for op in OPS:
        for width in (1,33,128,513):
            name,baseline=rowwise_source(op,width,4);candidate,changes=fma_local(baseline,op,width)
            assert baseline.count('shfl.sync')==candidate.count('shfl.sync')
            assert baseline.count('@%p1')==candidate.count('@%p1')
            assert candidate.count('fma.rn.f32')==sum(c['explicit_fmas'] for c in changes)


@pytest.mark.parametrize('op',OPS)
@pytest.mark.parametrize('width',[1,33,128,513])
def test_fma_native_candidate_retains_original_fp64_gate_and_canaries(op,width):
    rows=3
    x=[_f32(math.sin(i*.31)+(1000. if op.startswith('layernorm') else .75)) for i in range(rows*width)]
    z=[_f32(math.cos(i*.21)*.25) for i in range(rows*width)]
    if op=='softmax_backward':
        for r in range(rows):
            xs=x[r*width:(r+1)*width];ee=[math.exp(v-max(xs)) for v in xs]
            x[r*width:(r+1)*width]=[_f32(v/sum(ee)) for v in ee]
    name,source=rowwise_source(op,width,4);candidate,_=fma_local(source,op,width)
    actual,code=_emulate(candidate,name,[x,z],[rows*width],rows,128)
    assert sum(t.startswith('WARPSYNC') for t,_ in code)==source.count('shfl.sync')
    expected=[]
    for r in range(rows):
        xx=x[r*width:(r+1)*width];zz=z[r*width:(r+1)*width]
        if op=='softmax_backward':
            dot=sum(a*b for a,b in zip(xx,zz));yy=[a*(b-dot) for a,b in zip(xx,zz)]
        else:
            centered=[v-sum(xx)/width for v in xx] if op.startswith('layernorm') else xx
            inv=1/math.sqrt(sum(v*v for v in centered)/width+1e-5)
            if op.endswith('backward'):
                mean_dy=sum(zz)/width if op.startswith('layernorm') else 0
                dot=sum(a*b for a,b in zip(centered,zz))/width
                yy=[inv*(b-mean_dy-a*inv*inv*dot) for a,b in zip(centered,zz)]
            else:yy=[a*inv for a in centered]
        expected.extend(yy)
    rtol,atol=ROWWISE[op][2]
    for got,ref in zip(actual[0],expected):assert math.isfinite(got) and abs(got-ref)<=atol+rtol*abs(ref)
