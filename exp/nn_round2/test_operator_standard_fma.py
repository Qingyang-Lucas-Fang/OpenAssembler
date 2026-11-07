"""Standard-width candidates retain masks, synchronization and analytic gates."""
import math
import pytest
from exp.nn_round2.operator_standard_fma import generate_module
from exp.nn_round2.operator_candidates import fma_local
from exp.nn_round2.operator_suite import ROWWISE,rowwise_source
from exp.nn_round2.test_operator_suite import _emulate,_f32
from openassembler.ptx.parse import parse_module


def test_standard_width_source_matrix_and_unmodified_gates():
    source,kernels=generate_module();module=parse_module(source)
    assert len(kernels)==len(module.kernels)==16
    assert {k['width'] for k in kernels}=={512,1024}
    assert {k['op'] for k in kernels}=={'rmsnorm_backward','softmax_backward'}
    for k in kernels:assert (k['rtol'],k['atol'])==ROWWISE[k['op']][2]


@pytest.mark.parametrize('op',['rmsnorm_backward','softmax_backward'])
@pytest.mark.parametrize('width',[512,1024])
def test_standard_width_fma_full_native_outputs_match_fp64(op,width):
    rows=3;x=[_f32(math.sin(i*.23)) for i in range(rows*width)];z=[_f32(math.cos(i*.31)*.25) for i in range(rows*width)]
    if op=='softmax_backward':
        for r in range(rows):
            xx=x[r*width:(r+1)*width];ee=[math.exp(v-max(xx)) for v in xx];x[r*width:(r+1)*width]=[_f32(v/sum(ee)) for v in ee]
    name,source=rowwise_source(op,width,4);candidate,_=fma_local(source,op,width)
    actual,code=_emulate(candidate,name,[x,z],[rows*width],rows,128)
    assert sum(t.startswith('WARPSYNC') for t,_ in code)==source.count('shfl.sync')
    expected=[]
    for r in range(rows):
        xx=x[r*width:(r+1)*width];zz=z[r*width:(r+1)*width]
        if op=='softmax_backward':
            dot=sum(a*b for a,b in zip(xx,zz));yy=[a*(b-dot) for a,b in zip(xx,zz)]
        else:
            inv=1/math.sqrt(sum(a*a for a in xx)/width+1e-5);dot=sum(a*b for a,b in zip(xx,zz))/width
            yy=[inv*(b-a*inv*inv*dot) for a,b in zip(xx,zz)]
        expected.extend(yy)
    rtol,atol=ROWWISE[op][2]
    for got,ref in zip(actual[0],expected):assert math.isfinite(got) and abs(got-ref)<=atol+rtol*abs(ref)
