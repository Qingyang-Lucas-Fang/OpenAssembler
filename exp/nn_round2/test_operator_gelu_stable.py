"""Stable tanh-GELU derivative validates against unchanged FP64 gate."""
import math
import pytest
from exp.nn_round2.operator_gelu_stable import generate_module
from exp.nn_round2.test_operator_suite import _emulate,_f32
from openassembler.ptx.parse import parse_module


def test_stable_gelu_fixed_abi_and_same_original_gate():
    text,kernels=generate_module();module=parse_module(text)
    assert len(kernels)==len(module.kernels)==6
    assert {(k['rtol'],k['atol']) for k in kernels}=={(8e-5,2e-5)}
    assert all(len(kernel.params)==8 for kernel in module.kernels.values())


@pytest.mark.parametrize('mode',['scalar','scalar4','vector4'])
def test_stable_gelu_derivative_complete_outputs_guards_and_bounded_stress(mode):
    text,kernels=generate_module()
    k=next(k for k in kernels if k['candidate']=='stable_ex2' and k['mode']==mode)
    module=parse_module(text);name=k['kernel']
    # Extract just this source, leaving the native parser to enforce all fields.
    source=text[text.index('.visible .entry '+name):]
    source=source.split('\n.visible .entry',1)[0]
    xs=[-12.,-8.,-4.,-3.,-2.,-.752,-.75,-.748,-1e-6,0.,1e-6,.75,2.,3.,4.,8.,12.]
    n=64;x=[_f32(xs[i%len(xs)]) for i in range(n)];z=[_f32(math.sin(i*.31)*100) for i in range(n)]
    actual,_=_emulate(source,name,[x,z],[n],n,128)
    for got,a,dy in zip(actual[0],x,z):
        t=math.tanh(.7978845608028654*(a+.044715*a*a*a))
        reference=dy*(.5*(1+t)+.5*a*(1-t*t)*.7978845608028654*(1+3*.044715*a*a))
        assert math.isfinite(got) and abs(got-reference)<=k['atol']+k['rtol']*abs(reference)
