"""CPU contract/native-emulator checks for the isolated arithmetic campaign."""
import math
import struct

import pytest

from exp.nn_round2.operator_suite import (POINTWISE, ROWWISE, generate_module,
                                          pointwise_source, rowwise_source)
from openassembler.build import generate
from openassembler.emu.machine import Launch, Machine, Memory
from openassembler.emu.program import load_listing
from openassembler.ptx.parse import parse_module
from tests.backend_harness import listing


def _f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]


def test_all_cases_keep_fixed_abi_unique_names_and_no_unproven_sync_elision():
    text,kernels=generate_module();module=parse_module(text)
    assert len(kernels)==len(module.kernels)==94
    assert set(k['op'] for k in kernels)==set(POINTWISE)|set(ROWWISE)
    abi=[('u64',f'X{i}') for i in range(4)]+[('u64',f'Y{i}') for i in range(3)]+[('u32','N')]
    assert all(kernel.params==abi for kernel in module.kernels.values())
    assert 'shfl.sync.bfly.b32' in text
    for k in kernels:
        if k['kind']=='pointwise':assert k['elements_per_thread'] in (1,4)
        else:assert k['warps'] in (4,8)


def _emulate(source,name,inputs,outputs,n,threads):
    kernel=parse_module('.version 8.7\n.target sm_80\n.address_size 64\n'+source).kernels[name]
    code,labels,_,_,_=generate(kernel,'sm_80','opt')
    program=load_listing(listing(code,labels));memory=Memory()
    pointers=[]
    for i,values in enumerate(inputs):
        pointer=0x100000000+i*0x100000
        memory.write_bytes(pointer,struct.pack('<%df'%len(values),*values));pointers.append(pointer)
    while len(pointers)<4:pointers.append(pointers[0])
    guard=0xc640e400;storages=[]
    for i,count in enumerate(outputs):
        storage=0x200000000+i*0x100000;pointer=storage+256
        memory.write_bytes(storage,struct.pack('<%dI'%(count+128),*([guard]*64+[0x7fc00000]*count+[guard]*64)))
        pointers.append(pointer);storages.append((storage,pointer,count))
    while len(pointers)<7:pointers.append(pointers[4])
    Machine(program,Launch(block=(threads,1,1),grid=(1,1,1),ctaid=(0,0,0),params=struct.pack('<7QI',*pointers,n),poison=0xdeadbeef),memory).run()
    actual=[]
    for storage,pointer,count in storages:
        assert all(memory.read(storage+4*i,4)==guard for i in range(64))
        assert all(memory.read(pointer+4*(count+i),4)==guard for i in range(64))
        actual.append([struct.unpack('<f',struct.pack('<I',memory.read(pointer+4*i,4)))[0] for i in range(count)])
    return actual,code


@pytest.mark.parametrize('op',sorted(ROWWISE))
@pytest.mark.parametrize('width',[1,33,128])
def test_new_row_math_full_outputs_both_tails_and_preserved_warp_dependencies(op,width):
    rows=3
    x=[_f32(math.sin(i*.31)+(.75 if i%3 else -.5)) for i in range(rows*width)]
    z=[_f32(math.cos(i*.21)*.25) for i in range(rows*width)]
    if op=='softmax_backward':
        for r in range(rows):
            values=x[r*width:(r+1)*width];e=[math.exp(v-max(values)) for v in values]
            x[r*width:(r+1)*width]=[_f32(v/sum(e)) for v in e]
    name,source=rowwise_source(op,width,4)
    counts=[rows] if op=='row_sum' else [rows*width,rows] if op=='cross_entropy' else [rows*width]
    actual,code=_emulate(source,name,[x,z],counts,rows,128)
    communication=[text for text,_ in code if text.startswith(('WARPSYNC','SHFL'))]
    assert len(communication)>0 and all(t.startswith('WARPSYNC') for t in communication[::2])
    reference=[[],[]]
    for r in range(rows):
        xx=x[r*width:(r+1)*width];zz=z[r*width:(r+1)*width]
        if op in ('softmax','cross_entropy'):
            e=[math.exp(v-max(xx)) for v in xx];p=[v/sum(e) for v in e]
            if op=='cross_entropy':p[0]-=1;reference[1].append(math.log(sum(e))+max(xx)-xx[0])
            yy=p
        elif op=='softmax_backward':
            dot=sum(a*b for a,b in zip(xx,zz));yy=[a*(b-dot) for a,b in zip(xx,zz)]
        elif op=='row_sum':yy=[sum(xx)]
        else:
            centered=[v-sum(xx)/width for v in xx] if op.startswith('layernorm') else xx
            inv=1/math.sqrt(sum(v*v for v in centered)/width+1e-5)
            if op.endswith('backward'):
                mean_dy=sum(zz)/width if op.startswith('layernorm') else 0
                dot=sum(a*b for a,b in zip(centered,zz))/width
                yy=[inv*(b-mean_dy-a*inv*inv*dot) for a,b in zip(centered,zz)]
            else:yy=[a*inv for a in centered]
        reference[0].extend(yy)
    rtol,atol=ROWWISE[op][2]
    for a,expected in zip(actual,reference):
        for got,ref in zip(a,expected):assert math.isfinite(got) and abs(got-ref)<=atol+rtol*abs(ref),(op,width,got,ref)


@pytest.mark.parametrize('op',['residual_add','relu','relu_backward','gated_multiply','dropout_mask_scale'])
@pytest.mark.parametrize('mode',['scalar','scalar4','vector4'])
def test_vector_grouping_produces_complete_guarded_outputs(op,mode):
    n=64;x=[_f32(math.sin(i*.3)) for i in range(n)];z=[float(i%2) for i in range(n)]
    name,source=pointwise_source(op,mode);actual,_=_emulate(source,name,[x,z],[n],n,128)
    if op=='relu':ref=[max(v,0) for v in x]
    elif op=='relu_backward':ref=[b if a>0 else 0 for a,b in zip(x,z)]
    elif op=='residual_add':ref=[_f32(a+b) for a,b in zip(x,z)]
    else:ref=[_f32(a*b*(1.25 if op=='dropout_mask_scale' else 1.)) for a,b in zip(x,z)]
    assert actual[0]==ref


def test_scalar_nonmultiple_tail_is_guarded():
    n=67;x=[float(i-33) for i in range(n)];name,source=pointwise_source('relu','scalar')
    actual,_=_emulate(source,name,[x],[n],n,128)
    assert actual[0]==[max(v,0) for v in x]
