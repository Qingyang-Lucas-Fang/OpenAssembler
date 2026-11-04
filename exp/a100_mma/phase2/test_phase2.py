"""Contracts exposed by the hardware failures and multistage extension."""
import json,sys
from pathlib import Path
import pytest
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE))
import backend
from openassembler.backend.compile import Out
from openassembler.backend import ra
from openassembler.ptx.parse import parse_module
from openassembler.isa.word import control_valid
from pipeline import pipelined_gemm


@pytest.fixture(autouse=True)
def options():
    keys=('OPTIMIZE_LOCALS','HOIST_LOOP_INVARIANTS','RELAX_TENSOR','MINED_SCALAR',
          'TENSOR_ISSUE_GAP','LDSM_READ_BARRIER','LIST_SCHEDULE','TENSOR_REUSE',
          'EXPERIMENTAL_RECYCLE_WITHOUT_WAIT','INLINE_POINTER_INCREMENTS')
    old={key:getattr(backend,key) for key in keys}
    backend.OPTIMIZE_LOCALS=True;backend.HOIST_LOOP_INVARIANTS=False
    backend.RELAX_TENSOR=True;backend.MINED_SCALAR=True;backend.TENSOR_ISSUE_GAP=8
    backend.LDSM_READ_BARRIER=False;backend.LIST_SCHEDULE=False;backend.TENSOR_REUSE=False
    yield
    for key,value in old.items():setattr(backend,key,value)


def test_scoreboard_recycling_cannot_drop_an_unconsumed_load():
    # More outstanding loads than available normal scoreboards. All outputs
    # are consumed later, and shared addresses are independently live.
    body=[Out(f'LDSM.16.M88.2 R{20+2*i}, [R{i}]',True,[i],[20+2*i,21+2*i]) for i in range(8)]
    body += [Out('STG.E.128 [R100.64], R20',False,[100,101,20,21,22,23],[])]
    code=backend.tensor_schedule(body,{})
    active=set()
    for text,ctl in code:
        new={x for x in (ctl.wbar,ctl.rbar) if x!=7}
        assert all(ctl.wait&(1<<bar) for bar in new&active)
        active.difference_update(bar for bar in tuple(active) if ctl.wait&(1<<bar))
        active.update(new)
        assert 0 not in new


@pytest.mark.parametrize('stages',[2,3,4])
def test_multiple_pending_groups_keep_reserved_completion_counter(stages):
    source=pipelined_gemm(128,128,256,'f16',128,128,32,64,32,compact=True,stages=stages)
    k=parse_module(source).kernels['gemm']
    code,*_=backend.generate(k,'sm_80','safe')
    waits=[]
    for text,ctl in code:
        assert control_valid(ctl)
        if text=='LDGDEPBAR':assert ctl.wbar==0
        else:assert ctl.wbar!=0
        if text.startswith('DEPBAR'):
            assert ctl.stall>=4
            waits.append(int(text.rsplit(' ',1)[1],16))
    assert max(waits)==stages-2
    assert waits[-1]==0


@pytest.mark.parametrize('wait',[1,2])
def test_threshold_words_roundtrip_through_nvidia_decoder(wait):
    from openassembler import oracle
    text=f'DEPBAR.LE SB0, {wait:#x}'
    assert oracle.disasm_words([backend.encoder().encode(text)])==[text]


def test_invalid_memory_layout_and_stage_budget_are_rejected():
    with pytest.raises(ValueError):pipelined_gemm(128,128,256,stages=5)
    with pytest.raises(ValueError):pipelined_gemm(128,128,256,bk=32,swizzle_a=7)
    with pytest.raises(ValueError):pipelined_gemm(128,128,256,compact=False,b_window=1)


def test_fused_bias_uses_distinct_parameter_abi():
    source=pipelined_gemm(128,128,128,'f16',128,128,64,64,64,compact=True,bias=True)
    kernel=parse_module(source).kernels['gemm']
    assert kernel.params==[('u64','A'),('u64','B'),('u64','C'),('u64','Bias')]
    code,*_=backend.generate(kernel,'sm_80','safe')
    # Every epilogue value must remain defined with the extra pointer live.
    assert sum(text.startswith('FADD') for text,ctl in code)==128


def test_reported_gpu_evidence_really_contains_complete_output_checks():
    root=BASE/'results'
    for name in ('p2_paired_windows','p2_banking_paired'):
        record=json.loads((root/f'{name}.json').read_text())[0]
        assert 'error' not in record
        assert len(record['checks'])>=24
        for check in record['checks']:
            assert check['full_output_elements_checked']==4096*4096
            assert check['relative_l2']<=2e-5


@pytest.mark.parametrize('start',[0x1234FFFFFFC0,0xFFFFFFFFFFFFFFC0,0x123400000040])
def test_inline_pointer_increment_handles_low_word_and_full_64bit_wrap(start):
    import struct
    from openassembler.emu.machine import Machine,Memory,Launch
    from openassembler.emu.program import load_listing
    from tests.backend_harness import listing
    backend.INLINE_POINTER_INCREMENTS=True
    source=''' .version 8.7
.target sm_80
.address_size 64
.visible .entry pointers(.param .u64 output) {
.reg .b32 %i; .reg .b64 %ptr; .reg .pred %p;
ld.param.u64 %ptr,[output];mov.u32 %i,0;
LOOP: st.global.u32 [%ptr],%i;
add.u64 %ptr,%ptr,128;add.u32 %i,%i,1;setp.lt.u32 %p,%i,4;@%p bra LOOP;
ret; }
'''
    code,labels,*_=backend.generate(parse_module(source).kernels['pointers'],'sm_80','safe')
    for poison in (None,0xDEADBEEF):
        memory=Memory()
        program=load_listing(listing(code,labels))
        Machine(program,Launch(block=(1,1,1),params=struct.pack('<Q',start),poison=poison),memory).run()
        for i in range(4):assert memory.read((start+i*128)&((1<<64)-1),4)==i


def test_reuse_hints_never_alias_a_tensor_destination():
    from openassembler.emu.program import parse_insn
    backend.TENSOR_REUSE=True;backend.INLINE_POINTER_INCREMENTS=True
    source=pipelined_gemm(128,128,128,'f16',128,128,64,64,64,compact=True,b_window=8)
    code,*_=backend.generate(parse_module(source).kernels['gemm'],'sm_80','safe')
    count=0
    for (text,ctl),(nexttext,_) in zip(code,code[1:]):
        if not ctl.reuse:continue
        current=parse_insn(0,text,0);following=parse_insn(0,nexttext,0)
        assert current.op==following.op=='HMMA'
        for index,width in ((1,4),(2,2)):
            if ctl.reuse&(1<<(index-1)):
                assert current.ops[index]==following.ops[index]
                assert not set(range(current.ops[0].idx,current.ops[0].idx+4))&set(range(current.ops[index].idx,current.ops[index].idx+width))
                count+=1
    assert count>0
