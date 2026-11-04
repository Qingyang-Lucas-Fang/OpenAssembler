"""Semantic and binary-contract checks for the isolated compiler extension."""
import dataclasses
import random
import struct

import pytest

import backend
from backend import TensorSelector, encoder, generate
from kernels import warp_gemm
from openassembler.backend import ra
from openassembler.container.cubin import words_to_bytes
from openassembler.emu.machine import Launch, Machine, Memory
from openassembler.emu.program import load_listing
from openassembler.isa.word import control, control_valid
from openassembler.ptx.parse import parse_module


@pytest.fixture(autouse=True)
def restore_options():
    keys=('OPTIMIZE_LOCALS','HOIST_LOOP_INVARIANTS','RELAX_TENSOR','MINED_SCALAR','TENSOR_ISSUE_GAP','LDSM_READ_BARRIER','LIST_SCHEDULE','TENSOR_REUSE')
    old={k:getattr(backend,k) for k in keys}
    yield
    for k,v in old.items():setattr(backend,k,v)


def test_mma_loop_layout_and_liveness_against_integer_oracle():
    backend.OPTIMIZE_LOCALS=True
    backend.RELAX_TENSOR=True
    backend.MINED_SCALAR=True
    kernel=parse_module(warp_gemm(16,8,48)).kernels['gemm']
    code,labels,words,exits,regs=generate(kernel,'sm_80','safe')
    from tests.backend_harness import listing
    rng=random.Random(91)
    a=[rng.randrange(-3,4) for _ in range(16*48)]
    b=[rng.randrange(-3,4) for _ in range(8*48)]
    expected=[sum(a[row*48+q]*b[col*48+q] for q in range(48)) for row in range(16) for col in range(8)]
    for poison in (None,0xDEADBEEF):
        mem=Memory();ap,bp,cp=0x10000,0x20000,0x30000
        mem.write_bytes(ap,b''.join(struct.pack('<e',v) for v in a))
        mem.write_bytes(bp,b''.join(struct.pack('<e',v) for v in b))
        prog=load_listing(listing(code,labels))
        Machine(prog,Launch(block=(32,1,1),params=struct.pack('<3Q',ap,bp,cp),poison=poison),mem).run()
        actual=[struct.unpack('<f',struct.pack('<I',mem.read(cp+4*i,4)))[0] for i in range(128)]
        assert actual==expected
    assert all(control_valid(control(word)) for word in words)


@pytest.mark.parametrize('replacement',[
    'mma.sync.aligned.m16n8k16.col.col.f32.f16.f16.f32',
    'mma.sync.aligned.m16n8k16.row.col.f32.bf16.f16.f32',
    'mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32',
])
def test_unsupported_tensor_arithmetic_is_an_error(replacement):
    ptx=warp_gemm(16,8,16).replace('mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32',replacement)
    with pytest.raises(ra.NotSupported):
        TensorSelector(parse_module(ptx).kernels['gemm']).run()


def test_async_completion_and_synchronization_contract():
    from pipeline import pipelined_gemm
    backend.OPTIMIZE_LOCALS=True
    backend.RELAX_TENSOR=True
    backend.MINED_SCALAR=True
    kernel=parse_module(pipelined_gemm(64,64,128,bm=64,bn=64,wm=32,wn=32)).kernels['gemm']
    code,labels,words,exits,regs=generate(kernel,'sm_80','safe')
    for text,ctl in code:
        if text=='LDGDEPBAR':assert ctl.wbar==0
        else:assert ctl.wbar!=0
        assert ctl.rbar!=0
        if text.startswith('BAR.'):assert ctl.stall>=6
        if text.startswith('DEPBAR.'):assert ctl.stall>=4
    assert all(control_valid(control(word)) for word in words)


def test_async_register_fields_have_independent_operand_effects():
    from openassembler import oracle
    enc=encoder()
    texts=['LDGSTS.E.BYPASS.128 [R0], [R8.64]',
           'LDGSTS.E.BYPASS.128 [R9], [R12.64]',
           'LDGSTS.E.BYPASS.128 [R31], [R24.64]']
    assert oracle.disasm_words([enc.encode(text) for text in texts])==texts


def test_shared_address_cannot_change_before_all_load_readers_complete():
    from openassembler.backend.compile import Out
    backend.MINED_SCALAR=False
    backend.LDSM_READ_BARRIER=False
    body=[Out('LDSM.16.M88.4 R4, [R0]',True,[0],[4,5,6,7]),
          Out('LDSM.16.M88.4 R8, [R0]',True,[0],[8,9,10,11]),
          Out('IADD3 R0, R2, R3, RZ',False,[2,3],[0])]
    code=backend.tensor_schedule(body,{})
    writers={code[0][1].wbar,code[1][1].wbar}
    assert all(code[-1][1].wait & (1<<bar) for bar in writers)
    assert all(code[i][1].rbar==7 for i in (0,1))
