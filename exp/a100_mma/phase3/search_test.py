"""Exact output and dependency checks for isolated tensor sequence search."""
from __future__ import annotations
import random
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import search_sequences as search
from openassembler.backend import ra
from openassembler.emu.machine import Launch, Machine, Memory
from openassembler.emu.program import load_listing
from openassembler.ptx.parse import parse_module
from tests.backend_harness import listing


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
@pytest.mark.parametrize("paired,overlap,preaddress", [(True,False,False),(False,True,False),
                                                     (True,True,False),(True,False,True)])
def test_full_tile_matches_exact_integer_oracle_with_poison(dtype,paired,overlap,preaddress):
    search.configure()
    search.backend.prepare()
    m,n,k=64,64,128
    source=search.make_source(m,n,k,dtype,paired_b=paired,overlap=overlap,
                              bm=64,bn=64,bk=32,wm=32,wn=32,
                              swizzle_a=3,swizzle_b=3,paired_preaddress=preaddress)
    kernel=parse_module(source).kernels['gemm']
    with search.compiler_variant(paired_b=paired):
        code,labels,words,exits,regs=search.backend.generate(kernel,'sm_80','safe')
    rng=random.Random(297)
    av=[rng.randrange(-2,3) for _ in range(m*k)]
    bv=[rng.randrange(-2,3) for _ in range(n*k)]
    expected=[sum(av[r*k+i]*bv[c*k+i] for i in range(k)) for r in range(m) for c in range(n)]
    def pack(values):
        if dtype=='f16':
            return b''.join(struct.pack('<e',x) for x in values)
        return b''.join(struct.pack('<H',struct.unpack('<I',struct.pack('<f',x))[0]>>16) for x in values)
    a,b,c=0x100000,0x200000,0x300000
    prog=load_listing(listing(code,labels))
    for poison in (None,0xDEADBEEF):
        mem=Memory();mem.write_bytes(a,pack(av));mem.write_bytes(b,pack(bv))
        Machine(prog,Launch(grid=(1,1,1),block=(128,1,1),params=struct.pack('<QQQ',a,b,c),
                            smem_static=16384,poison=poison),mem).run(max_steps=3000000)
        actual=[struct.unpack('<f',struct.pack('<I',mem.read(c+4*i,4)))[0] for i in range(m*n)]
        assert actual==expected


def test_half_b_ir_declares_every_implicit_source_read():
    search.configure()
    source=search.make_source(128,128,128,'f16',paired_b=True)
    s=search.HalfBSelector(parse_module(source).kernels['gemm'])
    ops=ra.coalesce(ra.dce(s.run()))
    phys=ra.allocate(ops,s.r1)
    search.assert_allocation(ops,s.r1,phys)
    count=0
    for op in ops:
        if not op.fmt.startswith('HMMA'):
            continue
        assert len(op.uses)==4
        assert phys[op.uses[3]]==phys[op.uses[1]]+1
        assert phys[op.uses[1]]%2==0
        assert isinstance(op.uses[0],tuple) and len(op.uses[0])==4
        count+=1
    assert count==256


@pytest.mark.parametrize('paired',[False,True])
@pytest.mark.parametrize('bias,shift',[(1,0),(2,0),(0,1),(0,3)])
def test_register_search_preserves_abi_liveness_and_group_constraints(paired,bias,shift):
    search.configure()
    source=search.make_source(128,128,128,'f16',paired_b=paired)
    with search.compiler_variant(paired_b=paired,role_bias=bias,permute=shift):
        s=search.backend.TensorSelector(parse_module(source).kernels['gemm'])
        ops=ra.coalesce(ra.dce(s.run()))
        phys=ra.allocate(ops,s.r1)
    search.assert_allocation(ops,s.r1,phys)
    assert max(phys.values())<250
