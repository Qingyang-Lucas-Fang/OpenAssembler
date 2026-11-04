"""Numerical and dependency tests for paired-B plus row-epilogue composition."""
from __future__ import annotations
import random
import struct
import sys
from pathlib import Path
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import search_combined as combined
from openassembler.emu.machine import Launch,Machine,Memory
from openassembler.emu.program import load_listing
from openassembler.ptx.parse import parse_module
from tests.backend_harness import listing


@pytest.mark.parametrize('dtype',['f16','bf16'])
@pytest.mark.parametrize('role,gap',[(0,8),(1,8),(0,4)])
def test_combined_complete_tile_and_poison(dtype,role,gap):
    combined.search.configure(gap=gap,reuse=True)
    combined.search.backend.prepare()
    m,n,k=64,64,128
    source=combined.source_for(m,n,k,dtype,bm=64,bn=64,bk=32,wm=32,wn=32,
                               swizzle_a=3,swizzle_b=3)
    with combined.selector_and_allocator(True,role):
        code,labels,words,exits,regs=combined.search.backend.generate(
            parse_module(source).kernels['gemm'],'sm_80','safe')
    rng=random.Random(925)
    av=[rng.randrange(-2,3) for _ in range(m*k)];bv=[rng.randrange(-2,3) for _ in range(n*k)]
    expected=[sum(av[r*k+i]*bv[c*k+i] for i in range(k)) for r in range(m) for c in range(n)]
    def pack(values):
        if dtype=='f16':return b''.join(struct.pack('<e',x) for x in values)
        return b''.join(struct.pack('<H',struct.unpack('<I',struct.pack('<f',x))[0]>>16) for x in values)
    a,b,c=0x100000,0x200000,0x300000;program=load_listing(listing(code,labels))
    for poison in (None,0xDEADBEEF):
        memory=Memory();memory.write_bytes(a,pack(av));memory.write_bytes(b,pack(bv))
        Machine(program,Launch(grid=(1,1,1),block=(128,1,1),params=struct.pack('<QQQ',a,b,c),
                                smem_static=16384,poison=poison),memory).run(max_steps=3000000)
        actual=[struct.unpack('<f',struct.pack('<I',memory.read(c+4*i,4)))[0] for i in range(m*n)]
        assert actual==expected
    assert all(not text.startswith('LDSM.16.M88.2') for text,ctl in code)
    assert sum(text.startswith('STG.E.64') for text,ctl in code)==16


def test_combined_selector_preserves_both_implicit_halves_and_scalar_dependencies():
    combined.search.configure();source=combined.source_for(128,128,128,'f16')
    selector=combined.CombinedSelector(parse_module(source).kernels['gemm'])
    ops=combined.search.ra.coalesce(combined.search.ra.dce(selector.run()))
    phys=combined.search.ra.allocate(ops,selector.r1)
    combined.search.assert_allocation(ops,selector.r1,phys)
    tensor=[op for op in ops if op.fmt.startswith('HMMA')]
    stores=[op for op in ops if op.fmt.startswith('STG.E.64')]
    assert len(tensor)==256 and len(stores)==64
    for op in tensor:assert len(op.uses)==4 and phys[op.uses[3]]==phys[op.uses[1]]+1
    for op in stores:assert len(op.uses)==3 and phys[op.uses[2]]==phys[op.uses[1]]+1
