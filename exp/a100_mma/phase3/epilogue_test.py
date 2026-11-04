"""Full output-address and integer-oracle checks for vectorized epilogues."""
from __future__ import annotations
import random
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import epilogue
from openassembler.backend import ra
from openassembler.emu.machine import Machine, Memory, Launch
from openassembler.emu.program import load_listing
from openassembler.ptx.parse import parse_module
from tests.backend_harness import listing


@pytest.mark.parametrize("address_mode", ["pair", "row"])
def test_every_cta_epilogue_maps_to_exactly_one_aligned_output_pair(address_mode):
    # Check algebraic addressing across the complete grid, both row halves of
    # every warp fragment and all lanes, with a non-square output.
    m,n,bm,bn,wm,wn=256,384,128,128,64,64
    seen={}
    for cta_m in range(m//bm):
        for cta_n in range(n//bn):
            for warp in range(4):
                warp_m=(warp//2)*wm;warp_n=(warp%2)*wn
                for lane in range(32):
                    r=cta_m*bm+warp_m+lane//4
                    c=cta_n*bn+warp_n+2*(lane%4)
                    for mi in range(wm//16):
                        for ni in range(wn//8):
                            for dr in (0,8):
                                index=(r+mi*16+dr)*n+c+ni*8
                                assert (index*4)%8==0
                                for scalar in (index,index+1):
                                    assert scalar not in seen
                                    seen[scalar]=(cta_m,cta_n,warp,lane,mi,ni,dr)
    assert set(seen)==set(range(m*n))


@pytest.mark.parametrize("variant", ["pair_packed", "pair_native", "row_native"])
def test_epilogue_binary_backend_matches_full_integer_cpu_oracle(variant):
    # One complete 64x64 CTA keeps the tensor emulator check bounded. The
    # geometry uses two warp fragments in M and four in N, including both
    # low/high accumulator pairs and nonzero store offsets.
    epilogue.configure()
    m,n,k=64,64,128
    source=epilogue.pipelined_gemm(m,n,k,'f16',64,64,32,32,32,compact=True,b_window=4)
    source=epilogue.vectorize_epilogue(source,n=n,wm=32,wn=32,
                                     address_mode='row' if variant=='row_native' else 'pair')
    kernel=parse_module(source).kernels['gemm']
    with epilogue.native_pair_selector(variant!='pair_packed'):
        code,labels,words,exits,regs=epilogue.backend.generate(kernel,'sm_80','safe')
    rng=random.Random(151)
    av=[rng.randrange(-2,3) for _ in range(m*k)]
    bv=[rng.randrange(-2,3) for _ in range(n*k)]
    expected=[sum(av[r*k+i]*bv[c*k+i] for i in range(k)) for r in range(m) for c in range(n)]
    a,b,c=0x100000,0x200000,0x300000
    prog=load_listing(listing(code,labels))
    for poison in (None,0xDEADBEEF):
        mem=Memory()
        mem.write_bytes(a,b''.join(struct.pack('<e',x) for x in av))
        mem.write_bytes(b,b''.join(struct.pack('<e',x) for x in bv))
        Machine(prog,Launch(grid=(1,1,1),block=(128,1,1),params=struct.pack('<QQQ',a,b,c),
                            smem_static=16384,poison=poison),mem).run(max_steps=3000000)
        actual=[struct.unpack('<f',struct.pack('<I',mem.read(c+4*i,4)))[0] for i in range(m*n)]
        assert actual==expected
    assert sum(text.startswith('STG.E.64') for text,ctl in code)==16
    assert not any(text.startswith('STG.E ') for text,ctl in code)


def test_native_pair_declaration_keeps_both_scalar_source_dependencies():
    epilogue.configure()
    source=epilogue.pipelined_gemm(128,128,128,'f16',128,128,64,64,64,compact=True,b_window=8)
    source=epilogue.vectorize_epilogue(source,n=128,address_mode='row')
    selector=epilogue.NativePairSelector(parse_module(source).kernels['gemm'])
    ops=selector.run()
    assert selector.native_pair_stores==64
    for op in ops:
        if op.fmt.startswith('STG.E.64'):
            assert len(op.uses)==3
            assert isinstance(op.uses[0],tuple) and len(op.uses[0])==2
            assert all(isinstance(value,ra.V) for value in op.uses[1:])
