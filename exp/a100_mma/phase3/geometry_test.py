"""Exercise five-stage fill, wrap, drain and paired fragment register mapping."""
import random
import re
import struct
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import geometry
from openassembler.emu.machine import Machine,Memory,Launch
from openassembler.emu.program import load_listing
from openassembler.ptx.parse import parse_module
from tests.backend_harness import listing

def test_five_stage_wrap_and_drain_matches_entire_integer_output():
    geometry.search.configure()
    m=n=64;k=224;bk=32;wm=wn=32
    source=geometry.five_stage_generator(m,n,k,'f16',m,n,bk,wm,wn,compact=True,stages=5)
    pattern=r'add.u32 %r41,%r200,%r9;\n.*?(?=\nbar.sync 0;)'
    assert len(re.findall(pattern,source,flags=re.S))==5
    replacement=geometry.search.compute_sequence('f16',bk=bk,wm=wm,wn=wn,paired_b=True)
    source=re.sub(pattern,lambda _:replacement,source,flags=re.S)
    source=geometry.vectorize_epilogue(source,n=n,wm=wm,wn=wn,address_mode='row')
    kernel=parse_module(source).kernels['gemm']
    with geometry.five_stage_support():
        code,labels,words,exits,regs=geometry.backend.generate(kernel,'sm_80','safe')
    assert any(text=='DEPBAR.LE SB0, 0x3' for text,ctl in code)
    rng=random.Random(553)
    av=[rng.randrange(-2,3) for _ in range(m*k)]
    bv=[rng.randrange(-2,3) for _ in range(n*k)]
    expected=[sum(av[r*k+i]*bv[c*k+i] for i in range(k)) for r in range(m) for c in range(n)]
    a,b,c=0x100000,0x200000,0x300000
    mem=Memory()
    mem.write_bytes(a,b''.join(struct.pack('<e',v) for v in av))
    mem.write_bytes(b,b''.join(struct.pack('<e',v) for v in bv))
    Machine(load_listing(listing(code,labels)),
            Launch(grid=(1,1,1),block=(128,1,1),params=struct.pack('<QQQ',a,b,c),
                   smem_static=40960,poison=0xDEADBEEF),mem).run(max_steps=4000000)
    actual=[struct.unpack('<f',struct.pack('<I',mem.read(c+4*i,4)))[0] for i in range(m*n)]
    assert actual==expected
