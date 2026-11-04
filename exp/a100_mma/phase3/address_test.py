"""Semantic, range and binary checks for folded LDGSTS shared addresses."""
from __future__ import annotations

import random
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import address_shared as address
from address_stage import specialize_two_stage
from openassembler import oracle
from openassembler.emu.machine import Machine, Memory, Launch
from openassembler.emu.program import load_listing
from openassembler.isa import encode
from openassembler.ptx.parse import parse_module
from tests.backend_harness import listing


@pytest.mark.parametrize("bm,bn,bk,wm,wn,mask", [(128,128,64,64,64,7),
                                                       (64,64,32,32,32,3)])
def test_shared_immediate_preserves_every_copy_address(bm,bn,bk,wm,wn,mask):
    threads = bm//wm*(bn//wn)*32
    width = bk//8
    stride = 2*bk*(bm+bn)
    for stage in (0,stride):
        for which,rows in ((0,bm),(1,bn)):
            shared_base = 0 if which == 0 else 2*bm*bk
            for tid in range(threads):
                base_row,base_col = divmod(tid,width)
                base_col *= 8
                base = 2*(base_row*bk+(base_col ^ ((base_row&mask)*8)))+shared_base
                for chunk in range(rows*width//threads):
                    row,col = divmod(tid+chunk*threads,width)
                    col *= 8
                    old = 2*(row*bk+(col ^ ((row&mask)*8)))+shared_base+stage
                    new = base+stage+chunk*(threads//width)*bk*2
                    assert old == new
                    assert old % 16 == 0 and 0 <= old < 2*stride


def test_encoder_rejects_unproven_forms_and_out_of_range_offsets():
    enc = address.AddressEncoder(address.search.backend.encoder())
    for text in ("LDGSTS.E.BYPASS.128 [R0+0x80000], [R2.64]",
                 "LDGSTS.E.BYPASS.128 [R0], [R2.64+0x800]",
                 "LDGSTS.E.BYPASS.128 [R0+0x1], [R2.64]",
                 "LDGSTS.E.BYPASS.128 [R0], [R3.64]",
                 "LDGSTS.E.BYPASS.128 [R0], [R2.64+UR4]",
                 "LDGSTS.E.BYPASS.128.ZFILL [R0], [R2.64]"):
        with pytest.raises(encode.EncodeError):
            enc.encode(text)


@pytest.mark.parametrize("dtype,stage_pair", [("f16",False),("bf16",False),
                                             ("f16",True),("bf16",True)])
def test_encoded_gemm_matches_full_integer_oracle_with_poison(dtype,stage_pair):
    address.search.configure(gap=8,reuse=True)
    m,n,k = 64,64,256 if stage_pair else 128
    source = address.make_source(m,n,k,dtype,bm=64,bn=64,bk=32,wm=32,wn=32,
                                 swizzle_a=3,swizzle_b=3)
    if stage_pair:
        source = specialize_two_stage(source,k=k,bm=64,bn=64,bk=32)
    kernel = parse_module(source).kernels["gemm"]
    with address.address_backend():
        code,labels,words,exits,regs = address.search.backend.generate(kernel,"sm_80","safe")
    copies = [(text,word) for (text,ctl),word in zip(code,words) if text.startswith("LDGSTS")]
    assert len(copies) == (16 if stage_pair else 8)
    actual = oracle.disasm_words([word for text,word in copies])
    assert actual == [text for text,word in copies]
    assert any("+0x800]" in text for text,word in copies)
    rng = random.Random(211)
    av = [rng.randrange(-2,3) for _ in range(m*k)]
    bv = [rng.randrange(-2,3) for _ in range(n*k)]
    expected = [sum(av[r*k+i]*bv[c*k+i] for i in range(k))
                for r in range(m) for c in range(n)]
    pack = (lambda values: b"".join(struct.pack("<e",v) for v in values)) if dtype == "f16" else (
        lambda values: b"".join(struct.pack("<H",struct.unpack("<I",struct.pack("<f",v))[0] >> 16)
                               for v in values))
    a,b,c = 0x100000,0x200000,0x300000
    program = load_listing(listing(code,labels))
    for poison in (None,0xDEADBEEF):
        memory = Memory()
        memory.write_bytes(a,pack(av)); memory.write_bytes(b,pack(bv))
        Machine(program,Launch(grid=(1,1,1),block=(128,1,1),
                               params=struct.pack("<QQQ",a,b,c),
                               smem_static=16384,poison=poison),memory).run(max_steps=3000000)
        got = [struct.unpack("<f",struct.pack("<I",memory.read(c+4*i,4)))[0]
               for i in range(m*n)]
        assert got == expected
