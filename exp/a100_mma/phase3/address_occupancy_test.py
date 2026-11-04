"""Exact full-CTA integer-oracle checks for the bounded eight-warp sweep."""
from __future__ import annotations

import random
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import address_occupancy as experiment
from openassembler import oracle
from openassembler.emu.machine import Launch,Machine,Memory
from openassembler.emu.program import load_listing
from openassembler.ptx.parse import parse_module
from tests.backend_harness import listing


@pytest.mark.parametrize("dtype",["f16","bf16"])
@pytest.mark.parametrize("family,wm,wn,folded",experiment.FAMILIES)
def test_all_eight_warps_match_full_integer_oracle_with_poison(dtype,family,wm,wn,folded):
    experiment.combined.search.configure(gap=8,reuse=True)
    m,n,k = 128,128,128
    source = experiment.make_source(m,n,k,dtype,wm,wn,folded)
    with experiment.context(folded):
        code,labels,words,exits,regs = experiment.combined.search.backend.generate(
            parse_module(source).kernels["gemm"],"sm_80","safe")
    rng = random.Random(226)
    av = [rng.randrange(-2,3) for _ in range(m*k)]
    bv = [rng.randrange(-2,3) for _ in range(n*k)]
    expected = [sum(av[r*k+i]*bv[c*k+i] for i in range(k))
                for r in range(m) for c in range(n)]
    def pack(values):
        if dtype == "f16":
            return b"".join(struct.pack("<e",value) for value in values)
        return b"".join(struct.pack("<H",struct.unpack("<I",struct.pack("<f",value))[0]>>16)
                        for value in values)
    a,b,c = 0x100000,0x200000,0x300000
    memory = Memory(); memory.write_bytes(a,pack(av)); memory.write_bytes(b,pack(bv))
    memory.write_bytes(c,b"\x00\x00\xc0\x7f"*(m*n))
    program = load_listing(listing(code,labels))
    Machine(program,Launch(grid=(1,1,1),block=(256,1,1),
                           params=struct.pack("<QQQ",a,b,c),smem_static=65536,
                           poison=0xDEADBEEF),memory).run(max_steps=6000000)
    got = [struct.unpack("<f",struct.pack("<I",memory.read(c+4*i,4)))[0]
           for i in range(m*n)]
    assert got == expected
    assert sum(text.startswith("STG.E.64") for text,ctl in code) == 32
    assert all(not text.startswith("LDSM.16.M88.2") for text,ctl in code)
    # The binary includes trailing alignment padding after the scheduled body.
    # Check actual copy words instead of assuming equal body/padded lengths.
    copies = [(text,words[index]) for index,(text,ctl) in enumerate(code)
              if text.startswith("LDGSTS")]
    assert oracle.disasm_words([word for text,word in copies]) == [text for text,word in copies]
