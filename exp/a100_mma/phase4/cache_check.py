"""Functional copy probe and exact binary-policy audit before hardware checks."""
from __future__ import annotations

import json
import random
import struct
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from openassembler.emu.machine import Launch,Machine,Memory
from openassembler.emu.program import load_listing


def check(hint,poison):
    rng=random.Random(3517);source=bytes(rng.randrange(256) for _ in range(512))
    a,c=0x10000000,0x20000000
    memory=Memory();memory.write_bytes(a,source)
    sentinel=bytes([0xCE])*512;memory.write_bytes(c,sentinel)
    program=load_listing((HERE/f'cache_probe_{hint}.sass').read_text(),kernel='probe')
    machine=Machine(program,Launch(block=(32,1,1),params=struct.pack('<QQ',a,c),
        smem_static=512,poison=poison),memory)
    machine.run(max_steps=10000)
    expected=bytearray(sentinel)
    for lane in range(32):expected[lane*16:lane*16+4]=source[lane*16:lane*16+4]
    actual=bytes(memory.read(c+i,1) for i in range(512))
    shared=bytes(machine.smem.read(i,1) for i in range(512))
    row={'hint_bytes':hint,'poison':poison,'output_exact':actual==expected,
         'all512_shared_copy_bytes_exact':shared==source,'steps':machine.steps,
         'scope':'functional SASS model; actual L2-prefetch behavior and asynchronous hardware must be checked on GPU'}
    print(json.dumps(row),flush=True)
    if actual!=expected or shared!=source:raise AssertionError('copy semantics mismatch')
    return row


if __name__=='__main__':
    rows=[check(hint,poison) for hint in (0,64,128,256) for poison in (None,0xDEADBEEF)]
    (HERE/'cache_copy_semantics.json').write_text(json.dumps(rows,indent=2)+'\n')
