"""Check both fused-staging algorithms at nonzero CTA with poisoned registers."""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0,str(BASE))
import backend
from openassembler.emu.machine import Launch,Machine,Memory
from openassembler.emu.program import load_listing
from lowrank_oracle import compare,direct_reference,inputs,reference


def check(config,family):
    m,n=config['m'],config['n']
    av,bv=inputs(m,n,family)
    expected=reference(av,bv,m,n,config['scheme'])
    direct=direct_reference(av,bv,m,n)
    a,b,c=0x10000000,0x20000000,0x30000000
    memory=Memory()
    memory.write_bytes(a,b''.join(struct.pack('<e',x) for x in av))
    memory.write_bytes(b,b''.join(struct.pack('<e',x) for x in bv))
    sentinel=0x7FC12345
    memory.write_bytes(c,struct.pack('<I',sentinel)*(m*n))
    directory=BASE/config['build']
    entry=json.loads((directory/'manifest.json').read_text())['entries'][0]
    program=load_listing((directory/entry['sass']).read_text(),kernel='gemm')
    cta=(1,1,0)
    machine=Machine(program,Launch(grid=(m//32,n//32,1),block=(32,1,1),ctaid=cta,
        params=struct.pack('<3Q',a,b,c),smem_static=1024,poison=0xDEADBEEF),memory)
    machine.run()
    actual=[];wanted=[];outside=0
    for row in range(m):
        for col in range(n):
            bits=memory.read(c+(row*n+col)*4,4)
            if cta[0]*32<=row<(cta[0]+1)*32 and cta[1]*32<=col<(cta[1]+1)*32:
                actual.append(struct.unpack('<f',struct.pack('<I',bits))[0])
                wanted.append(expected[row*n+col])
            else:outside+=bits!=sentinel
    result={'name':config['name'],'family':family,'cta':cta,
            'matched_policy':compare(actual,wanted,family=='integer'),
            'outside_cta_writes':outside,'emulator_warp_instructions':machine.steps,
            'matched_oracle_vs_direct_max_abs':max(abs(x-y) for x,y in zip(expected,direct)),
            'scope':'functional SASS emulation only; physical HMMA rounding/timing requires GPU'}
    print(json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    cases=json.loads((HERE/'lowrank_suite.json').read_text())
    results=[check(case,family) for case in cases for family in ('integer','floating','cancellation')]
    (HERE/'lowrank_emulation.json').write_text(json.dumps(results,indent=2)+'\n')
    if any(not x['matched_policy']['passed'] or x['outside_cta_writes'] for x in results):
        raise SystemExit(1)
