"""Poisoned integer SASS checks and original per-output K-order comparison."""
from __future__ import annotations

from collections import Counter,defaultdict
import json
import random
import struct
from pathlib import Path

import fragment_windows as fragment
from openassembler import oracle
from openassembler.emu.machine import Launch,Machine,Memory
from openassembler.emu.program import load_listing
from openassembler.ptx.parse import parse_module
from tests.backend_harness import listing

HERE=Path(__file__).resolve().parent


def accumulation_order(sequence):
    # Compare each output vector's chronological sequence of K-slice addresses.
    # Register windows intentionally rename B registers, so use load-address
    # provenance instead of comparing printed register numbers.
    kernel=parse_module('.version8.7\n.target sm_80\n.address_size64\n.visible .entry gemm(){\n'+sequence+'\nret;\n}').kernels['gemm']
    values={};orders=defaultdict(list);current_k=None
    for ins in kernel.body:
        if ins.op=='add' and getattr(ins.args[0],'name','')=='%r41':
            current_k=(int(ins.args[1].name.removeprefix('%r'))-200)//4
        if ins.op=='ldmatrix' and len(ins.args[0].elems)==4:
            address=ins.args[1]
            if address.base=='%r42':
                first_column=address.off//(16*64)
                for component,var in enumerate(ins.args[0].elems):
                    values[var.name]=(current_k,first_column+component//2)
        if ins.op=='mma':
            output=tuple(x.name for x in ins.args[0].elems)
            b=ins.args[2].elems
            orders[output].append(values[b[0].name])
    return dict(orders)


def check(dtype,window,prefetch):
    backend=fragment.initialize()
    old=fragment.address.search.compute_sequence(dtype,bk=64,wm=64,wn=64,paired_b=True)
    new=fragment.compute(dtype,bk=64,wm=64,wn=64,window=window,prefetch=prefetch)
    if accumulation_order(old)!=accumulation_order(new):raise AssertionError('peroutputKorderchanged')
    for op in ('mma.sync','ldmatrix.sync'):
        if old.count(op)!=new.count(op):raise AssertionError('semanticopcountchanged')
    m=n=64;k=128
    source=fragment.make_source(m,n,k,dtype,window=window,prefetch=prefetch,
                                  bm=64,bn=64,bk=32,wm=64,wn=64)
    kernel=parse_module(source).kernels['gemm']
    with fragment.address.address_backend():
        code,labels,words,exits,regs=backend.generate(kernel,'sm_80','safe')
    copies=[word for (text,ctl),word in zip(code,words) if text.startswith(('LDSM','HMMA','LDGSTS'))]
    actual=oracle.disasm_words(copies)
    expected=[text for text,ctl in code if text.startswith(('LDSM','HMMA','LDGSTS'))]
    norm=lambda x:''.join(x.replace('.reuse','').split())
    if [norm(x) for x in actual]!=[norm(x) for x in expected]:raise AssertionError('instructionroundtripfailed')
    rng=random.Random(947)
    av=[rng.randrange(-2,3) for _ in range(m*k)]
    bv=[rng.randrange(-2,3) for _ in range(n*k)]
    def pack(values):
        if dtype=='f16':return b''.join(struct.pack('<e',v) for v in values)
        return b''.join(struct.pack('<H',struct.unpack('<I',struct.pack('<f',v))[0]>>16) for v in values)
    a,b,c=0x10000000,0x20000000,0x30000000
    memory=Memory();memory.write_bytes(a,pack(av));memory.write_bytes(b,pack(bv))
    sentinel=0x7FC12345;memory.write_bytes(c,struct.pack('<I',sentinel)*(m*n+64))
    program=load_listing(listing(code,labels))
    machine=Machine(program,Launch(block=(32,1,1),params=struct.pack('<QQQ',a,b,c),
        smem_static=16384,poison=0xDEADBEEF),memory)
    machine.run(max_steps=1000000)
    expected=[sum(av[r*k+q]*bv[col*k+q] for q in range(k)) for r in range(m) for col in range(n)]
    actual=[struct.unpack('<f',struct.pack('<I',memory.read(c+i*4,4)))[0] for i in range(m*n)]
    bad=[(i,x,y) for i,(x,y) in enumerate(zip(actual,expected)) if x!=y]
    canary=all(memory.read(c+(m*n+i)*4,4)==sentinel for i in range(64))
    row={'dtype':dtype,'window':window,'prefetch':prefetch,'poison':0xDEADBEEF,
        'full_output_elements':m*n,'mismatches':len(bad),'examples':bad[:8],'guard_passed':canary,
        'per_output_k_order_preserved':True,'hmma_ldmatrix_counts_preserved':True,
        'exact_tensor_copy_nvdisasm':True,'registers':regs,'emulator_steps':machine.steps,
        'scope':'one64x64x128CTA functional model; actual GPU output/timing pending'}
    print(json.dumps(row),flush=True)
    if bad or not canary:raise AssertionError('integeroraclefailed')
    return row


if __name__=='__main__':
    rows=[check(dtype,window,prefetch) for dtype in ('f16','bf16')
          for window,prefetch in ((2,False),(4,False),(4,True))]
    (HERE/'fragment_semantics.json').write_text(json.dumps(rows,indent=2)+'\n')
