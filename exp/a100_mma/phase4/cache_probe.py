"""Independently recover SM80 LDGSTS L2 prefetch fields from matching PTX."""
from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from openassembler import oracle
from openassembler.isa.word import control,set_bits


def source(hint):
    suffix=f'.L2::{hint}B' if hint else ''
    return f'''.version 8.7
.target sm_80
.address_size 64
.visible .entry probe(.param .u64 A,.param .u64 C) {{
.reg .b32 %r<8>; .reg .b64 %d<5>;
.shared .align 16 .b8 smem[512];
ld.param.u64 %d0,[A];ld.param.u64 %d1,[C];
mov.u32 %r0,%tid.x;mul.lo.u32 %r1,%r0,16;
mul.wide.u32 %d2,%r0,16;add.u64 %d3,%d0,%d2;
cp.async.cg.shared.global{suffix} [%r1],[%d3],16;
cp.async.commit_group;cp.async.wait_group 0;bar.sync 0;
ld.shared.b32 %r2,[%r1];add.u64 %d4,%d1,%d2;
st.global.b32 [%d4],%r2;ret;
}}
'''


def discover():
    HERE.mkdir(parents=True,exist_ok=True)
    oracle.SCRATCH=HERE/'scratch'
    probe=[];policies={}
    for hint in (0,64,128,256):
        ptx=HERE/f'cache_probe_{hint}.ptx';ptx.write_text(source(hint))
        binary=HERE/f'cache_probe_{hint}.cubin'
        try:
            oracle.ptxas(ptx,binary)
            text=oracle.disasm_text(binary)
            (HERE/f'cache_probe_{hint}.sass').write_text(text)
            insns=[x for x in oracle.parse_listing(text) if x.text.startswith('LDGSTS')]
            if len(insns)!=1:raise AssertionError(f'{hint}: expected one copy')
            instruction=insns[0]
            policies[str(hint)]={'supported':True,'ptx':f'cp.async.cg.shared.global{f".L2::{hint}B" if hint else ""}',
                'word':f'{instruction.word:032x}','text':instruction.text}
        except subprocess.CalledProcessError as exc:
            policies[str(hint)]={'supported':False,'stderr':exc.stderr,'returncode':exc.returncode}
    base=int(policies['0']['word'],16)
    # Normalize source/destination register fields and controls before comparing
    # matching ptxas instructions. Everything else must be the hint alone.
    clear=((1<<48)-1)<<16 | ((1<<21)-1)<<105
    for hint,record in policies.items():
        if not record['supported']:continue
        word=int(record['word'],16)
        delta=(word^base)&~clear
        record['nonoperand_noncontrol_xor']=f'{delta:032x}'
        record['changed_bits']=[bit for bit in range(128) if delta>>bit&1]
        if int(hint)==0:continue
        expected=f'LDGSTS.E.BYPASS.LTC{hint}B.128'
        if not record['text'].startswith(expected):
            raise AssertionError((hint,record['text']))
        # Exhaustive all-four policy-field values must decode predictably.
        if delta & ~((1<<71)|(1<<72)):
            raise AssertionError('L2 hint changes an unaccounted instruction field')
    seed=set_bits(set_bits(base,16,8,16),24,8,8)
    allfield=[set_bits(seed,71,2,v) for v in range(4)]
    fieldtexts=oracle.disasm_words(allfield)
    expected=['LDGSTS.E.BYPASS.128 [R16], [R8.64]']+[
        f'LDGSTS.E.BYPASS.LTC{hint}B.128 [R16], [R8.64]' for hint in (64,128,256)]
    if fieldtexts!=expected:raise AssertionError((fieldtexts,expected))
    rng=random.Random(6749);roundtrips=[];words=[];expected=[]
    for hint,field in ((0,0),(64,1),(128,2),(256,3)):
        for i in range(20):
            dr=rng.randrange(255);sr=2*rng.randrange(127)
            do=rng.randrange(-32768,32768)*16;so=rng.randrange(-128,128)*16
            word=set_bits(set_bits(seed,16,8,dr),24,8,sr)
            word=set_bits(set_bits(word,44,20,do&1048575),32,12,so&4095)
            word=set_bits(word,71,2,field)
            def off(value):return (f'+-0x{-value:x}' if value<0 else f'+0x{value:x}') if value else ''
            text=f'LDGSTS.E.BYPASS.{f"LTC{hint}B." if hint else ""}128 [R{dr}{off(do)}], [R{sr}.64{off(so)}]'
            words.append(word);expected.append(text)
            roundtrips.append({'hint_bytes':hint,'destination_reg':dr,'source_reg':sr,
                'shared_offset':do,'global_offset':so,'word':f'{word:032x}','expected':text,
                'control':vars(control(word))})
    actual=oracle.disasm_words(words)
    if actual!=expected:raise AssertionError([(a,b) for a,b in zip(expected,actual) if a!=b])
    for row,text in zip(roundtrips,actual):row['actual']=text
    evidence={'arch':'sm_80','oracle':'CUDA12.8 ptxas and nvdisasm, matching PTX copy forms',
        'policies':policies,'field':{'low_bit':71,'width':2,'values':{'0':0,'64':1,'128':2,'256':3}},
        'all_policy_field_roundtrips':list(zip([f'{w:032x}' for w in allfield],fieldtexts)),
        'random_operand_roundtrips':roundtrips,
        'scope':'valid instruction encoding and PTX acceptance; cache behavior and speed require hardware'}
    (HERE/'cache_encoding_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps({'policies':policies,'roundtrips':len(roundtrips)},indent=2))
    return evidence


if __name__=='__main__':discover()
