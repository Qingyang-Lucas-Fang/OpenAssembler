"""Independent retained-evidence audit for the two-bit L2-prefetch experiment."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from openassembler.container.cubin import text_words
from openassembler.container.elf import Elf
from openassembler.isa.word import control


def audit():
    evidence=json.loads((HERE/'cache_encoding_evidence.json').read_text())
    semantics=json.loads((HERE/'cache_copy_semantics.json').read_text())
    if not all(x['output_exact'] and x['all512_shared_copy_bytes_exact'] for x in semantics):
        raise AssertionError('functionalcheckfailed')
    suites=json.loads((HERE/'cache_suite.json').read_text())
    rows=[]
    for baseline in suites:
        for variant in baseline['alternatives']:
            directory=(BASE/'build'/variant['build']).resolve()
            meta=json.loads((directory/'experiment.json').read_text())
            oldroot=BASE/'build'/baseline['build']
            entry=json.loads((oldroot/'manifest.json').read_text())['entries'][0]
            old=(oldroot/entry['cubin']).read_bytes();new=(directory/'gemm.cubin').read_bytes()
            if hashlib.sha256(old).hexdigest()!=meta['baseline_cubin_sha256'] or hashlib.sha256(new).hexdigest()!=meta['native_cubin_sha256']:
                raise AssertionError('binaryhashmismatch')
            oldwords=text_words(old,'gemm');newwords=text_words(new,'gemm')
            hint=meta['candidate']['l2_prefetch_bytes']
            delta=int(evidence['policies'][str(hint)]['nonoperand_noncontrol_xor'],16)
            changed=[i for i,(x,y) in enumerate(zip(oldwords,newwords)) if x!=y]
            if len(oldwords)!=len(newwords) or len(changed)!=64:raise AssertionError('wrongcopycount')
            if any(oldwords[i]^newwords[i]!=delta or control(oldwords[i])!=control(newwords[i]) for i in changed):
                raise AssertionError('changedunprovenbits')
            before=Elf.parse(old);after=Elf.parse(new)
            allowed={before.section('.text.gemm').offset+index*16+bit//8
                     for index in changed for bit in meta['postpass_changed_only_bits']}
            changedbytes={i for i,(x,y) in enumerate(zip(old,new)) if x!=y}
            if len(old)!=len(new) or changedbytes!=allowed:raise AssertionError('unexpectedwholebinarydiff')
            rows.append({'name':variant['name'],'hint_bytes':hint,'copies':64,
                         'changed_bits':meta['postpass_changed_only_bits'],
                         'full_cubin_only_expected_bytes_changed':True,
                         'all_schedule_controls_exact':True,'register_count_unchanged':True,
                         'native_cubin_sha256':hashlib.sha256(new).hexdigest()})
    result={'passed':True,'cases':len(suites),'new_variants':len(rows),
            'random_operand_roundtrips':len(evidence['random_operand_roundtrips']),
            'functional_copy_checks':len(semantics),'variants':rows,
            'scope':'static/probe functional verification; whole-GEMM GPU correctness and performance pending'}
    (HERE/'cache_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='variants'},indent=2))


if __name__=='__main__':audit()
