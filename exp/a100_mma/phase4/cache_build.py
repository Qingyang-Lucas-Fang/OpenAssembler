"""Apply a proven two-bit copy-cache postpass to native stage-pair cubins.

The selected native instructions, operands, code size, resource metadata, and
all scheduling controls remain exact. Matching hinted PTX is independently
assembled by ptxas as an oracle, never substituted for the native candidate.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import re
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from openassembler import oracle
from openassembler.container.cubin import text_words,words_to_bytes
from openassembler.container.elf import Elf
from openassembler.isa.word import control,control_valid,set_bits


def patch_binary(binary,hint,evidence):
    record=evidence['policies'][str(hint)]
    if not record['supported']:raise ValueError(f'unsupported sm80hint {hint}')
    field=evidence['field']['values'][str(hint)]
    words=text_words(binary,'gemm')
    # Decode the complete text section rather than matching arbitrary byte
    # patterns. The full-section listing is provided by the caller's cubin.
    native=Elf.parse(binary)
    return native,words,field


def compile_variant(baseline,hint,evidence):
    record=evidence['policies'][str(hint)]
    source_root=(BASE/'build'/baseline['build']).resolve()
    entry=json.loads((source_root/'manifest.json').read_text())['entries'][0]
    metadata=json.loads((source_root/'experiment.json').read_text())
    if not metadata['candidate']['stage_pair'] or not metadata['candidate']['shared_chunk_immediates']:
        raise ValueError('require validated stage-pair reference')
    baseline_source=(source_root/entry['ptx']).read_text()
    if hashlib.sha256(baseline_source.encode()).hexdigest()!=metadata['source_ptx_sha256']:
        raise ValueError('sourcehash does not match baseline metadata')
    name=f'cache_l2_{hint}_{baseline["build"].removeprefix("address_stagepair_")}'
    directory=HERE/'build'/name
    directory.mkdir(parents=True,exist_ok=True)
    ptx=directory/'gemm.ptx'
    source=baseline_source.replace('cp.async.cg.shared.global ',f'cp.async.cg.shared.global.L2::{hint}B ')
    if source==baseline_source:raise AssertionError('no asynccopies transformed')
    ptx.write_text(source)
    reference=directory/'reference.cubin';oracle.ptxas(ptx,reference)
    oracle.SCRATCH=HERE/'scratch'
    binary=(source_root/entry['cubin']).read_bytes()
    elf,words,field=patch_binary(binary,hint,evidence)
    old_listing=(source_root/entry['sass']).read_text()
    instructions=oracle.parse_listing(old_listing)
    copies=[i for i,x in enumerate(instructions) if x.text.startswith('LDGSTS.')]
    if len(copies)!=64:raise AssertionError(f'expected64staticstagepaircopies;got{len(copies)}')
    target=[]
    for index in copies:
        instruction=instructions[index]
        if instruction.word!=words[index] or not instruction.text.startswith('LDGSTS.E.BYPASS.128 '):
            raise ValueError('unproven original async instruction')
        word=set_bits(instruction.word,71,2,field)
        if (word^instruction.word)!=int(record['nonoperand_noncontrol_xor'],16):
            raise AssertionError('candidate changes bits beyond discovered policy field')
        if control(word)!=control(instruction.word) or not control_valid(control(word)):
            raise AssertionError('control changed')
        words[index]=word
        target.append({'instruction_index':index,'address':instruction.addr,
            'old_word':f'{instruction.word:032x}','new_word':f'{word:032x}',
            'changed_bits':record['changed_bits'],'old_text':instruction.text})
    elf.replace_data('.text.gemm',words_to_bytes(words))
    patched=elf.serialize();path=directory/'gemm.cubin';path.write_bytes(patched)
    expected_bytes={elf.section('.text.gemm').offset+index*16+bit//8
                    for index in copies for bit in record['changed_bits']}
    changed_bytes={index for index,(old,new) in enumerate(zip(binary,patched)) if old!=new}
    if len(binary)!=len(patched) or changed_bytes!=expected_bytes:
        raise AssertionError('complete cubin changed outside exact discovered hint-bit bytes')
    before=Elf.parse(binary);after=Elf.parse(patched)
    for section in before.sections:
        if section.name!='.text.gemm' and after.section(section.name).data!=section.data:
            raise AssertionError(f'nontextsectionchanged:{section.name}')
    actual_words=text_words(patched,'gemm')
    if actual_words!=words:raise AssertionError('serialized text mismatch')
    new_listing=oracle.disasm_text(path);(directory/'gemm.sass').write_text(new_listing)
    actual=oracle.parse_listing(new_listing)
    if len(actual)!=len(instructions):raise AssertionError('code length changed')
    for i,(old,new) in enumerate(zip(instructions,actual)):
        if i in copies:
            expected=old.text.replace('LDGSTS.E.BYPASS.128',f'LDGSTS.E.BYPASS.LTC{hint}B.128')
            if new.text!=expected:raise AssertionError((expected,new.text))
        elif old.word!=new.word or old.text!=new.text:
            raise AssertionError('noncopyinstructionchanged')
    provenance={'compiler':'native OpenAssembler stage-pair plus isolated validated L2-prefetch bit pass',
        'fallback':False,'candidate':{'l2_prefetch_bytes':hint,'stage_pair':True,'paired_b':True,'row_epilogue':True},
        'geometry':metadata['geometry'],'kernels':metadata['kernels'],
        'baseline_build':baseline['build'],'baseline_source_sha256':hashlib.sha256(baseline_source.encode()).hexdigest(),
        'baseline_cubin_sha256':hashlib.sha256(binary).hexdigest(),
        'source_ptx_sha256':hashlib.sha256(source.encode()).hexdigest(),
        'native_cubin_sha256':hashlib.sha256(patched).hexdigest(),
        'ptxas_reference_sha256':hashlib.sha256(reference.read_bytes()).hexdigest(),
        'evidence_sha256':hashlib.sha256((HERE/'cache_encoding_evidence.json').read_bytes()).hexdigest(),
        'postpass_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'postpass_changed_only_bits':record['changed_bits'],'copy_instructions_changed':target,
        'all_controls_and_noncopy_words_exact':True,'all_nontext_section_data_exact':True,
        'complete_cubin_diff_only_expected_hint_bytes':True,
        'valid_nvdisasm':True,'reference_is_matching_hinted_ptx':True,
        'sass_opcode_counts':dict(Counter(x.text.split()[0] for x in actual))}
    (directory/'experiment.json').write_text(json.dumps(provenance,indent=2)+'\n')
    new_entry=dict(entry,source='gemm.ptx',source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                   ptx='gemm.ptx',cubin='gemm.cubin',reference='reference.cubin',sass='gemm.sass')
    manifest={'schema':1,'arch':'sm_80','root':str(directory),'entries':[new_entry]}
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    result={key:value for key,value in baseline.items() if key not in ('alternatives','steady_warmup_rounds')}
    result.update(build=f'../phase4/build/{name}',name=name)
    print(name,provenance['postpass_changed_only_bits'],len(target),flush=True)
    return result


def main():
    evidence=json.loads((HERE/'cache_encoding_evidence.json').read_text())
    rows=json.loads((BASE/'phase3/address-stage-confirm-suite.json').read_text())
    suite=[]
    for row in rows:
        baseline=dict(row['alternatives'][0]);baseline['alternatives']=[]
        baseline['name']=f'cache_compare_{baseline["build"].removeprefix("address_stagepair_")}'
        baseline['steady_warmup_rounds']=16
        for hint in ((64,128,256) if row['m']==row['n'] else (128,)):
            baseline['alternatives'].append(compile_variant(row['alternatives'][0],hint,evidence))
        suite.append(baseline)
    (HERE/'cache_suite.json').write_text(json.dumps(suite,indent=2)+'\n')
    print('cases',len(suite),'new variants',sum(len(x['alternatives']) for x in suite))


if __name__=='__main__':main()
