"""Rebuild OA words in memory and audit retained cubin disassembly exactly."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import search_sequences as search
from openassembler import oracle
from openassembler.isa.word import control,control_valid
from openassembler.ptx.parse import parse_module


def audit(config):
    directory=search.BASE/'build'/config['build']
    metadata=json.loads((directory/'experiment.json').read_text())
    manifest=json.loads((directory/'manifest.json').read_text())
    entry=manifest['entries'][0]
    source=(directory/entry['ptx']).read_text()
    opts=metadata['candidate']
    search.configure(opts['gap'],opts['reuse'])
    with search.compiler_variant(paired_b=opts['paired_b'],role_bias=opts['role_bias'],
                                 permute=opts['permute']):
        code,labels,words,exits,regs=search.backend.generate(
            parse_module(source).kernels['gemm'],'sm_80','safe')
    records=oracle.parse_listing((directory/entry['sass']).read_text())
    assert [r.word for r in records[:len(words)]]==words,config['build']
    for (text,ctl),record in zip(code,records):
        assert control_valid(control(record.word)) and ctl==control(record.word)
        if text.startswith(('HMMA','LDSM','LDGSTS','LDGDEPBAR','DEPBAR','BAR')):
            norm=lambda value:re.sub(r'\s+','',value.replace('.reuse',''))
            assert norm(text)==norm(record.text),(text,record.text)
    return dict(build=config['build'],exact_words=len(words),
                exact_controls=len(code),exact_tensor_memory_operands=True,registers=regs)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('suite')
    args=parser.parse_args();search.backend.install()
    results=[]
    for root in json.loads((search.HERE/args.suite).read_text()):
        for config in [root,*root.get('alternatives',[])]:
            result=audit(config);results.append(result);print(config['build'],flush=True)
    output=search.HERE/(Path(args.suite).stem+'_audit.json')
    output.write_text(json.dumps(results,indent=2)+'\n')
    print(output)
