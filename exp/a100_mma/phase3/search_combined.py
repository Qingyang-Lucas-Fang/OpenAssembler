"""Compose GPU-proven B-fragment pairing with the native row epilogue.

Reuses the epilogue agent's scalar/row reference builds. Creates only new
search_combined* artifacts and performs no GPU execution.
"""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import sys
import time

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
sys.path.insert(0,str(HERE))
import search_sequences as search
import epilogue
from openassembler import build,oracle
from openassembler.cli import main
from openassembler.isa.word import control,control_valid


class CombinedSelector(search.HalfBSelector,epilogue.NativePairSelector):
    """B half-quad tensor reads plus accumulator half-quad output stores."""


@contextmanager
def selector_and_allocator(row=True,role_bias=0):
    original=search.backend.TensorSelector
    original_alloc=search.ra._alloc
    search.backend.TensorSelector=CombinedSelector if row else search.HalfBSelector
    if role_bias:
        search.ra._alloc=lambda ops,r1,hints,max_reg:search.biased_alloc(
            ops,r1,hints,max_reg,role_bias=role_bias)
    try:
        yield
    finally:
        search.backend.TensorSelector=original
        search.ra._alloc=original_alloc


def source_for(m,n,k,dtype,*,row=True,**geometry):
    source=search.make_source(m,n,k,dtype,paired_b=True,**geometry)
    if row:
        source=epilogue.vectorize_epilogue(source,n=n,wm=geometry.get('wm',64),
                                          wn=geometry.get('wn',64),address_mode='row')
    return source


def compile_case(tag,m,n,k,dtype,*,row=True,role_bias=0,gap=8):
    name=f'search_combined_{tag}_{dtype}'
    search.configure(gap=gap,reuse=True)
    source=source_for(m,n,k,dtype,row=row)
    path=HERE/f'{name}.ptx';path.write_text(source)
    output=BASE/'build'/name
    search.backend.STAGE_RECORDS.clear()
    captured=[];old_generate=build.generate
    def capture(kernel,arch,opt_level):
        result=old_generate(kernel,arch,opt_level);captured.append(result);return result
    build.generate=capture
    started=time.perf_counter()
    try:
        with selector_and_allocator(row,role_bias):
            rc=main(['compile',str(path),'--out-dir',str(output),'--opt-level','safe'])
    finally:
        build.generate=old_generate
    if rc:
        raise RuntimeError(f'{name}: compile rc={rc}')
    manifest=json.loads((output/'manifest.json').read_text());entry=manifest['entries'][0]
    records=oracle.parse_listing((output/entry['sass']).read_text())
    if len(captured)!=1:
        raise AssertionError('expected one OA kernel')
    code,labels,words,exits,regs=captured[0]
    assert [r.word for r in records[:len(words)]]==words
    norm=lambda text:re.sub(r'\s+','',text.replace('.reuse',''))
    for (want,ctl),actual in zip(code,records):
        assert control_valid(control(actual.word)) and ctl==control(actual.word)
        if want.startswith(('HMMA','LDSM','LDGSTS','LDGDEPBAR','DEPBAR','BAR','STG')):
            assert norm(want)==norm(actual.text),(want,actual.text)
    sources=[Path(__file__),Path(search.__file__),Path(epilogue.__file__),
             BASE/'backend.py',BASE/'pipeline.py',BASE.parents[1]/'openassembler/backend/ra.py',
             BASE.parents[1]/'openassembler/isa/encode.py']
    metadata=dict(compiler='OpenAssembler isolated B-LDSM/row-epilogue composition',fallback=False,
                  reference_and_ELF_container='NVIDIA ptxas on matching PTX',
                  candidate=dict(paired_b=True,paired_preaddress=False,row=row,
                                 role_bias=role_bias,gap=gap,reuse=True),
                  geometry=dict(m=m,n=n,k=k,bm=128,bn=128,bk=64,wm=64,wn=64,stages=2),
                  source_ptx_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  compiler_source_hashes={str(p.relative_to(BASE.parents[1])):hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in sources},
                  cli_wall_ms=(time.perf_counter()-started)*1000,
                  kernels=search.backend.STAGE_RECORDS.copy(),
                  sass_opcode_counts=dict(Counter(r.text.split()[0] for r in records)),
                  nvdisasm_exact_tensor_memory_store_operands=True,
                  spliced_words_and_control_exact=True)
    (output/'experiment.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return dict(build=name,name=name,m=m,n=n,k=k,dtype=dtype,
                tile_m=128,tile_n=128,threads=128,smem=65536)


def cli():
    parser=argparse.ArgumentParser()
    parser.add_argument('--references',default='epilogue-confirm-suite.json')
    args=parser.parse_args();search.backend.install()
    references=json.loads((HERE/args.references).read_text())
    suite=[];failures=[]
    for reference in references:
        # Deep copy prevents accidental mutation of the retained reference file.
        root=json.loads(json.dumps(reference));alternatives=root.pop('alternatives')
        if len(alternatives)!=1:
            raise ValueError('expected scalar baseline plus one row epilogue reference')
        row_reference=alternatives[0]
        if 'row_native' not in row_reference['build']:
            raise ValueError('expected a native row epilogue reference')
        dtype=root['dtype'];m,n,k=(root[x] for x in ('m','n','k'))
        shape=next(tag for tag,dims in [('square',(4096,4096,4096)),
                    ('mlp',(4096,16384,4096)),('tall',(16384,4096,4096))] if dims==(m,n,k))
        candidates=[row_reference]
        for variant,opts in [('bpair',dict(row=False)),('row_bpair',{}),
                             ('row_bpair_role1',dict(role_bias=1)),
                             ('row_bpair_gap4',dict(gap=4))]:
            try:
                candidates.append(compile_case(f'{shape}_{variant}',m,n,k,dtype,**opts))
            except Exception as exc:
                failures.append(dict(shape=shape,dtype=dtype,variant=variant,error=repr(exc)))
        root['name']=f'search_combined_{shape}_{dtype}'
        root['steady_warmup_rounds']=16;root['alternatives']=candidates;suite.append(root)
    (HERE/'search_combined_suite.json').write_text(json.dumps(suite,indent=2)+'\n')
    (HERE/'search_combined_failures.json').write_text(json.dumps(failures,indent=2)+'\n')
    print(json.dumps(dict(cases=len(suite),builds=sum(1+len(r['alternatives']) for r in suite),
                          failures=failures),indent=2))


if __name__=='__main__':
    cli()
