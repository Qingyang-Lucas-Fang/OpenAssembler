"""Bound paired-B fragment lifetimes in the validated stage-pair generator."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import re
import shutil
import sys
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
PHASE3=BASE/'phase3'
ROOT=HERE.parents[2]
sys.path.insert(0,str(PHASE3))
import address_shared as address
from address_stage import specialize_two_stage
from openassembler import build,oracle
from openassembler.cli import main
from openassembler.isa import encode
from openassembler.isa.word import control,control_valid


def compute(dtype='f16',*,bk=64,wm=64,wn=64,window=2,prefetch=False):
    if window not in (2,4) or (prefetch and window!=4):raise ValueError('bounded windows2/4, prefetchonly4')
    ma,nb=wm//16,wn//8
    if nb%window or ma!=4:raise ValueError('window must cover full paired N fragments; retain4Arows')
    lines=['shr.u32 %r47,%r3,4;',f'mul.lo.u32 %r47,%r47,{16*bk};']
    def load(pair,slot):
        regs=','.join(f'%r{500+slot*4+j}' for j in range(4))
        return f'ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{regs}}},[%r42+{pair*16*bk}];'
    def mma(mi,ni,slot):
        a=','.join(f'%r{400+mi*4+j}' for j in range(4))
        b=','.join(f'%r{500+slot*4+(ni%2)*2+j}' for j in range(2))
        c=','.join(f'%c{(mi*nb+ni)*4+j}' for j in range(4))
        return f'mma.sync.aligned.m16n8k16.row.col.f32.{dtype}.{dtype}.f32 {{{c}}},{{{a}}},{{{b}}},{{{c}}};'
    for ik in range(bk//16):
        lines += [f'add.u32 %r41,%r{200+ik*ma},%r9;',
                  f'add.u32 %r42,%r{300+ik*nb},%r9;',
                  'add.u32 %r42,%r42,%r47;']
        for mi in range(ma):
            regs=','.join(f'%r{400+mi*4+j}' for j in range(4))
            lines += [f'ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{regs}}},[%r41+{mi*32*bk}];']
        if prefetch:
            lines += [load(0,0),load(2,1)]
            for pair in range(0,nb,2):
                slot=(pair//2)%2
                lines += [mma(mi,ni,slot) for ni in (pair,pair+1) for mi in range(ma)]
                # Both halves of this enclosing quad have no remaining readers
                # in this K slice. The next pair reuses exactly that slot; the
                # other quad remains live and supplies the next compute epoch.
                if pair+4<nb:lines += [load(pair+4,slot)]
        else:
            for first in range(0,nb,window):
                lines += [load(pair,(pair-first)//2) for pair in range(first,first+window,2)]
                lines += [mma(mi,ni,(ni-first)//2)
                          for ni in range(first,first+window) for mi in range(ma)]
    return '\n'.join(lines)


def make_source(m,n,k,dtype,*,window=2,prefetch=False,bm=128,bn=128,bk=64,wm=64,wn=64):
    mask=bk//8-1
    source=address.make_source(m,n,k,dtype,bm=bm,bn=bn,bk=bk,wm=wm,wn=wn,
                               swizzle_a=mask,swizzle_b=mask)
    old=address.search.compute_sequence(dtype,bk=bk,wm=wm,wn=wn,paired_b=True)
    if source.count(old)!=2:raise ValueError('expected2identicalcomputeepochs before stagespecialization')
    new=compute(dtype,bk=bk,wm=wm,wn=wn,window=window,prefetch=prefetch)
    source=source.replace(old,new)
    return specialize_two_stage(source,k=k,bm=bm,bn=bn,bk=bk)


def initialize():
    # Redirect experimental cache/discovery writes before instantiating any
    # encoder. Read old evidence and templates; never alter phase3 or corefiles.
    backend=address.search.backend
    backend.HERE=HERE
    HERE.mkdir(exist_ok=True);(HERE/'scratch').mkdir(exist_ok=True)
    for name in ('fieldmaps.json','templates.json'):
        destination=HERE/name
        if not destination.exists() and (BASE/name).exists():shutil.copyfile(BASE/name,destination)
    backend.install();address.search.configure(gap=8,reuse=True)
    return backend


def compile_case(dtype,window,prefetch=False,m=4096,n=4096,k=4096):
    backend=initialize()
    name=f'fragment_w{window}{"_prefetch" if prefetch else ""}_{dtype}'
    source=make_source(m,n,k,dtype,window=window,prefetch=prefetch)
    source_path=HERE/f'{name}.ptx';source_path.write_text(source)
    directory=HERE/'build'/name
    captured=[];old=build.generate
    def capture(*args):
        result=old(*args);captured.append(result);return result
    build.generate=capture;backend.STAGE_RECORDS.clear();start=time.perf_counter()
    try:
        with address.address_backend():
            rc=main(['compile',str(source_path),'--out-dir',str(directory),'--opt-level','safe'])
    finally:build.generate=old
    if rc or len(captured)!=1:raise RuntimeError(f'{name}compilefailed:{rc}')
    entry=json.loads((directory/'manifest.json').read_text())['entries'][0]
    records=oracle.parse_listing((directory/entry['sass']).read_text())
    code,labels,words,exits,regs=captured[0]
    if [x.word for x in records[:len(words)]]!=words:raise AssertionError('wordsplicechanged')
    norm=lambda s:re.sub(r'\s+','',s.replace('.reuse','').replace('+-','-'))
    for (text,ctl),actual in zip(code,records):
        if ctl!=control(actual.word) or not control_valid(ctl):raise AssertionError('controlroundtripfailed')
        if text.startswith(('LDGSTS','LDSM','HMMA','BAR','DEPBAR','LDGDEPBAR','STG')) and norm(text)!=norm(actual.text):
            raise AssertionError((text,actual.text))
    counts=Counter(x.text.split()[0] for x in records)
    hmma=sum(v for key,v in counts.items() if key.startswith('HMMA'))
    if hmma!=512 or counts['LDGSTS.E.BYPASS.128']!=64 or counts['LDSM.16.M88.4']!=128:
        raise AssertionError(('instructioncountchanged',hmma,counts))
    metadata={'compiler':'OpenAssembler isolated paired-B fragment lifetime experiment','fallback':False,
        'candidate':{'window_columns':window,'prefetch_next_pair':prefetch,'retained_a_quads':4,
                     'max_b_quads':window//2,'stage_pair':True,'paired_b':True,'row_epilogue':True},
        'geometry':{'m':m,'n':n,'k':k,'bm':128,'bn':128,'bk':64,'wm':64,'wn':64,'stages':2},
        'kernels':backend.STAGE_RECORDS.copy(),'source_ptx_sha256':hashlib.sha256(source.encode()).hexdigest(),
        'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'nvdisasm_words_controls_tensor_memory_exact':True,'sass_opcode_counts':dict(counts),
        'hmma_per_static_full_stage_pair':hmma,'async_copy_count':64,'ldmatrix_x4_count':128,
        'timing_controls':'source order HMMAgap8+epochdrain15; LDSMaddressWARretained; no readdependency deletion',
        'cli_wall_ms':(time.perf_counter()-start)*1000}
    (directory/'experiment.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(name,'regs',regs,'instructions',len(words),flush=True)
    return {'name':name,'build':f'../phase4/build/{name}','m':m,'n':n,'k':k,'dtype':dtype,
            'tile_m':128,'tile_n':128,'threads':128,'smem':65536}


if __name__=='__main__':
    suite=[]
    for dtype in ('f16','bf16'):
        baseline={'name':f'fragment_compare_{dtype}','build':f'address_stagepair_square_{dtype}',
                  'm':4096,'n':4096,'k':4096,'dtype':dtype,'tile_m':128,'tile_n':128,'threads':128,'smem':65536,
                  'steady_warmup_rounds':16,'alternatives':[]}
        for window,prefetch in ((2,False),(4,False),(4,True)):
            baseline['alternatives'].append(compile_case(dtype,window,prefetch))
        suite.append(baseline)
    (HERE/'fragment_suite.json').write_text(json.dumps(suite,indent=2)+'\n')
