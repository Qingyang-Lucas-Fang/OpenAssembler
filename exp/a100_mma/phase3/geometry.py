"""Independent tile/stage experiments informed by observed cuBLAS resources.

The existing generator is extended in memory only to five stages. The fifth
stage requires wait_group3: its DEPBAR threshold is round-tripped before any
kernel is compiled. No production or phase2 source is changed.
"""
import argparse
from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path
import re
import sys
import time

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
sys.path[:0]=[str(HERE),str(BASE)]
import backend
import pipeline
import search_sequences as search
from epilogue import vectorize_epilogue
from openassembler.cli import main
from openassembler.ptx.parse import ImmI
from openassembler.isa.word import set_bits
from openassembler import oracle

source=inspect.getsource(pipeline.pipelined_gemm)
assert source.count('stages not in (2,3,4)')==1
namespace={}
exec(source.replace('stages not in (2,3,4)','stages not in (2,3,4,5)'),namespace)
five_stage_generator=namespace['pipelined_gemm']

class Stage5Selector(search.HalfBSelector):
    def async_copy(self,ins):
        if (ins.guard is None and ins.mods==('async','wait_group') and len(ins.args)==1
                and isinstance(ins.args[0],ImmI) and ins.args[0].value==3):
            self.emit('DEPBAR.LE SB0, 0x3',side=True)
            return
        return super().async_copy(ins)

class Stage5Encoder:
    def __init__(self,base):
        self.base=base
        self.word=set_bits(base.encode('DEPBAR.LE SB0, 0x0'),38,6,3)
        if oracle.disasm_words([self.word])!=['DEPBAR.LE SB0, 0x3']:
            raise AssertionError('wait_group3 encoding failed disassembly roundtrip')
    def __getattr__(self,name):
        return getattr(self.base,name)
    def encode(self,text,ctl=None):
        if text=='DEPBAR.LE SB0, 0x3':
            return self.base._ctl(self.word,ctl) if ctl is not None else self.word
        return self.base.encode(text,ctl)

@contextmanager
def five_stage_support():
    old_selector,old_encoder=backend.TensorSelector,backend._ENCODER
    backend._ENCODER=Stage5Encoder(backend.encoder())
    backend.TensorSelector=Stage5Selector
    try:
        yield
    finally:
        backend.TensorSelector,backend._ENCODER=old_selector,old_encoder

CONFIGS=[
    ('base',128,128,64,2),
    ('wide64s2',128,256,64,2),
    ('tall64s2',256,128,64,2),
    ('wide32s3',128,256,32,3),
    ('wide32s4',128,256,32,4),
    ('wide32s5',128,256,32,5),
    ('square32s5',128,128,32,5),
]

def make_source(m,n,k,dtype,bm,bn,bk,stages):
    wm=wn=64
    source=five_stage_generator(m,n,k,dtype,bm,bn,bk,wm,wn,compact=True,stages=stages)
    pattern=r'add.u32 %r41,%r200,%r9;\n.*?(?=\nbar.sync 0;)'
    blocks=re.findall(pattern,source,flags=re.S)
    if len(blocks)!=stages or any(block!=blocks[0] for block in blocks):
        raise AssertionError('compute blocks differ or stage count changed')
    replacement=search.compute_sequence(dtype,bk=bk,wm=wm,wn=wn,paired_b=True,column=True)
    source=re.sub(pattern,lambda _:replacement,source,flags=re.S)
    return vectorize_epilogue(source,n=n,wm=wm,wn=wn,address_mode='row')

def compile_one(tag,m,n,k,dtype,config):
    label,bm,bn,bk,stages=config
    search.configure()
    source=make_source(m,n,k,dtype,bm,bn,bk,stages)
    ptx=HERE/f'geometry_{tag}_{dtype}_{label}.ptx'
    ptx.write_text(source)
    output=BASE/'build'/ptx.stem
    backend.STAGE_RECORDS.clear()
    started=time.perf_counter()
    with five_stage_support():
        rc=main(['compile',str(ptx),'--out-dir',str(output),'--opt-level','safe'])
    if rc:raise RuntimeError(f'compile failed: {ptx.stem}')
    manifest=json.loads((output/'manifest.json').read_text())
    records=oracle.parse_listing((output/manifest['entries'][0]['sass']).read_text())
    from collections import Counter
    metadata=dict(compiler='OpenAssembler isolated geometry experiment',fallback=False,
                  reference_and_ELF_container='NVIDIA ptxas on matching PTX',
                  geometry=dict(m=m,n=n,k=k,bm=bm,bn=bn,bk=bk,wm=64,wn=64,stages=stages),
                  paired_b=True,row_pair_epilogue=True,nvdisasm_valid=bool(records),
                  source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  experiment_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  kernels=backend.STAGE_RECORDS.copy(),
                  sass_opcode_counts=dict(Counter(r.text.split()[0] for r in records)),
                  cli_wall_ms=(time.perf_counter()-started)*1000)
    (output/'experiment.json').write_text(json.dumps(metadata,indent=2)+'\n')
    smem=2*stages*bk*(bm+bn)
    return dict(build=output.name,name=output.name,m=m,n=n,k=k,dtype=dtype,
                tile_m=bm,tile_n=bn,threads=bm//64*(bn//64)*32,smem=smem if smem>49152 else 0)

def cli():
    p=argparse.ArgumentParser();p.add_argument('--large',action='store_true');a=p.parse_args()
    backend.install()
    tag='large' if a.large else 'smoke'
    m,n,k=(4096,4096,4096) if a.large else (512,512,320)
    suite=[];failures=[]
    for dtype in ('f16','bf16'):
        cases=[]
        for config in CONFIGS:
            try:cases.append(compile_one(tag,m,n,k,dtype,config))
            except Exception as exc:
                failures.append(dict(dtype=dtype,config=config,error=repr(exc)))
        if cases:
            root=cases[0];root['name']=f'geometry_{tag}_{dtype}'
            root['alternatives']=cases[1:];root['steady_warmup_rounds']=16;suite.append(root)
    (HERE/f'geometry_{tag}_suite.json').write_text(json.dumps(suite,indent=2)+'\n')
    (HERE/f'geometry_{tag}_failures.json').write_text(json.dumps(failures,indent=2)+'\n')
    print('failures',failures)

if __name__=='__main__':cli()
