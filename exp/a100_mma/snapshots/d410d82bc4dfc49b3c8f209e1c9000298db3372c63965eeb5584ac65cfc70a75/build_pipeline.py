"""Compile bounded full-tile two-stage candidates, retaining provenance."""
import json
import time
import hashlib
from backend import HERE, install
import backend
from pipeline import pipelined_gemm
from openassembler.cli import main


def compile_one(name,m,n,k,dtype,bm,bn,bk,wm,wn,compact=False,stages=2,b_window=0):
    source=HERE/'ptx'/f'{name}.ptx'
    source.write_text(pipelined_gemm(m,n,k,dtype,bm,bn,bk,wm,wn,compact=compact,stages=stages,b_window=b_window))
    output=HERE/'build'/name
    backend.OPTIMIZE_LOCALS=True
    backend.HOIST_LOOP_INVARIANTS=False
    backend.MINED_SCALAR=True
    backend.RELAX_TENSOR=True
    backend.STAGE_RECORDS.clear()
    start=time.perf_counter()
    rc=main(['compile',str(source),'--out-dir',str(output),'--opt-level','safe'])
    if rc: raise RuntimeError(name)
    sources={p.name:p.read_bytes() for p in HERE.glob('*.py')}
    digests={name:hashlib.sha256(data).hexdigest() for name,data in sources.items()}
    snapshot_hash=hashlib.sha256(json.dumps(digests,sort_keys=True).encode()).hexdigest()
    snapshot=HERE/'snapshots'/snapshot_hash
    snapshot.mkdir(parents=True,exist_ok=True)
    for filename,data in sources.items():
        if not (snapshot/filename).exists():(snapshot/filename).write_bytes(data)
    (output/'experiment.json').write_text(json.dumps({
        'compiler':'OpenAssembler experimental MMA selector','fallback':False,
        'reference_and_ELF_container':'NVIDIA ptxas','cli_rc':rc,
        'cli_wall_ms':(time.perf_counter()-start)*1000,
        'compiler_source_hashes':digests,'compiler_source_snapshot':str(snapshot.relative_to(HERE)),
        'geometry':{'m':m,'n':n,'k':k,'bm':bm,'bn':bn,'bk':bk,'wm':wm,'wn':wn,'compact_addresses':compact,'stages':stages,'b_window':b_window},
        'kernels':backend.STAGE_RECORDS.copy()},indent=2)+'\n')
    return dict(build=name,m=m,n=n,k=k,dtype=dtype,tile_m=bm,tile_n=bn,threads=bm//wm*(bn//wn)*32,
                smem=2*stages*bk*(bm+bn) if 2*stages*bk*(bm+bn)>49152 else 0)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--large',action='store_true')
    args=parser.parse_args()
    install()
    configs=[]
    if args.large:
        for issue in (4,8):
            backend.TENSOR_ISSUE_GAP=issue
            for bm,bn,bk,wm,wn in ((64,64,32,32,32),(128,128,32,64,32),(128,128,32,64,64)):
                configs.append(compile_one(f'barfix_pipe_{bm}_{bn}_{bk}_{wm}_{wn}_i{issue}',
                                           4096,4096,4096,'f16',bm,bn,bk,wm,wn))
        target=HERE/'pipeline-large-suite.json'
    else:
        for dtype in ('f16','bf16'):
            for bm,bn,bk,wm,wn in ((64,64,32,32,32),(128,128,32,64,32),(128,128,32,64,64)):
                configs.append(compile_one(f'pipe_{bm}_{bn}_{bk}_{wm}_{wn}_{dtype}',
                                           256,384,256,dtype,bm,bn,bk,wm,wn))
        target=HERE/'pipeline-suite.json'
    target.write_text(json.dumps(configs,indent=2)+'\n')
