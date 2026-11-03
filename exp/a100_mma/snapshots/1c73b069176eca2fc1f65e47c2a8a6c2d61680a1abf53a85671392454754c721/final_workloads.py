"""Training projection shapes and repeated square runs for the passing compiler."""
import argparse
import json
from backend import HERE, install
import backend
from build_pipeline import compile_one

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--tiles',action='store_true')
    args=parser.parse_args()
    install();suite=[];failed=[]
    backend.TENSOR_ISSUE_GAP=8
    backend.LDSM_READ_BARRIER=False
    backend.LIST_SCHEDULE=False
    if args.tiles:
        configs=[('wide_tile',4096,4096,4096,'f16',128,256),
                 ('tall_tile',4096,4096,4096,'f16',256,128)]
        target='large-tile-suite.json'
    else:
        configs=[(f'validated_{tag}_{dtype}',m,n,4096,dtype,128,128)
                 for dtype in ('f16','bf16')
                 for tag,m,n in (('square',4096,4096),('mlp',4096,16384),('tall',16384,4096))]
        target='validation-suite.json'
    for name,m,n,k,dtype,bm,bn in configs:
        try:
            config=compile_one(name,m,n,k,dtype,bm,bn,64,64,64,compact=True)
            if name=='validated_square_bf16':config['triton_sweep']=True
            suite.append(config)
        except RuntimeError as exc:
            failed.append({'name':name,'compile_error':str(exc)})
    (HERE/target).write_text(json.dumps(suite,indent=2)+'\n')
    (HERE/f'{target}.failures.json').write_text(json.dumps(failed,indent=2)+'\n')
