"""Two/three/four-stage candidates using the existing isolated compiler."""
import argparse,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
import backend
from build_pipeline import compile_one

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--large',action='store_true');args=parser.parse_args()
    backend.install();backend.TENSOR_ISSUE_GAP=8;backend.LDSM_READ_BARRIER=False
    suite=[];failed=[]
    for stages in (2,3,4):
        for bk,wm,wn in ((32,64,32),(64,64,64)):
            name=f'p2_s{stages}_k{bk}_w{wm}x{wn}_{"large" if args.large else "smoke"}'
            m,n,k=(4096,4096,4096) if args.large else (256,384,512)
            try:suite.append(compile_one(name,m,n,k,'f16',128,128,bk,wm,wn,compact=True,stages=stages))
            except RuntimeError as exc:failed.append({'name':name,'compile_error':str(exc)})
    tag='large' if args.large else 'smoke'
    (BASE/f'phase2/multistage-{tag}-suite.json').write_text(json.dumps(suite,indent=2)+'\n')
    (BASE/f'phase2/multistage-{tag}-compile-failures.json').write_text(json.dumps(failed,indent=2)+'\n')
