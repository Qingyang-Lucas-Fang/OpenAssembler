"""Validate production CLI builds against the retained research instruction stream."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
ROOT=BASE.parents[1]


def main():
    from openassembler.container.elf import Elf
    old=json.loads((BASE/'phase3/final_suite.json').read_text())
    suite=[];evidence=[]
    for item in old:
        dtype=item['dtype'];m,n,k=(item[x] for x in ('m','n','k'))
        shape='square' if m==n else ('mlp' if n>m else 'tall')
        name=f'p4_cli_{shape}_{dtype}'
        output=BASE/'build'/name
        argv=[sys.executable,'-m','openassembler','gemm','--m',str(m),'--n',str(n),
              '--k',str(k),'--dtype',dtype,'--out-dir',str(output),'-o','gemm']
        subprocess.run(argv,check=True,cwd=ROOT)
        manifest=json.loads((output/'manifest.json').read_text());entry=manifest['entries'][0]
        kernel=entry['kernels'][0]
        assert kernel['compiled_by']=='openassembler' and kernel['gemm']['schedule']=='stagepair'
        actual=(output/entry['cubin']).read_bytes()
        reference=item['alternatives'][-1]
        directory=BASE/'build'/reference['build']
        previous=json.loads((directory/'manifest.json').read_text())['entries'][0]
        expected=(directory/previous['cubin']).read_bytes()
        actual_text=Elf.parse(actual).section('.text.gemm').data
        expected_text=Elf.parse(expected).section('.text.gemm').data
        assert actual_text==expected_text,(name,'native code differs from the measured kernel')
        info=dict(build=name,argv=argv,native_cli=True,phase3_reference=reference['build'],
                  text_equal=True,text_sha256=hashlib.sha256(actual_text).hexdigest(),
                  cubin_sha256=hashlib.sha256(actual).hexdigest(),registers=kernel['registers'],
                  geometry=dict(m=m,n=n,k=k,bm=128,bn=128,bk=64,wm=64,wn=64,stages=2),
                  compiler='OpenAssembler production CLI SM80 tensor backend',fallback=False,
                  contract=kernel['gemm'])
        (output/'experiment.json').write_text(json.dumps(info,indent=2)+'\n')
        evidence.append(info)
        case=dict(build=name,name=name,m=m,n=n,k=k,dtype=dtype,tile_m=128,tile_n=128,
                  threads=128,smem=65536,steady_warmup_rounds=24,
                  alternatives=[dict(reference)])
        suite.append(case)
    (HERE/'native_suite.json').write_text(json.dumps(suite,indent=2)+'\n')
    (HERE/'native-build-evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print('All six native CLI instruction streams match the measured research kernels.')

if __name__=='__main__':main()
