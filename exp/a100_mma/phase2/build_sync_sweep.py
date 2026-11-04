"""Change exactly one synchronization issue-gap family in a passing binary."""
import hashlib,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE))
from backend import encoder
from openassembler import build,oracle
from openassembler.container.cubin import text_words
from openassembler.isa.word import control
import dataclasses

if __name__=='__main__':
    enc=encoder();source=BASE/'build/validated_square_f16'
    manifest=json.loads((source/'manifest.json').read_text());entry=manifest['entries'][0]
    original=(source/entry['cubin']).read_bytes()
    rows=oracle.parse_listing((source/entry['sass']).read_text())
    words=text_words(original,'gemm')
    exits=[i*16 for i,r in enumerate(rows) if r.text=='EXIT']
    regs=entry['kernels'][0]['registers'];suite=[]
    for family,stalls in (('BAR',(1,2,4,6,8)),('DEPBAR',(1,2,3,4,6))):
        for stall in stalls:
            name=f'p2_sync_{family.lower()}{stall}';dest=BASE/'build'/name;changed=list(words)
            positions=[]
            for i,r in enumerate(rows):
                if r.text.split('.')[0].split()[0]==family:
                    changed[i]=enc._ctl(words[i],dataclasses.replace(control(words[i]),stall=stall,yield_=False))
                    positions.append(i)
            binary=build.splice_kernels(original,{'gemm':(changed,regs,exits)})
            for key in ('cubin','reference','ptx','sass'):
                if entry.get(key):
                    target=dest/entry[key];target.parent.mkdir(parents=True,exist_ok=True)
                    target.write_bytes(binary if key=='cubin' else (source/entry[key]).read_bytes())
            (dest/entry['sass']).write_text(build.disasm(binary))
            (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            (dest/'experiment.json').write_text(json.dumps({
                'origin_binary_sha256':hashlib.sha256(original).hexdigest(),'mutation':{'opcode':family,'stall':stall,'positions':positions},
                'semantics':'Only control stall fields differ; no instruction, register, or memory address changes'},indent=2)+'\n')
            suite.append(dict(build=name,m=4096,n=4096,k=4096,dtype='f16',tile_m=128,tile_n=128,threads=128,smem=65536))
    (BASE/'phase2/sync-suite.json').write_text(json.dumps(suite,indent=2)+'\n')
