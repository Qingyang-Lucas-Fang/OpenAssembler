"""Clock only the intended SASS sequence; no hidden fragment-copy instructions."""
import hashlib,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE))
import backend
from openassembler import build
from openassembler.isa.word import Control


def make(family,n,gap):
    enc=backend.encoder();code=[]
    def emit(text,stall=6,wb=7,rb=7,wait=0):
        code.append((text,Control(stall,stall>11,wb,rb,wait,0)))
    emit('MOV R1, c[0x0][0x28]')
    emit('ULDC.64 UR4, c[0x0][0x118]')
    emit('S2R R0, SR_TID.X',2,1)
    emit('MOV R6, 0x10')
    emit('IMAD.WIDE.U32 R2, R0, R6, c[0x0][0x160]',wait=2)
    for r in range(8,14):emit(f'MOV R{r}, 0x3c003c00')
    groups=1 if family in ('dependent','ldsm') else 8
    for r in range(32,32+groups*4):emit(f'MOV R{r}, 0x0')
    if family=='ldsm':
        emit('IMAD R22, R0, 0x10, RZ')
        emit('STS.128 [R22], R8',2,7,1)
        emit('BAR.SYNC.DEFER_BLOCKING 0x0',6,wait=2)
        emit('MOV R0, R22')
    emit('CS2R.32 R240, SR_CLOCKLO')
    for i in range(n):
        if family=='ldsm':
            emit('LDSM.16.M88.4 R16, [R0]',4,1)
            emit('LOP3.LUT R20, R16, 0x0, RZ, 0xc0, !PT',6,wait=2)
            emit('IADD3 R0, R22, R20, RZ',6)
        else:
            r=32+(i%groups)*4
            emit(f'HMMA.16816.F32 R{r}, R8, R12, R{r}',gap)
    emit('NOP',15)
    emit('CS2R.32 R241, SR_CLOCKLO')
    emit('IADD3 R242, R241, -R240, RZ')
    for g in range(1,groups):emit(f'FADD R32, R32, R{32+g*4}')
    emit('MOV R243, R16' if family=='ldsm' else 'MOV R243, R32')
    emit('STG.E.64 [R2.64], R242',5,7,1)
    emit('EXIT',5,wait=2)
    from clocked import ra_encode
    words=ra_encode(enc,code)
    ptx='.version 8.7\n.target sm_80\n.address_size 64\n.visible .entry gemm(.param .u64 out,.param .u64 inp,.param .u32 k,.param .f32 kf) {'
    if family=='ldsm':ptx+=' .shared .align 16 .b8 smem[512]; .reg .b32 %r; mov.u32 %r,smem;st.shared.u32 [%r],1;bar.sync 0;'
    ptx+='ret;}\n'
    container=build.ptxas_build(ptx,'sm_80')
    binary=build.splice_kernels(container,{'gemm':(words,246,[16*(len(code)-1)])})
    return binary


if __name__=='__main__':
    root=BASE/'phase2/clocked-exact';root.mkdir(exist_ok=True);suite=[]
    for family in ('dependent','independent','ldsm'):
        for gap in ((4,) if family=='ldsm' else (2,4,8)):
            for n in (16,64,128,256):
                name=f'exact_{family}_g{gap}_n{n}';binary=make(family,n,gap)
                p=root/(name+'.cubin');p.write_bytes(binary)
                if n==64:p.with_suffix('.sass').write_text(build.disasm(binary))
                suite.append(dict(name=name,kind='clock',family=family,steps=n,encoded_issue_gap=gap,
                                  cubin_path=str(p.relative_to(BASE)),sha256=hashlib.sha256(binary).hexdigest(),
                                  expected_raw=0x3c003c00 if family=='ldsm' else None))
    (root/'suite.json').write_text(json.dumps(suite,indent=2)+'\n')
