import random, sys
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0,'tools')
import mine_latency as ML, predict_stall as PS
from openassembler.emu.program import load_listing
from openassembler.isa.word import control
files=sorted(Path('corpus/sm_80/sass').glob('*.sass'))+sorted(Path('corpus/sm_80/divconst/sass').glob('*.sass'))
rng=random.Random(0); rng.shuffle(files)
train,test=files[:len(files)//2],files[len(files)//2:]
fixed,pair_min,var,_,_,_,_=ML.mine(train)
lat={c:min(h) for c,h in fixed.items() if h}
pairs={f"{a}->{b}":v for (a,b),v in pair_min.items()}
def blks(fs):
    for f in fs:
        yield from ML.blocks(load_listing(f.read_text()))
occ=defaultdict(Counter); gap=defaultdict(Counter)
for blk in blks(train):
    for k in range(len(blk)-1):
        a,b=blk[k],blk[k+1]; da,_=ML.defs_uses(a); _,sb=ML.defs_uses(b)
        if set(da)&set(sb): continue
        gap[ML.op_class(b)][control(a.word).stall]+=1
        if not control(b.word).wait and PS.pipe(a)==PS.pipe(b): occ[PS.pipe(a)][control(a.word).stall]+=1
occ={p:min(c) for p,c in occ.items() if sum(c.values())>=5}
gap={c:min(h) for c,h in gap.items() if sum(h.values())>=3}
tot=hit=0; seen=unseen=0
for blk in blks(test):
    if len(blk)<2: continue
    pred=PS.predict_block(blk,lat,pairs,occ,gap=gap)
    for k,p in enumerate(pred):
        if blk[k].op=='NOP': continue
        tot+=1; hit+= p==control(blk[k].word).stall
print(f"train {len(train)} listings / test {len(test)}: held-out exact stall match {hit}/{tot} = {100*hit/tot:.1f}%")
