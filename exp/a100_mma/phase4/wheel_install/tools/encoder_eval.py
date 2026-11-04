import json, random, re
from openassembler.isa.encode import Encoder, EncodeError
from openassembler.isa.word import control
from openassembler import oracle
enc=Encoder('sm_80')
rows=[json.loads(l) for l in open('corpus/sm_80/pairs.jsonl')]
uniq={}
for r in rows: uniq.setdefault(int(r['word'],16), r['text'])
ok=bad=refused=0; badex=[]
for w,t in uniq.items():
    try: g=enc.encode(t, control(w))
    except EncodeError: refused+=1; continue
    if g==w: ok+=1
    else:
        bad+=1
        if len(badex)<6: badex.append((t,hex(w^g)))
print(f"round-trip on corpus: {ok} exact, {bad} differ, {refused} refused, of {len(uniq)}")
for b in badex: print('  differ:',b)
rng=random.Random(5); novel=[]
for w,t in uniq.items():
    if t.startswith(('BRA','CALL','RET','BSSY','EXIT','NOP','BRX')) or '`' in t: continue
    t2=re.sub(r'\bR(\d+)\b', lambda m: f"R{rng.randrange(0,250)&~1 if '.64' in t or 'WIDE' in t or '128' in t else rng.randrange(0,250)}", t)
    t2=re.sub(r'\+0x[0-9a-f]+\]', lambda m: f"+{rng.randrange(0,1<<20)*4:#x}]", t2)
    try: novel.append((t2, enc.encode(t2, control(w))))
    except EncodeError: pass
txt=oracle.disasm_words([w for _,w in novel])
norm=lambda s: re.sub(r'\.reuse','',s or '').replace(' ','')
agree=sum(norm(a)==norm(b) for (a,_),b in zip(novel,txt))
print(f"novel operands (random registers + offsets): {agree}/{len(novel)} = {100*agree/len(novel):.1f}% disassemble to the intended text")
for (a,_),b in [x for x in zip(novel,txt) if norm(x[0][0])!=norm(x[1])][:8]: print('   want',a,'| got',b)
