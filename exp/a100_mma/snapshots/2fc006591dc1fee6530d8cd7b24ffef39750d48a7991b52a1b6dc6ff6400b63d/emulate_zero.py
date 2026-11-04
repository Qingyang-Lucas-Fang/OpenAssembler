"""One-CTA functional check ignores timing, separating timing from IR bugs."""
import json
import struct
import sys
import random
from pathlib import Path
from backend import HERE
from openassembler.emu.machine import Machine, Memory, Launch
from openassembler.emu.program import load_listing

name = sys.argv[1]
root = HERE / "build" / name
manifest = json.loads((root/"manifest.json").read_text())
entry = manifest["entries"][0]
memory = Memory()
a,b,c = 0x10000000,0x20000000,0x30000000
integer = len(sys.argv) > 2 and sys.argv[2] == "integer"
rng = random.Random(9)
av = [rng.randrange(-3,4) if integer else 0 for _ in range(128*256)]
bv = [rng.randrange(-3,4) if integer else 1 for _ in range(192*256)]
memory.write_bytes(a, b''.join(struct.pack('<e',v) for v in av))
memory.write_bytes(b, b''.join(struct.pack('<e',v) for v in bv))
params = struct.pack("<3Q",a,b,c)
program = load_listing((root/entry["sass"]).read_text(),kernel="gemm")
Machine(program,Launch(grid=(2,3,1),block=(128,1,1),ctaid=(0,0,0),params=params,smem_static=16384,
                       poison=0xDEADBEEF),memory).run(max_steps=1000000)
values = [struct.unpack('<f',struct.pack('<I',memory.read(c+(r*192+col)*4,4)))[0]
          for r in range(64) for col in range(64)]
expected = [sum(av[r*256+q]*bv[col*256+q] for q in range(256)) for r in range(64) for col in range(64)]
bad = [(i,x,y) for i,(x,y) in enumerate(zip(values,expected)) if x!=y]
print(json.dumps({"name":name,"integer":integer,"mismatches":len(bad),"examples":bad[:16]}))
