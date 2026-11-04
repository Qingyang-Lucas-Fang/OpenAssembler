"""Generate and z3-prove multiply-high replacements for div/rem by constants.

Usage: uv run --no-project --with z3-solver==4.15.3.0 python tools/prove_divconst.py [N_RANDOM]
Compares instruction counts against ptxas 12.8 (corpus/sm_80/div_const_survey.json).
"""
import json, random, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from openassembler.opt.divconst import lower, prove, run, to_sass

ROOT = Path(__file__).resolve().parent.parent
survey = json.loads((ROOT / "corpus/sm_80/div_const_survey.json").read_text())
divs = [2, 3, 5, 6, 7, 10, 16, 25, 60, 100, 255, 641, 1000, 0x1234, 65537, 1000000007, 0x7FFFFFFF]
rng = random.Random(1)
extra = int(sys.argv[1]) if len(sys.argv) > 1 else 200
divs_all = divs + [rng.randrange(2, 1 << 31) for _ in range(extra)]
rows, t0 = [], time.time()
proved = failed = 0
for op in ("div", "rem"):
    for signed in (False, True):
        for d in divs_all:
            seq = lower(op, signed, d)
            # cheap concrete check first (edges + random), then the proof
            for x in [0, 1, d - 1, d, d + 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF] + [rng.getrandbits(32) for _ in range(64)]:
                xv = x & 0xFFFFFFFF
                if signed:
                    sx = xv - (1 << 32) if xv >> 31 else xv
                    q = abs(sx) // d * (1 if sx >= 0 else -1)
                    want = (q if op == "div" else sx - q * d) & 0xFFFFFFFF
                else:
                    want = xv // d if op == "div" else xv % d
                assert run(seq, xv) == want, (op, signed, d, x)
            res = prove(seq, d, signed, op == "rem")
            proved += res == "proved"; failed += res != "proved"
            key = f"{op}.{'s' if signed else 'u'}32.{d}"
            ptx = survey.get(key)
            rows.append({"case": key, "kind": seq.kind, "ours": seq.n, "ptxas_extra": ptx["extra"] if ptx else None,
                         "ptxas_xu": ptx["xu"] if ptx else None, "proof": res, "sass": to_sass(seq)})
out = ROOT / "corpus/sm_80/divconst_proofs.json"
out.write_text(json.dumps(rows, indent=1))
print(f"{proved} proved, {failed} not proved, {len(rows)} cases, {time.time() - t0:.1f}s")
print(f"{'case':22} {'kind':14} {'ours':>4} {'ptxas':>5} {'ptxas XU':>8}  proof")
for r in rows:
    if r["ptxas_extra"] is not None:
        print(f"{r['case']:22} {r['kind']:14} {r['ours']:>4} {r['ptxas_extra']:>5} {r['ptxas_xu']:>8}  {r['proof']}")
print("e.g. div.u32 by 7:", " ; ".join(next(r for r in rows if r['case'] == 'div.u32.7')['sass']))
print("e.g. div.s32 by 1000:", " ; ".join(next(r for r in rows if r['case'] == 'div.s32.1000')['sass']))
