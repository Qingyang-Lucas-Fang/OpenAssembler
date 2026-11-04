"""M2 gate: predict ptxas's stall counts from a mined machine model.

Usage: uv run --no-project python tools/predict_stall.py [sm_80]

Model (in-order issue simulation per basic block):
  issue(k+1) = max( issue(k) + 1,
                    issue(i) + L(i)            for fixed-latency producers i of k+1's sources,
                    issue(p) + occ(pipe)       for the last instruction p on k+1's pipe )
  stall(k)   = issue(k+1) - issue(k)
L comes from corpus/<arch>/machine_model.json (min distance ptxas ever used).
Pipe occupancy is mined too: the min stall ptxas puts between two independent
instructions on the same pipe.

Reports exact-match accuracy of the predicted stall field per instruction class.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mine_latency import blocks, defs_uses, op_class  # noqa: E402
from openassembler.emu.program import load_listing  # noqa: E402
from openassembler.isa.word import control  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

PIPE = {}
for o in ("IADD3", "LOP3", "SHF", "ISETP", "SEL", "PRMT", "MOV", "LEA", "IMNMX", "IABS", "FLO", "POPC",
          "BREV", "SGXT", "BMSK", "PLOP3", "FSEL", "FSETP", "FMNMX", "P2R", "R2P", "VOTE", "CS2R", "FCHK",
          "HMNMX2", "VABSDIFF", "IDP"):
    PIPE[o] = "alu"
for o in ("FFMA", "FADD", "FMUL", "IMAD", "HFMA2", "HADD2", "HMUL2"):
    PIPE[o] = "fma"
for o in ("ULDC", "UMOV", "UIADD3", "ULOP3", "USHF", "UISETP", "S2UR", "ULEA", "USEL", "UPLOP3"):
    PIPE[o] = "uniform"


def pipe(ins) -> str:
    return PIPE.get(ins.op, "other:" + ins.op)


def load_model(arch: str):
    m = json.loads((ROOT / "corpus" / arch / "machine_model.json").read_text())
    lat = {c: v["fixed_min"] for c, v in m["classes"].items() if v["fixed_min"] is not None}
    return lat, m["pairs"]


def iter_blocks(arch):
    corpus = ROOT / "corpus" / arch
    for f in sorted((corpus / "sass").glob("*.sass")) + sorted((corpus / "divconst" / "sass").glob("*.sass")):
        prog = load_listing(f.read_text())
        for blk in blocks(prog):
            yield blk


def mine_occupancy(arch) -> dict[str, int]:
    """Min stall between consecutive *independent* instructions on the same pipe."""
    occ = defaultdict(Counter)
    for blk in iter_blocks(arch):
        for k in range(len(blk) - 1):
            a, b = blk[k], blk[k + 1]
            da, _ = defs_uses(a)
            _, sb = defs_uses(b)
            if set(da) & set(sb) or control(b.word).wait:
                continue
            if pipe(a) == pipe(b):
                occ[pipe(a)][control(a.word).stall] += 1
    return {p: min(c) for p, c in occ.items() if sum(c.values()) >= 5}, occ


def mine_min_gap(arch) -> dict[str, int]:
    """Min stall ptxas ever puts on the instruction *before* a given consumer class
    (independent pair). Captures issue constraints of the consumer's pipe, e.g.
    S2R / LDG / MUFU need >= 2 cycles after the previous issue."""
    gap = defaultdict(Counter)
    for blk in iter_blocks(arch):
        for k in range(len(blk) - 1):
            a, b = blk[k], blk[k + 1]
            da, _ = defs_uses(a)
            _, sb = defs_uses(b)
            if set(da) & set(sb):
                continue
            gap[op_class(b)][control(a.word).stall] += 1
    return {c: min(h) for c, h in gap.items() if sum(h.values()) >= 3}


def predict_block(blk, lat, pairs, occ, use_pairs=True, use_occ=True, gap=None):
    issue = [0]
    last_on_pipe = {}
    last_write = {}
    preds = []
    for k, ins in enumerate(blk):
        if k > 0:
            t = issue[k - 1] + 1
            _, srcs = defs_uses(ins)
            for r in srcs:
                if r in last_write:
                    i, prod = last_write[r]
                    pc = op_class(prod)
                    c = control(prod.word)
                    if c.wbar != 7:
                        continue  # scoreboarded: consumer waits, no stall needed
                    L = pairs.get(f"{pc}->{op_class(ins)}") if use_pairs else None
                    L = L if L is not None else lat.get(pc)
                    if L is not None:
                        t = max(t, issue[i] + L)
            if gap and op_class(ins) in gap:
                t = max(t, issue[k - 1] + gap[op_class(ins)])
            p = pipe(ins)
            if use_occ and p in occ and p in last_on_pipe:
                t = max(t, issue[last_on_pipe[p]] + occ[p])
            issue.append(t)
            preds.append(min(15, t - issue[k - 1]))
        d, _ = defs_uses(ins)
        for r in d:
            last_write[r] = (k, ins)
        last_on_pipe[pipe(ins)] = k
    return preds  # stall for blk[0..n-2]


def evaluate(arch, lat, pairs, occ, **kw):
    tot = hit = 0
    by_cls = defaultdict(lambda: [0, 0])
    errs = Counter()
    for blk in iter_blocks(arch):
        if len(blk) < 2:
            continue
        pred = predict_block(blk, lat, pairs, occ, **kw)
        for k, p in enumerate(pred):
            ins = blk[k]
            if ins.op == "NOP":
                continue
            real = control(ins.word).stall
            tot += 1
            ok = p == real
            hit += ok
            c = by_cls[op_class(ins)]
            c[0] += 1
            c[1] += ok
            if not ok:
                errs[(op_class(ins), op_class(blk[k + 1]), p, real)] += 1
    return tot, hit, by_cls, errs


def main(arch="sm_80"):
    lat, pairs = load_model(arch)
    occ, occ_hist = mine_occupancy(arch)
    print("mined pipe occupancy (min stall between independent same-pipe insns):", occ)
    gap = mine_min_gap(arch)
    for name, kw in (("latency only", dict(use_pairs=False, use_occ=False)),
                     ("+pair latencies", dict(use_pairs=True, use_occ=False)),
                     ("+pipe occupancy", dict(use_pairs=True, use_occ=True)),
                     ("+consumer issue gap", dict(use_pairs=True, use_occ=True, gap=gap))):
        tot, hit, by_cls, errs = evaluate(arch, lat, pairs, occ, **kw)
        print(f"{name:18} exact stall match: {hit}/{tot} = {100 * hit / tot:.1f}%")
    print("\nper producer class (final model):")
    for cls, (n, h) in sorted(by_cls.items(), key=lambda kv: -kv[1][0])[:22]:
        print(f"  {cls:22} {h:5}/{n:<5} {100 * h / n:5.1f}%")
    print("\ntop mispredictions (insn, next insn, predicted, ptxas):")
    for (a, b, p, r), n in errs.most_common(15):
        print(f"  {n:4}  {a:20} -> {b:20} pred={p:2} ptxas={r:2}")
    out = ROOT / "corpus" / arch / "stall_model_eval.json"
    out.write_text(json.dumps({"occupancy": occ, "accuracy": hit / tot, "n": tot,
                               "by_class": {c: v for c, v in by_cls.items()}}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
