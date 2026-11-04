"""Scoreboard (dependency barrier) policy of ptxas, and how much it over-waits.

Usage: uv run --no-project python tools/sb_policy.py [sm_80]

For each wait on scoreboard b at instruction j, the "needed" producers are the
in-flight variable-latency instructions that set b and whose result (write SB)
or source registers (read SB) instruction j actually touches.
  false wait:   j waits on b but none of b's in-flight producers is needed by j
  merged wait:  b has >1 in-flight producer, j needs only some of them
                (j is forced to wait for the slowest one)
Straight-line (per basic block) analysis; cross-block waits are counted separately.
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


def main(arch="sm_80"):
    corpus = ROOT / "corpus" / arch
    files = sorted((corpus / "sass").glob("*.sass"))
    wsb_by = defaultdict(Counter)
    stats = Counter()
    examples = []
    for f in files:
        prog = load_listing(f.read_text())
        for blk in blocks(prog):
            inflight = defaultdict(list)  # sb -> [(idx, insn, kind, regs)]
            for j, ins in enumerate(blk):
                c = control(ins.word)
                d, s = defs_uses(ins)
                touch = set(d) | set(s)
                for b in range(6):
                    if not (c.wait >> b & 1):
                        continue
                    stats["waits"] += 1
                    prods = inflight.get(b, [])
                    if not prods:
                        stats["wait_cross_block"] += 1
                        continue
                    needed = [p for p in prods if (p[2] == "w" and set(p[3]) & touch) or
                              (p[2] == "r" and set(p[3]) & set(d))]
                    if not needed:
                        stats["false_wait"] += 1
                        if len(examples) < 8:
                            examples.append((f.stem, blk[prods[0][0]].text, ins.text))
                    elif len(needed) < len(prods):
                        stats["merged_wait"] += 1
                    else:
                        stats["exact_wait"] += 1
                    inflight[b] = []
                if c.wbar != 7:
                    wsb_by[op_class(ins)][c.wbar] += 1
                    inflight[c.wbar].append((j, ins, "w", d))
                if c.rbar != 7:
                    inflight[c.rbar].append((j, ins, "r", s))
    sb_counts = Counter()
    for cls, h in wsb_by.items():
        for b, n in h.items():
            sb_counts[b] += n
    print(f"{arch}: {stats['waits']} scoreboard waits")
    for k in ("exact_wait", "merged_wait", "false_wait", "wait_cross_block"):
        print(f"  {k:17} {stats[k]:6}  {100 * stats[k] / max(1, stats['waits']):5.1f}%")
    print("write-SB usage over all ops:", dict(sorted(sb_counts.items())))
    print("write-SB by class (top):")
    for cls, h in sorted(wsb_by.items(), key=lambda kv: -sum(kv[1].values()))[:12]:
        print(f"  {cls:18} {dict(sorted(h.items()))}")
    for e in examples:
        print("  false-wait e.g.", e)
    (corpus / "sb_policy.json").write_text(json.dumps({"stats": stats, "wsb": wsb_by}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
