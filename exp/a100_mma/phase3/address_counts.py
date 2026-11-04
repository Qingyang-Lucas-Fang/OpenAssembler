"""Count the executed instruction path analytically, without a timing claim."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE.parents[1]))
from openassembler.emu.program import load_listing


def count(name, iterations):
    build = BASE/"build"/name
    manifest = json.loads((build/"manifest.json").read_text())
    program = load_listing((build/manifest["entries"][0]["sass"]).read_text())
    branch = next(ins for ins in program.insns.values()
                  if ins.op == "BRA" and ins.guard is not None)
    start = program.target(branch.ops[0])
    exit_addr = next(ins.addr for ins in program.insns.values() if ins.op == "EXIT")
    counts = Counter()
    regions = {}
    for tag, lo, hi, executions in [("prefix",0,start,1),
                                     ("loop",start,branch.addr+16,iterations),
                                     ("tail",branch.addr+16,exit_addr+16,1)]:
        block = [ins for ins in program.insns.values() if lo <= ins.addr < hi]
        regions[tag] = dict(static_instructions=len(block), executions=executions)
        for ins in block:
            counts[ins.op] += executions
    assert counts["HMMA"] == 8192 and counts["LDSM"] == 2048
    assert counts["LDGSTS"] == 1024 and counts["BAR"] == 128
    return dict(build=name, shape_mnk=[4096,4096,4096], regions=regions,
                dynamic_warp_instruction_count=sum(counts.values()),
                dynamic_opcode_counts=dict(counts),
                limitation="Analytical path counts with uniform loop bounds; not measured issue cycles, stalls or latency")


if __name__ == "__main__":
    rows = [count(name,iterations) for name,iterations in (
        ("search_combined_square_row_bpair_f16",63),
        ("address_shared_square_f16",63),
        ("address_stagepair_square_f16",31))]
    target = HERE/"address-dynamic-counts.json"
    target.write_text(json.dumps(rows,indent=2)+"\n")
    print(target)
