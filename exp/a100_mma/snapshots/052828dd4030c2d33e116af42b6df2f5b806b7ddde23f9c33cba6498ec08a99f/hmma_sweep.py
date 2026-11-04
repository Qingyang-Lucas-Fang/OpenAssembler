"""HMMA accumulator-dependency stall experiment; not a throughput benchmark."""
from __future__ import annotations
import base64
import dataclasses
import json
import sys

sys.dont_write_bytecode = True
from backend import HERE, encoder, generate, prepare
from openassembler import build
from openassembler.isa.word import control
from openassembler.ptx.parse import parse_module


def probe(steps=128):
    text = [".version 8.7\n.target sm_80\n.address_size 64",
            ".visible .entry gemm(.param .u64 out, .param .u64 inp, .param .u32 k, .param .f32 kf) {",
            ".reg .b32 %r<20>; .reg .b64 %d<10>; .reg .f32 %c<4>;",
            "ld.param.u64 %d0,[out]; ld.param.u64 %d1,[inp]; mov.u32 %r0,%tid.x;",
            "mul.wide.u32 %d2,%r0,16; add.u64 %d3,%d1,%d2; add.u64 %d4,%d0,%d2;",
            "ld.global.v4.f32 {%c0,%c1,%c2,%c3},[%d3];"]
    text += [f"mov.b32 %r{i},0x3c003c00;" for i in range(1, 7)]
    text += ["mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
             "{%c0,%c1,%c2,%c3},{%r1,%r2,%r3,%r4},{%r5,%r6},{%c0,%c1,%c2,%c3};"] * steps
    text += ["st.global.v4.f32 [%d4],{%c0,%c1,%c2,%c3}; ret; }\n"]
    return "\n".join(text)


if __name__ == "__main__":
    prepare()
    root = HERE / "hmma"
    root.mkdir(exist_ok=True)
    source = probe()
    (root / "dependency.ptx").write_text(source)
    module = parse_module(source)
    code, labels, original, exits, regs = generate(module.kernels["gemm"], "sm_80", "safe")
    reference = build.ptxas_build(source, "sm_80")
    (root / "reference.cubin").write_bytes(reference)
    positions = [i for i, (text, _) in enumerate(code) if text.startswith("HMMA.")]
    if len(positions) != 128:
        raise AssertionError("unexpected number of dependency-chain operations")
    enc = encoder()
    suite = []
    for gap in list(range(2, 21)) + [24, 30]:
        words = list(original)
        for i in positions[:-1]:
            first = min(15, gap - 1)
            second = gap - first
            for pos, stall in ((i, first), (i + 1, second)):
                ctl = dataclasses.replace(control(words[pos]), stall=stall, yield_=stall > 11)
                words[pos] = enc._ctl(words[pos], ctl)
        cubin = build.splice_kernels(reference, {"gemm": (words, regs, exits)})
        path = root / f"dependency-gap{gap:02d}.cubin"
        path.write_bytes(cubin)
        if gap in (2, 8, 16, 30):
            path.with_suffix(".sass").write_text(build.disasm(cubin))
        suite.append({"kind": "hmma", "name": f"hmma_gap{gap:02d}", "gap": gap,
                      "steps": 128, "cubin_path": str(path.relative_to(HERE))})
    (HERE / "hmma-suite.json").write_text(json.dumps(suite, indent=2) + "\n")
    stores = []
    for gap in list(range(2, 17)) + [24, 30]:
        words = list(original)
        i = positions[-1]
        first = min(15, gap - 1)
        for pos, stall in ((i, first), (i + 1, gap - first)):
            ctl = dataclasses.replace(control(words[pos]), stall=stall, yield_=stall > 11)
            words[pos] = enc._ctl(words[pos], ctl)
        path = root / f"store-gap{gap:02d}.cubin"
        path.write_bytes(build.splice_kernels(reference, {"gemm": (words, regs, exits)}))
        stores.append({"kind": "hmma", "name": f"hmma_store_gap{gap:02d}", "gap": gap,
                       "steps": 128, "cubin_path": str(path.relative_to(HERE)), "family": "result-to-store"})
    (HERE / "hmma-store-suite.json").write_text(json.dumps(stores, indent=2) + "\n")
