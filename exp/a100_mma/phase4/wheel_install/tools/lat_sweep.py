"""Measure fixed-pipeline read-after-write latencies on real hardware.

    PYTHONPATH=. python3 tools/lat_sweep.py                 # -> corpus/sm_80/latsweep
    modal run tools/gpu_modal.py --which dir:corpus/sm_80/latsweep --trials 3 --budget 200
    PYTHONPATH=. python3 tools/lat_sweep.py --report corpus/sm_80/gpu_a100_40gb_latsweep.json

Each kernel is a producer immediately followed by a consumer of its result in a given
operand slot. Our compiler builds it; then the producer's stall is set to s = 1..8 and
one cubin per s is written ("s<s>@<kernel>.cubin"). The GPU result is compared with the
emulator (which has no timing): the smallest s that matches on every lane is the real
latency for (producer, consumer slot). ptxas's mined minimum distances turned out not to
be safe latencies (IMAD -> LOP3 at 4 cycles gave wrong results on the A100).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

import ptx_probes  # noqa: E402
import random  # noqa: E402

from bench_build import _reschedule_k, check  # noqa: E402
from openassembler import oracle  # noqa: E402
from openassembler.backend import ra  # noqa: E402
from openassembler.container.cubin import required_regcount, splice_text  # noqa: E402
from openassembler.isa.encode import Encoder  # noqa: E402
from openassembler.ptx.parse import parse_module  # noqa: E402

# name: (producer opcode prefix, PTX body). Inputs %r1..%r4 / %f1..%f3; out %r10, %r11.
F = " mov.b32 %r11, %f21; mov.b32 %r10, %f20;"
SWEEP = {
    "IMAD>LOP3.a": ("IMAD ", " mad.lo.u32 %r20, %r1, %r2, %r3; xor.b32 %r11, %r20, %r4; mov.u32 %r10, %r20;"),
    "IMAD>LOP3.b": ("IMAD ", " mad.lo.u32 %r20, %r1, %r2, %r3; xor.b32 %r11, %r4, %r20; mov.u32 %r10, %r20;"),
    "IMAD>IMAD.a": ("IMAD ", " mad.lo.u32 %r20, %r1, %r2, %r3; mad.lo.u32 %r11, %r20, %r4, %r1; mov.u32 %r10, %r20;"),
    "IMAD>IMAD.b": ("IMAD ", " mad.lo.u32 %r20, %r1, %r2, %r3; mad.lo.u32 %r11, %r4, %r20, %r1; mov.u32 %r10, %r20;"),
    "IMAD>IMAD.c": ("IMAD ", " mad.lo.u32 %r20, %r1, %r2, %r3; mad.lo.u32 %r11, %r4, %r1, %r20; mov.u32 %r10, %r20;"),
    "IMAD>IADD3.a": ("IMAD ", " mad.lo.u32 %r20, %r1, %r2, %r3; add.u32 %r11, %r20, %r4; mov.u32 %r10, %r20;"),
    "IMAD.HI>IADD3.a": ("IMAD.HI", " mul.hi.u32 %r20, %r1, %r2; add.u32 %r11, %r20, %r4; mov.u32 %r10, %r20;"),
    "IADD3>IADD3.a": ("IADD3 ", " add.u32 %r20, %r1, %r2; add.u32 %r11, %r20, %r4; mov.u32 %r10, %r20;"),
    "IADD3>IMAD.a": ("IADD3 ", " add.u32 %r20, %r1, %r2; mad.lo.u32 %r11, %r20, %r4, %r3; mov.u32 %r10, %r20;"),
    "IADD3>IMAD.c": ("IADD3 ", " add.u32 %r20, %r1, %r2; mad.lo.u32 %r11, %r4, %r3, %r20; mov.u32 %r10, %r20;"),
    "IADD3>LOP3.a": ("IADD3 ", " add.u32 %r20, %r1, %r2; xor.b32 %r11, %r20, %r4; mov.u32 %r10, %r20;"),
    "LOP3>IMAD.a": ("LOP3", " xor.b32 %r20, %r1, %r2; mad.lo.u32 %r11, %r20, %r4, %r3; mov.u32 %r10, %r20;"),
    "LOP3>IMAD.c": ("LOP3", " xor.b32 %r20, %r1, %r2; mad.lo.u32 %r11, %r4, %r3, %r20; mov.u32 %r10, %r20;"),
    "LOP3>LOP3.a": ("LOP3", " xor.b32 %r20, %r1, %r2; xor.b32 %r11, %r20, %r4; mov.u32 %r10, %r20;"),
    "LOP3>IADD3.a": ("LOP3", " xor.b32 %r20, %r1, %r2; add.u32 %r11, %r20, %r4; mov.u32 %r10, %r20;"),
    "SHF>IADD3.a": ("SHF", " shl.b32 %r20, %r1, 3; add.u32 %r11, %r20, %r4; mov.u32 %r10, %r20;"),
    "IMNMX>IADD3.a": ("IMNMX", " min.u32 %r20, %r1, %r2; add.u32 %r11, %r20, %r4; mov.u32 %r10, %r20;"),
    "FFMA>FFMA.a": ("FFMA", " fma.rn.f32 %f20, %f1, %f2, %f3; fma.rn.f32 %f21, %f20, %f2, %f1;" + F),
    "FFMA>FFMA.b": ("FFMA", " fma.rn.f32 %f20, %f1, %f2, %f3; fma.rn.f32 %f21, %f2, %f20, %f1;" + F),
    "FFMA>FFMA.c": ("FFMA", " fma.rn.f32 %f20, %f1, %f2, %f3; fma.rn.f32 %f21, %f2, %f1, %f20;" + F),
    "FADD>FFMA.a": ("FADD", " add.rn.f32 %f20, %f1, %f2; fma.rn.f32 %f21, %f20, %f2, %f1;" + F),
    "FMUL>FADD.a": ("FMUL", " mul.rn.f32 %f20, %f1, %f2; add.rn.f32 %f21, %f20, %f3;" + F),
    "FFMA>FMUL.a": ("FFMA", " fma.rn.f32 %f20, %f1, %f2, %f3; mul.rn.f32 %f21, %f20, %f2;" + F),
    "FMNMX>FADD.a": ("FMNMX", " min.f32 %f20, %f1, %f2; add.rn.f32 %f21, %f20, %f3;" + F),
    "MOV>IADD3.a": (r"MOV R\d+, 0x1234", " mov.u32 %r20, 0x1234; add.u32 %r11, %r20, %r4; mov.u32 %r10, %r20;"),
    "SEL>IADD3.a": ("SEL", " setp.lt.u32 %p1, %r1, %r2; selp.u32 %r20, %r3, %r4, %p1;"
                    " add.u32 %r11, %r20, %r1; mov.u32 %r10, %r20;"),
    "ISETP>SEL.p": ("ISETP", " setp.lt.u32 %p1, %r1, %r2; selp.u32 %r11, %r3, %r4, %p1; mov.u32 %r10, %r1;"),
    "ISETP>IADD3.g": ("ISETP", " mov.u32 %r11, %r1; setp.lt.u32 %p1, %r1, %r2; @%p1 add.u32 %r11, %r3, %r4;"
                      " mov.u32 %r10, %r2;"),
    "FSETP>FADD.g": ("FSETP", " mov.f32 %f21, %f1; setp.lt.f32 %p1, %f1, %f2; @%p1 add.rn.f32 %f21, %f2, %f3;"
                     " mov.b32 %r11, %f21; mov.u32 %r10, %r2;"),
    "ISETP>IMAD.g": ("ISETP", " mov.u32 %r11, %r1; setp.lt.u32 %p1, %r1, %r2; @%p1 mad.lo.u32 %r11, %r3, %r4, %r2;"
                     " mov.u32 %r10, %r2;"),
    "FSETP>SEL.p": ("FSETP", " setp.lt.f32 %p1, %f1, %f2; selp.u32 %r11, %r3, %r4, %p1; mov.u32 %r10, %r1;"),
    "FFMA>LOP3.a": ("FFMA", " fma.rn.f32 %f20, %f1, %f2, %f3; mov.b32 %r20, %f20; xor.b32 %r11, %r20, %r4; mov.u32 %r10, %r20;"),
    "IADD3>FFMA.a": ("IADD3", " add.u32 %r20, %r1, %r2; mov.b32 %f20, %r20; fma.rn.f32 %f21, %f20, %f2, %f3;" + F),
    "LOP3>FADD.a": ("LOP3", " xor.b32 %r20, %r1, %r2; mov.b32 %f20, %r20; add.rn.f32 %f21, %f20, %f3;" + F),
    "IADD3>ISETP.a": ("IADD3", " add.u32 %r20, %r1, %r2; setp.lt.u32 %p1, %r20, %r3; selp.u32 %r11, %r3, %r4, %p1;"
                      " mov.u32 %r10, %r20;"),
    "IMAD>ISETP.a": ("IMAD ", " mad.lo.u32 %r20, %r1, %r2, %r3; setp.lt.u32 %p1, %r20, %r3; selp.u32 %r11, %r3, %r4, %p1;"
                     " mov.u32 %r10, %r20;"),
    "IMAD.WIDE>IADD3.a": (r"IMAD.WIDE.U32 R\d+, R\d+, R\d+, RZ", " mul.wide.u32 %rd10, %r1, %r2; mov.b64 {%r20, %r21}, %rd10;"
                          " add.u32 %r11, %r21, %r4; mov.u32 %r10, %r20;"),
    "IMAD.HI>IMAD.a": ("IMAD.HI", " mul.hi.u32 %r20, %r1, %r2; mad.lo.u32 %r11, %r20, %r4, %r3; mov.u32 %r10, %r20;"),
    "SHF>IMAD.a": ("SHF", " shl.b32 %r20, %r1, 3; mad.lo.u32 %r11, %r20, %r4, %r3; mov.u32 %r10, %r20;"),
    "IADD3>STG.d": ("IADD3", " mov.u32 %r10, %r3; add.u32 %r11, %r1, %r2;"),
    "IMAD>STG.d": ("IMAD ", " mov.u32 %r10, %r4; mad.lo.u32 %r11, %r1, %r2, %r3;"),
    "FFMA>STG.d": ("FFMA", " mov.u32 %r10, %r4; fma.rn.f32 %f21, %f1, %f2, %f3; mov.b32 %r11, %f21;"),
}
STALLS = range(1, 9)
GUARD_STALLS = range(1, 16)   # predicate -> guard: much longer (see report)


def stalls(n: str):
    return GUARD_STALLS if n.endswith('.g') else STALLS


def kname(n: str) -> str:
    return ptx_probes.kernel_name("lat." + n)


def build(arch: str) -> None:
    out = ROOT / "corpus" / arch / "latsweep"
    out.mkdir(parents=True, exist_ok=True)
    for f in out.iterdir():
        f.unlink()
    for n, (_p, body) in SWEEP.items():
        (out / f"{kname(n)}.ptx").write_text(ptx_probes.module(arch, "lat." + n, body, False))
    oracle.run_tool("bash", "-c", f"cd {shlex.quote(str(out))} && for f in *.ptx; do "
                    f"ptxas -arch={arch} -O3 -o ptxas@${{f%.ptx}}.cubin $f && "
                    f"nvdisasm -hex -c ptxas@${{f%.ptx}}.cubin > ptxas@${{f%.ptx}}.sass; done")
    enc = Encoder(arch)
    for n, (prod, _b) in SWEEP.items():
        kn = kname(n)
        ptx = (out / f"{kn}.ptx").read_text()
        code, labels = _reschedule_k(parse_module(ptx).kernels[kn], "safe")
        # producer = first `prod` op whose result the very next op reads
        idx = next((i for i, (t, _c) in enumerate(code[:-1]) if re.match(prod, t)), None)
        if idx is None:
            print(f"{n:18} producer {prod!r} not found:", [t for t, _ in code])
            continue
        err = check("lat." + n, kn, ptx, code, labels, random.Random(1))
        if err:
            print(f"{n:18} EMULATOR CHECK FAILED: {err}")
            continue
        base = (out / f"ptxas@{kn}.cubin").read_bytes()
        for s in stalls(n):
            c = list(code)
            c[idx] = (c[idx][0], dataclasses.replace(c[idx][1], stall=s, yield_=c[idx][1].yield_ or s > 11))
            words = ra.encode_code(enc, c, labels)
            nop = enc.encode("NOP")
            words.append(enc.encode_branch("BRA `(.L_x_0)", 16 * len(words), 16 * len(words)))
            while len(words) % 8:
                words.append(nop)
            exits = [16 * i for i, (t, _) in enumerate(c) if t.endswith("EXIT")]
            (out / f"s{s}@{kn}.cubin").write_bytes(
                splice_text(base, kn, words, required_regcount([t for t, _ in c]), exits))
        print(f"{n:18} {code[idx][0]:42} -> {code[idx + 1][0]}")


def report(path: Path) -> dict:
    rows = json.loads(path.read_text())["probes"]
    res = {}
    for n in SWEEP:
        ok = {s: rows.get(f"s{s}@{kname(n)}", {}).get("status") == "match" for s in stalls(n)}
        lat = next((s for s in stalls(n) if all(ok[t] for t in stalls(n) if t >= s)), None)
        res[n] = lat
        print(f"{n:18} latency {lat}   " + " ".join(f"{s}:{'ok' if ok[s] else '--'}" for s in stalls(n)))
    out = ROOT / "corpus" / "sm_80" / "hw_latency.json"
    out.write_text(json.dumps({"source": str(path), "latency": res}, indent=1))
    print(f"-> {out.relative_to(ROOT)}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="sm_80")
    ap.add_argument("--report")
    a = ap.parse_args()
    report(Path(a.report)) if a.report else build(a.arch)
