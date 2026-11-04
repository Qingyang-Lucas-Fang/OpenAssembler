"""Survey: how does ptxas lower integer division/remainder by a compile-time constant?

Usage: uv run --no-project python tools/div_const_survey.py [sm_80]

For each (op, type, divisor) we compile a probe whose divisor is a PTX immediate,
then classify the SASS:
  shift     power of two handled with shifts/LOP3 only
  magic     multiply-high sequence (IMAD.HI / IMAD.WIDE), no MUFU/I2F/F2I
  runtime   reciprocal computed at run time (I2F + MUFU.RCP + F2I + Newton)
  call      calls an internal helper (64-bit slow path)
Writes corpus/<arch>/div_const_survey.json and docs/findings-div-const.md data table.
"""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ptx_probes  # noqa: E402
from openassembler import oracle  # noqa: E402

DIVISORS = [2, 3, 5, 6, 7, 10, 16, 25, 60, 100, 255, 641, 1000, 0x1234, 65537, 1000000007, 0x7FFFFFFF]
TYPES = {"u32": ("%r1", "%r10", ""), "s32": ("%r1", "%r10", ""),
         "u64": ("%rd6", "%rd10", " mov.b64 {%r10, %r11}, %rd10;"),
         "s64": ("%rd6", "%rd10", " mov.b64 {%r10, %r11}, %rd10;")}
XU = ("MUFU", "I2F", "F2I")

VM_SCRIPT = r"""
cd {dir} && ls ptx/*.ptx | xargs -P4 -I{{}} sh -c '
  b=$(basename {{}} .ptx)
  ptxas -arch={target} -O3 -o cubin/$b.cubin {{}} && nvdisasm -c cubin/$b.cubin > sass/$b.sass'
"""


def body(op: str, t: str, d: int, loop: bool) -> str:
    src, dst, post = TYPES[t]
    one = f" {op}.{t} {dst}, {src}, {d};{post}"
    if not loop:
        return one
    # Loop variant: the divisor is invariant, so any setup should be hoisted out.
    acc = "%r10"
    return (f" mov.u32 %r30, 0; mov.u32 %r31, 0; LP: {op}.u32 %r32, %r1, {d}; add.u32 %r31, %r31, %r32;"
            f" add.u32 %r1, %r1, %r2; add.u32 %r30, %r30, 1; setp.lt.u32 %p1, %r30, %r3; @%p1 bra LP;"
            f" mov.u32 {acc}, %r31;")


def classify(lines: list[str]) -> tuple[str, int, int]:
    ops = [l.split()[0].lstrip("@!P0123456T ") for l in lines]
    mn = [l.split()[1] if l.startswith("@") else l.split()[0] for l in lines]
    base = [m.split(".")[0] for m in mn]
    n_xu = sum(b in XU for b in base)
    if any(b == "CALL" for b in base):
        kind = "call"
    elif n_xu:
        kind = "runtime"
    elif any(m.startswith(("IMAD.HI", "IMAD.WIDE")) and "c[0x0]" not in l for m, l in zip(mn, lines)):
        kind = "magic"
    else:
        kind = "shift"
    return kind, len(lines), n_xu


def insns(sass: str) -> list[str]:
    out = []
    for i in oracle.parse_listing(sass) if False else []:
        pass
    for line in sass.splitlines():
        line = line.strip()
        if line.startswith("/*") and "*/" in line:
            t = line.split("*/", 1)[1].split(";")[0].strip()
            if t and t != "NOP":
                out.append(t)
    return out


def main(target: str = "sm_80") -> None:
    work = oracle.ROOT / "corpus" / target / "divconst"
    for sub in ("ptx", "cubin", "sass"):
        (work / sub).mkdir(parents=True, exist_ok=True)
        for f in (work / sub).iterdir():
            f.unlink()
    cases = {}
    for op in ("div", "rem"):
        for t in TYPES:
            for d in DIVISORS:
                if t.startswith("s") and d > 0x7FFFFFFF:
                    continue
                cases[f"{op}.{t}.{d}"] = body(op, t, d, False)
        for d in DIVISORS:
            cases[f"{op}.u32.{d}.loop"] = body(op, "u32", d, True)
    # Baseline: same probe with a trivial op, to subtract the fixed prologue/epilogue.
    cases["base.u32"] = " add.u32 %r10, %r1, 7;"
    cases["base.u64"] = " add.u64 %rd10, %rd6, 7; mov.b64 {%r10, %r11}, %rd10;"
    for name, b in cases.items():
        (work / "ptx" / f"{ptx_probes.kernel_name(name)}.ptx").write_text(
            ptx_probes.module(target, name, b, False))
    oracle.run_tool("bash", "-c", VM_SCRIPT.format(dir=shlex.quote(str(work)), target=target))

    res = {}
    for name in cases:
        lines = insns((work / "sass" / f"{ptx_probes.kernel_name(name)}.sass").read_text())
        res[name] = dict(zip(("kind", "insns", "xu"), classify(lines)))
    b32, b64 = res["base.u32"]["insns"], res["base.u64"]["insns"]
    for name, r in res.items():
        if not name.startswith("base"):
            r["extra"] = r["insns"] - (b64 if "64." in name else b32)
    (oracle.ROOT / "corpus" / target / "div_const_survey.json").write_text(json.dumps(res, indent=1))
    for op in ("div", "rem"):
        for t in TYPES:
            row = [f"{d}:{res[k]['kind'][0]}{res[k]['extra']}" for d in DIVISORS
                   if (k := f"{op}.{t}.{d}") in res]
            print(f"{op}.{t:4} " + " ".join(row))
        row = [f"{d}:{res[k]['kind'][0]}{res[k]['xu']}" for d in DIVISORS if (k := f"{op}.u32.{d}.loop") in res]
        print(f"{op}.u32 loop (kind, XU ops) " + " ".join(row))


if __name__ == "__main__":
    main(*sys.argv[1:])
