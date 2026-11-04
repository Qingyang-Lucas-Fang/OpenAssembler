"""Exhaustive proof of div/rem-by-constant sequences over all 2^32 inputs.

Usage: uv run --no-project python tools/verify_divconst_exhaustive.py [--jobs 8]

z3 times out on 64-bit multiply-high with non-power-of-two constants, so for
32-bit operations we prove by exhaustion instead: each sequence is compiled to C
(host clang -O3) and compared with C's own / and % on every x in [0, 2^32).
That is a complete proof for 32-bit operands. Writes corpus/sm_80/divconst_proofs.json.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openassembler.opt.divconst import M32, lower, to_sass  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DIVISORS = [2, 3, 5, 6, 7, 10, 16, 25, 60, 100, 255, 641, 1000, 0x1234, 65537, 1000000007, 0x7FFFFFFF]


def c_expr(seq) -> str:
    lines = []
    declared = {"x"}

    def v(o):
        return f"{o & M32}u" if isinstance(o, int) else o

    for op in seq.ops:
        k, d = op[0], op[1]
        a = op[2:]
        e = {
            "mov": lambda: v(a[0]),
            "mulhi_u": lambda: f"(uint32_t)(((uint64_t){v(a[0])} * (uint64_t){v(a[1])}) >> 32)",
            "mulhi_s": lambda: f"(uint32_t)(((int64_t)(int32_t){v(a[0])} * (int64_t)(int32_t){v(a[1])}) >> 32)",
            "shr_u": lambda: f"({v(a[0])} >> {a[1]})",
            "shr_s": lambda: f"(uint32_t)((int32_t){v(a[0])} >> {a[1]})",
            "add": lambda: f"({v(a[0])} + {v(a[1])})",
            "sub": lambda: f"({v(a[0])} - {v(a[1])})",
            "lea_hi_sign": lambda: f"({v(a[0])} + ({v(a[1])} >> 31))",
            "mad": lambda: f"({v(a[0])} * {v(a[1])} + {v(a[2])})",
            "geu": lambda: f"(uint32_t)({v(a[0])} >= {v(a[1])})",
        }[k]()
        decl = "" if d in declared else "uint32_t "
        declared.add(d)
        lines.append(f"{decl}{d} = {e};")
    return "\n    ".join(lines) + f"\n    return {seq.result};"


TEMPLATE = r"""
#include <stdint.h>
#include <stdio.h>
static inline uint32_t f(uint32_t x) {{
    {body}
}}
int main(void) {{
    uint64_t bad = 0; uint32_t first = 0;
    for (uint64_t i = 0; i <= 0xFFFFFFFFull; i++) {{
        uint32_t x = (uint32_t)i, want;
        {want}
        if (f(x) != want) {{ if (!bad) first = x; bad++; }}
    }}
    printf("%llu %u\n", (unsigned long long)bad, first);
    return 0;
}}
"""


def check(case):
    op, signed, d = case
    seq = lower(op, signed, d)
    if signed:
        want = f"want = (uint32_t)((int32_t)x {'/' if op == 'div' else '%'} (int32_t){d});"
    else:
        want = f"want = x {'/' if op == 'div' else '%'} {d}u;"
    src = TEMPLATE.format(body=c_expr(seq), want=want)
    with tempfile.TemporaryDirectory() as td:
        c, exe = Path(td) / "v.c", Path(td) / "v"
        c.write_text(src)
        subprocess.run([CC, "-O3", "-march=native", "-o", str(exe), str(c)], check=True)
        t = time.time()
        out = subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout.split()
    bad, first = int(out[0]), int(out[1])
    return case, seq, bad, first, time.time() - t


def _find_cc() -> str:
    """The exhaustive check compiles a tiny C program that loops over all 2^32 inputs, so it
    needs a C compiler: clang, gcc or cc, whichever the machine has (macOS ships clang; most
    Linux installs have gcc)."""
    import shutil

    for cc in (os.environ.get("CC"), "clang", "gcc", "cc"):
        if cc and shutil.which(cc):
            return cc
    sys.exit("verify_divconst_exhaustive.py needs a C compiler (clang or gcc): "
             "`sudo apt install gcc` on Debian/Ubuntu, `xcode-select --install` on macOS.")


CC = _find_cc() if __name__ == "__main__" else "cc"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    survey = json.loads((ROOT / "corpus/sm_80/div_const_survey.json").read_text())
    cases = [(op, s, d) for op in ("div", "rem") for s in (False, True) for d in DIVISORS
             if not (s and d > 0x7FFFFFFE)]
    t0 = time.time()
    rows = []
    with ThreadPoolExecutor(args.jobs) as ex:
        for (op, s, d), seq, bad, first, dt in ex.map(check, cases):
            key = f"{op}.{'s' if s else 'u'}32.{d}"
            ptx = survey.get(key, {})
            rows.append({"case": key, "kind": seq.kind, "ours": seq.n, "ptxas": ptx.get("extra"),
                         "ptxas_xu": ptx.get("xu"), "proof": "exhaustive" if bad == 0 else f"FAILS {bad} (x={first})",
                         "sass": to_sass(seq)})
    (ROOT / "corpus/sm_80/divconst_proofs.json").write_text(json.dumps(rows, indent=1))
    ok = sum(r["proof"] == "exhaustive" for r in rows)
    print(f"{ok}/{len(rows)} sequences proved over all 2^32 inputs in {time.time() - t0:.0f}s")
    print(f"{'case':22} {'kind':14} {'ours':>4} {'ptxas':>5} {'XU':>3}  proof")
    for r in rows:
        print(f"{r['case']:22} {r['kind']:14} {r['ours']:>4} {r['ptxas']!s:>5} {r['ptxas_xu']!s:>3}  {r['proof']}")


if __name__ == "__main__":
    main()
