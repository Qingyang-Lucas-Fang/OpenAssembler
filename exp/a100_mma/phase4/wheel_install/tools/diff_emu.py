"""Differential harness: SASS emulator (on ptxas output) vs PTX interpreter.

Usage: uv run --no-project python tools/diff_emu.py [sm_80] [--trials N] [--only SUBSTR]

For each probe: random inputs (mixing edge values) -> run both -> compare the
8 output bytes per thread. Lanes where the PTX result is spec-undefined are
skipped; lanes whose result came from an `.approx` op are compared within a
ulp tolerance. Writes corpus/<arch>/diff_report.json and prints a summary.
"""

from __future__ import annotations

import argparse
import json
import random
import struct
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ptx_probes  # noqa: E402
from openassembler.emu.machine import EmuError, Memory, Unsupported  # noqa: E402
from openassembler.emu.probe import IN_BASE, OUT_BASE, cubin_consts, run_probe  # noqa: E402
from openassembler.ptx.interp import Interp, Launch, NotModeled  # noqa: E402
from openassembler.ptx.parse import parse_module  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EDGE32 = [0, 1, 2, 3, 7, 31, 32, 33, 0x7F, 0x80, 0xFF, 0xFFFF, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF,
          0xFFFFFFFE, 0x1234, 0x3F800000, 0xBF800000, 0x7F800000, 0xFF800000, 0x7FC00000,
          0x00000001, 0x007FFFFF, 0x80800000, 0x4B000000, 0x5F000000, 0xCF000000]
ULP_TOL = 4  # for .approx results; the emulator models MUFU as correctly rounded
ABS_TOL = {"sin.approx.f32": 2 ** -14.7, "cos.approx.f32": 2 ** -14.7}  # PTX spec, |x| <= 100*pi
REL_TOL = {"tanh.approx.f32": 2 ** -11}  # PTX spec: max relative error 2^-11 over the full range


def is_nan32(x: int) -> bool:
    return (x & 0x7F800000) == 0x7F800000 and bool(x & 0x7FFFFF)


def f32v(x: int) -> float:
    return struct.unpack("<f", struct.pack("<I", x))[0]


def is_nan64(lo: int, hi: int) -> bool:
    return (hi & 0x7FF00000) == 0x7FF00000 and bool((hi & 0xFFFFF) | lo)


def same(probe: str, g, e, approx: bool) -> bool:
    # f64 results occupy both output words: any NaN equals any NaN (PTX: "unspecified NaN").
    # The A100 keeps the input NaN's payload (gpu_a100_40gb_v2), the spec interpreter makes a fresh one.
    if probe.endswith("f64") and len(g) == 2 and is_nan64(*g) and is_nan64(*e):
        return True
    for x, y in zip(g, e):
        if x == y or (is_nan32(x) and is_nan32(y)):
            continue  # PTX: NaN results are "unspecified NaN" (payload/sign free)
        if not approx:
            return False
        if probe in REL_TOL and not (is_nan32(x) or is_nan32(y)):
            fx, fy = f32v(x), f32v(y)
            if abs(fx - fy) <= REL_TOL[probe] * max(abs(fy), 2 ** -126):
                continue
            return False
        if probe in ABS_TOL and not (is_nan32(x) or is_nan32(y)):
            if abs(f32v(x) - f32v(y)) <= ABS_TOL[probe]:
                continue
            return False
        if not ulp_close(x, y):
            return False
    return True


def rand32(rng: random.Random) -> int:
    r = rng.random()
    if r < 0.25:
        return rng.choice(EDGE32)
    if r < 0.45:
        return rng.getrandbits(rng.choice([3, 5, 8, 16]))
    return rng.getrandbits(32)


def run_ptx(src: str, kernel: str, inputs, k: int, kf: int):
    mod = parse_module(src)
    gmem = Memory()
    for t, words in enumerate(inputs):
        gmem.write_bytes(IN_BASE + 16 * t, struct.pack("<4I", *words))
    it = Interp(mod, kernel, Launch(block=(len(inputs), 1, 1),
                                     params={"out": OUT_BASE, "in": IN_BASE, "k": k, "kf": kf}), gmem)
    it.run()
    outs = []
    for t, ln in enumerate(it.warps[0] if len(it.warps) == 1 else [l for w in it.warps for l in w]):
        approx = bool(ln.regs.get("%%approx"))
        outs.append(((gmem.read(OUT_BASE + 16 * t, 4), gmem.read(OUT_BASE + 16 * t + 4, 4)),
                     ln.undefined, approx))
    return outs


def ulp_close(a: int, b: int) -> bool:
    if a == b:
        return True
    # compare as ordered f32 integers; NaN vs NaN of any payload counts as equal
    nan = lambda x: (x & 0x7F800000) == 0x7F800000 and x & 0x7FFFFF  # noqa: E731
    if nan(a) and nan(b):
        return True
    key = lambda x: x if x < 0x80000000 else 0x80000000 - x  # noqa: E731
    return abs(key(a) - key(b)) <= ULP_TOL


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arch", nargs="?", default="sm_80")
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--only", default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    corpus = ROOT / "corpus" / args.arch
    rng = random.Random(args.seed)

    report, status = {}, Counter()
    for name, _body, _shared in ptx_probes.probes(args.arch):
        if args.only and args.only not in name:
            continue
        kname = ptx_probes.kernel_name(name)
        sass = corpus / "sass" / f"{kname}.sass"
        ptx = corpus / "ptx" / f"{kname}.ptx"
        if not sass.exists():
            report[name] = {"status": "no-sass"}
            status["no-sass"] += 1
            continue
        sass_text, ptx_text = sass.read_text(), ptx.read_text()
        cub = corpus / "cubin" / f"{kname}.cubin"
        consts = cubin_consts(cub.read_bytes()) if cub.exists() else {}
        lanes = mism = skipped = 0
        first = None
        try:
            for _ in range(args.trials):
                inputs = [tuple(rand32(rng) for _ in range(4)) for _ in range(32)]
                k, kf = rand32(rng), rand32(rng)
                got = run_probe(sass_text, inputs, k, kf, args.arch, consts=consts)
                exp = run_ptx(ptx_text, kname, inputs, k, kf)
                for t, (g, (e, undef, approx)) in enumerate(zip(got, exp)):
                    if undef:
                        skipped += 1
                        continue
                    lanes += 1
                    ok = same(name, g, e, approx)
                    if not ok:
                        mism += 1
                        if first is None:
                            first = {"in": [f"{w:#x}" for w in inputs[t]], "k": f"{k:#x}", "kf": f"{kf:#x}",
                                     "sass": [f"{w:#x}" for w in g], "ptx": [f"{w:#x}" for w in e]}
        except Unsupported as e:
            report[name] = {"status": "sass-unsupported", "why": str(e)[:120]}
            status["sass-unsupported"] += 1
            continue
        except NotModeled as e:
            report[name] = {"status": "ptx-unsupported", "why": str(e)[:120]}
            status["ptx-unsupported"] += 1
            continue
        except EmuError as e:
            report[name] = {"status": "sass-error", "why": str(e)[:120]}
            status["sass-error"] += 1
            continue
        st = "match" if mism == 0 else "MISMATCH"
        status[st] += 1
        report[name] = {"status": st, "lanes": lanes, "mismatches": mism, "skipped_undefined": skipped}
        if first:
            report[name]["first"] = first

    out = corpus / "diff_report.json"
    out.write_text(json.dumps(report, indent=1))
    total = sum(status.values())
    print(f"{args.arch}: {total} probes  " + "  ".join(f"{k}={v}" for k, v in status.most_common()))
    for name, r in report.items():
        if r["status"] == "MISMATCH":
            print(f"  MISMATCH {name}: {r['mismatches']}/{r['lanes']} lanes, e.g. {r['first']}")
    print(f"-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
