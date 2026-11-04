"""Replay recorded GPU outputs against the current emulator (offline, no GPU, $0).

    PYTHONPATH=. uv run --no-project python tools/gpu_replay.py corpus/sm_80/gpu_a100_40gb_tensor.raw.json.gz
    ... --only mma.f16 --show 3

Every Modal/SSH run writes <result>.raw.json.gz (inputs + per-thread GPU outputs).
This re-runs the emulator on exactly those inputs, so emulator changes can be checked
against hardware without paying for another run.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

from openassembler.emu.probe import cubin_consts, run_probe  # noqa: E402


def replay(raw_path: Path, only: str = "", show: int = 0, which_sass: str = "sass") -> dict:
    raw = json.load(gzip.open(raw_path, "rt"))
    arch = raw["arch"]
    corpus = ROOT / "corpus" / arch
    out = {}
    for name, rec in raw["probes"].items():
        if only and only not in name:
            continue
        sass = (corpus / which_sass / f"{rec['kname']}.sass").read_text()
        cub = corpus / ("ours" if which_sass == "ours" else "cubin") / f"{rec['kname']}.cubin"
        consts = cubin_consts(cub.read_bytes()) if cub.exists() else {}
        lanes = bad = 0
        shown = 0
        for (inputs, k, kf), gpu in zip(rec["trials"], rec["out"]):
            try:
                emu = run_probe(sass, [tuple(i) for i in inputs], k, kf, arch, consts=consts)
            except Exception as e:  # noqa: BLE001
                out[name] = {"status": "emu-error", "why": str(e)[:80]}
                break
            for t in range(len(gpu)):
                lanes += 1
                if list(emu[t]) != list(gpu[t]):
                    bad += 1
                    if shown < show:
                        shown += 1
                        print(f"  {name} t{t} in={[hex(x) for x in inputs[t]]} "
                              f"gpu={[hex(x) for x in gpu[t]]} emu={[hex(x) for x in emu[t]]}")
        else:
            out[name] = {"status": "match" if bad == 0 else "DIFFER", "lanes": lanes, "bad": bad}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("--only", default="")
    ap.add_argument("--show", type=int, default=0)
    ap.add_argument("--sass", default="sass", help="corpus subdir holding the listings (sass | ours)")
    a = ap.parse_args()
    import gpu_common

    # Probes recorded before they were fixed (racy shared-memory stores, ldmatrix reading
    # uninitialised smem): superseded by the fix5 run, and exempted by tests/test_gpu_truth.py.
    STALE = {"st.shared.b32", "st.shared.b64", "st.shared.v2.b32", "st.shared.v4.b32", "ldmatrix.trans"}
    res = replay(Path(a.raw), a.only, a.show, a.sass)
    print("columns: probe, status, mismatching lanes / lanes compared. This tool compares BIT-EXACTLY;")
    print("rows marked (expected) differ for a known reason and are not failures.\n")
    n_match = n_expected = n_other = 0
    for name, r in res.items():
        why = gpu_common.expected_reason(name) or ("recorded with an old probe, since fixed" if name in STALE else None)
        tag = f"(expected: {why})" if r["status"] != "match" and why else r.get("why", "")
        print(f"{name:20} {r['status']:9} {r.get('bad', '')}/{r.get('lanes', '')} {tag}")
        if r["status"] == "match":
            n_match += 1
        elif why:
            n_expected += 1
        else:
            n_other += 1
    print(f"\n{len(res)} probes: {n_match} bit-exact, {n_expected} differ for an expected reason "
          f"(approximate instructions, clocks/ids, racy probes), {n_other} unexplained")
    print("`pytest tests/test_gpu_truth.py` is the pass/fail check: it applies the tolerances and exclusions.")


if __name__ == "__main__":
    main()
