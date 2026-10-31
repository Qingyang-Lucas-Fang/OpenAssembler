"""Local half of GPU validation, shared by every transport (Modal, SSH, local).

build_jobs():   choose cubins + random inputs (same generator as tools/diff_emu.py)
compare():      GPU outputs vs SASS emulator (on the same ptxas cubin) and vs the
                PTX interpreter; classifies every probe and writes JSON.

Probe sets:
  ptxas   corpus/<arch>/cubin/*.cubin    (validates the emulator and our hypotheses)
  ours    corpus/<arch>/ours/*.cubin     (cubins produced by OUR compiler)
"""

from __future__ import annotations

import json
import random
import struct
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

import diff_emu  # noqa: E402
import ptx_probes  # noqa: E402
from openassembler.emu.machine import EmuError  # noqa: E402
from openassembler.emu.probe import run_probe  # noqa: E402

THREADS = 32

# Probes whose GPU output may legitimately differ from the emulator, and why.
EXPECTED = {
    "approx": ("approx", "sin.", "cos.", "ex2", "lg2", "tanh", "rsqrt", "div.full"),
    "nondeterministic": ("clock", "globaltimer", "smid", "gridid", "warpid", "nsmid", "nwarpid",
                         "cvta.shared", "isspacep", "dynamic_smem", "total_smem"),
}


def expected_reason(name: str) -> str | None:
    for why, keys in EXPECTED.items():
        if any(k in name for k in keys):
            return why
    return None


def _finite(rng: random.Random, fmt: str) -> int:
    """Finite value bits with exponents spread enough to exercise alignment/rounding:
    70% in [2^-6, 2^6], 25% in [2^-20, 2^20], 5% exact small integers / zero."""
    r = rng.random()
    if r < 0.05:
        x = float(rng.randint(-4, 4))
    else:
        span = 6 if r < 0.75 else 20
        x = rng.uniform(1, 2) * 2.0 ** rng.randint(-span, span) * rng.choice((1, -1))
    if fmt == "f16":
        span16 = min(span if r >= 0.05 else 0, 12)
        x = rng.uniform(1, 2) * 2.0 ** rng.randint(-span16, span16) * rng.choice((1, -1)) if r >= 0.05 else x
        return struct.unpack("<H", struct.pack("<e", x))[0]
    if fmt == "bf16":
        return struct.unpack("<I", struct.pack("<f", x))[0] >> 16
    if fmt == "f32":
        return struct.unpack("<I", struct.pack("<f", x))[0]
    raise ValueError(fmt)


def structured_inputs(probe: str, rng: random.Random):
    """Per-thread 4 x u32 inputs whose bit patterns are meaningful for `probe`,
    or None to use the generic random generator."""
    if probe in ("mma.f16", "mma.bf16"):
        f = probe.split(".")[1]
        return [[_finite(rng, f) | (_finite(rng, f) << 16) for _ in range(4)] for _ in range(THREADS)]
    if probe == "mma.tf32":
        return [[_finite(rng, "f32") for _ in range(4)] for _ in range(THREADS)]
    if probe == "mma.f64":
        rows = []
        for _ in range(THREADS):
            w = []
            for _ in range(2):
                x = rng.uniform(1, 2) * 2.0 ** rng.randint(-30, 30) * rng.choice((1, -1))
                b = struct.unpack("<Q", struct.pack("<d", x))[0]
                w += [b & 0xFFFFFFFF, b >> 32]
            rows.append(w)
        return rows
    return None


def build_jobs(arch: str, which: str, only: str, trials: int, seed: int, limit: int = 0):
    """which: "ptxas" | "ours" | "dir:<path>". A dir holds files named
    "<variant>@<kernel>.cubin"; each is checked against the emulator running
    ptxas's SASS for <kernel> (any correct cubin must compute the same thing)."""
    corpus = ROOT / "corpus" / arch
    rng = random.Random(seed)
    jobs = []
    if which.startswith("dir:"):
        names = {ptx_probes.kernel_name(n): n for n, _b, _s in ptx_probes.probes(arch)}
        try:
            import bench_kernels
            bench = {ptx_probes.kernel_name(n): n for n in bench_kernels.BENCH}
        except ImportError:
            bench = {}
        names.update(bench)
        try:
            import lat_sweep
            bench.update({lat_sweep.kname(n): "lat." + n for n in lat_sweep.SWEEP})
        except ImportError:
            pass
        names.update(bench)
        for f in sorted((ROOT / which[4:]).glob("*.cubin")):
            kname = f.stem.split("@")[-1]
            if kname not in names or (only and only not in f.stem):
                continue
            ts = []
            for _ in range(trials):
                inputs = [[diff_emu.rand32(rng) for _ in range(4)] for _ in range(THREADS)]
                # bench kernels loop k times: keep correctness trials short
                k = rng.randrange(1, 64) if kname in bench else diff_emu.rand32(rng)
                ts.append((inputs, k, diff_emu.rand32(rng)))
            jobs.append({"name": f.stem, "probe": names[kname], "kname": kname, "cubin": f.read_bytes(),
                         "trials": ts})
        return jobs
    src = {"ptxas": corpus / "cubin", "ours": corpus / "ours"}[which]
    # `only`: substring, or a comma-separated list of exact probe names ("a,b,c")
    exact = set(only.split(",")) if "," in only else None
    for name, _b, _s in ptx_probes.probes(arch):
        if exact is not None:
            if name not in exact:
                continue
        elif only and only not in name:
            continue
        kname = ptx_probes.kernel_name(name)
        cubin = src / f"{kname}.cubin"
        if not cubin.exists():
            continue
        ts = []
        for _ in range(trials):
            inputs = structured_inputs(name, rng) or [[diff_emu.rand32(rng) for _ in range(4)] for _ in range(THREADS)]
            ts.append((inputs, diff_emu.rand32(rng), diff_emu.rand32(rng)))
        jobs.append({"name": name, "kname": kname, "cubin": cubin.read_bytes(), "trials": ts})
        if limit and len(jobs) >= limit:
            break
    return jobs


def _emu(arch, which, kname, inputs, k, kf):
    corpus = ROOT / "corpus" / arch
    if which.startswith("dir:") and (ROOT / which[4:] / f"ptxas@{kname}.sass").exists():
        sass_path = ROOT / which[4:] / f"ptxas@{kname}.sass"      # bench dir: ptxas's own listing
    else:
        sass_path = corpus / ("ours" if which == "ours" else "sass") / f"{kname}.sass"
    return run_probe(sass_path.read_text(), [tuple(i) for i in inputs], k, kf, arch)


def _ptx_path(arch, which, kname):
    if which.startswith("dir:") and (ROOT / which[4:] / f"{kname}.ptx").exists():
        return ROOT / which[4:] / f"{kname}.ptx"
    return ROOT / "corpus" / arch / "ptx" / f"{kname}.ptx"


def compare(arch: str, which: str, jobs, gpu_res: dict, device: dict, out_path: Path) -> Counter:
    """gpu_res: {job index: {out|error, attrs, oob_bytes}}"""
    corpus = ROOT / "corpus" / arch
    status, rows = Counter(), {}
    for idx, job in enumerate(jobs):
        name, kname, trials = job["name"], job["kname"], job["trials"]
        probe = job.get("probe", name)
        r = gpu_res.get(idx, {"error": "missing"})
        row = {}
        if "error" in r:
            st, row = "gpu-error", {"error": r["error"]}
        else:
            row["attrs"] = r.get("attrs")
            if any(r.get("oob_bytes", [])):
                row["oob_bytes"] = r["oob_bytes"]
            try:
                emu = [_emu(arch, which, kname, *t) for t in trials]
            except EmuError as e:
                emu = None
                row["emu"] = str(e)[:80]
            # PTX-level reference (spec semantics); None where not modeled
            try:
                ptxref = [diff_emu.run_ptx(_ptx_path(arch, which, kname).read_text(), kname, *t) for t in trials]
            except Exception as e:  # noqa: BLE001
                ptxref = None
                row["ptx"] = str(e)[:80]
            gpu = r["out"]
            lanes = [(t, i) for t in range(len(trials)) for i in range(THREADS)]
            if emu is None:
                st = "gpu-only"
            else:
                bad = [(t, i) for t, i in lanes if list(gpu[t][i]) != list(emu[t][i])]
                why = expected_reason(probe)
                # approx ops: compare to the emulator within the harness tolerance
                if bad and why == "approx":
                    bad = [(t, i) for t, i in bad
                           if not diff_emu.same(probe, tuple(gpu[t][i]), tuple(emu[t][i]), True)]
                if not bad:
                    st = "match"
                else:
                    st = f"differ-{why}" if why else "DIFFER"
                    t, i = bad[0]
                    row.update({"lanes": f"{len(bad)}/{len(lanes)}",
                                "in": [hex(x) for x in trials[t][0][i]], "k": hex(trials[t][1]),
                                "kf": hex(trials[t][2]), "gpu": [hex(x) for x in gpu[t][i]],
                                "emu": [hex(x) for x in emu[t][i]]})
            if ptxref is not None:
                pbad = 0
                for t, i in lanes:
                    e, undef, approx = ptxref[t][i]
                    if not undef and not diff_emu.same(probe, tuple(gpu[t][i]), tuple(e), approx):
                        pbad += 1
                row["vs_ptx_spec"] = "match" if not pbad else f"{pbad}/{len(lanes)} differ"
        status[st] += 1
        rows[name] = {"status": st, **row}
    # Raw ground truth (inputs + GPU outputs) for offline replay: tools/gpu_replay.py
    import gzip
    raw = {job["name"]: {"kname": job["kname"], "probe": job.get("probe", job["name"]),
                         "trials": job["trials"], "out": gpu_res[i]["out"]}
           for i, job in enumerate(jobs) if "out" in gpu_res.get(i, {})}
    with gzip.open(out_path.with_suffix(".raw.json.gz"), "wt") as f:
        json.dump({"device": device, "set": which, "arch": arch, "probes": raw}, f)
    out_path.write_text(json.dumps({"device": device, "set": which, "arch": arch,
                                    "trials": len(jobs[0]["trials"]) if jobs else 0,
                                    "summary": dict(status), "probes": rows}, indent=1))
    return status


def print_summary(status: Counter, out_path: Path) -> None:
    rows = json.loads(out_path.read_text())["probes"]
    print("  ".join(f"{k}={v}" for k, v in status.most_common()))
    for name, row in rows.items():
        if row["status"] in ("DIFFER", "gpu-error") or (row.get("vs_ptx_spec", "match") != "match"
                                                         and row["status"] == "match"):
            print(f"  {row['status']:10} {name:24} {row.get('lanes', '')} {row.get('error', '')} "
                  f"in={row.get('in', '')} gpu={row.get('gpu', '')} emu={row.get('emu', '')} "
                  f"spec={row.get('vs_ptx_spec', '')}")
    print(f"-> {out_path.relative_to(ROOT)}")


def pack(x: int) -> bytes:
    return struct.pack("<I", x)


ORDER = ("safe", "mined", "sched", "opt", "optY", "optN")


def report_times(jobs, gpu_res, device, out_path) -> dict:
    """Timing table: per kernel, per variant (file prefix before '@'), median ms and
    speed-up vs ptxas. Writes JSON next to the correctness result."""
    rows = {}
    for i, j in enumerate(jobs):
        if j.get("mode") != "time":
            continue
        r = gpu_res.get(i, {})
        variant, _, kname = j["name"].split("#time")[0].partition("@")
        v = r.get("time_ms", {}).get("median") if "time_ms" in r else f"error: {r.get('error', '?')[:40]}"
        old = rows.setdefault(kname, {}).get(variant)
        # best (lowest) median over rounds; an error in any round is kept visible
        if old is None or (isinstance(old, float) and isinstance(v, float) and v < old) or isinstance(v, str):
            rows[kname][variant] = v
    out_path.write_text(json.dumps({"device": device, "kernels": rows}, indent=1))
    variants = ["ptxas"] + sorted({x for v in rows.values() for x in v} - {"ptxas"},
                                  key=lambda x: ORDER.index(x) if x in ORDER else 9)
    best = "opt" if "opt" in variants else variants[-1]
    print(f"\n{'kernel (ms)':22}" + "".join(f"{v:>10}" for v in variants) + f"   ptxas/{best}")
    for k, v in rows.items():
        cells = [v.get(x) for x in variants]
        fmt = lambda c: f"{c:10.4f}" if isinstance(c, float) else f"{str(c)[:9]:>10}"  # noqa: E731
        ratio = (f"{v['ptxas'] / v[best]:.2f}x" if all(isinstance(v.get(x), float) for x in ("ptxas", best))
                 else "-")
        print(f"{k:22}" + "".join(fmt(c) for c in cells) + f"   {ratio}")
    print(f"-> {out_path.relative_to(ROOT)}")
    return rows
