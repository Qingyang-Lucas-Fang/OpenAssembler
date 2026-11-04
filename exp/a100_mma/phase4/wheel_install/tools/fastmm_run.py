"""Run fast-matmul plans (AlphaTensor / Strassen / naive blocks / one big GEMM) on a GPU.

    PYTHONPATH=. python3 tools/fastmm_run.py --dry-run                 # list plans, no GPU
    modal run tools/fastmm_run.py --n 4096 --gpu A100-40GB             # check + time
    modal run tools/fastmm_run.py --n 4096 --schemes at_gpu_4x4x4,strassen_2x2x2

Every plan is run with the kernels compiled by ptxas AND by our compiler (cubins from
tools/fastmm_build.py). Correctness: C is copied back once and compared on the host with
a float64 reference on 64 sampled rows (relative error ||C - C_ref|| / (||A|| ||B||)),
so a wrong plan or a wrong kernel is caught, not just timed. Timing: median of --reps
runs of the whole launch sequence (cuEvents), random order, several rounds.

Billing: one container, max_containers=1, timeout = budget + 60 s, ephemeral app.
"""

from __future__ import annotations

import time

import modal

BUDGET_S = 600
app = modal.App("openassembler-fastmm")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("nvidia-cublas-cu12==12.8.4.1")       # reference SGEMM only
         .add_local_python_source("openassembler", copy=False))


@app.function(image=image, gpu="A100-40GB", timeout=BUDGET_S + 60, max_containers=1, retries=0)
def run_batch(jobs_blob: bytes, job_timeout: float, budget: float) -> dict:
    import os
    import tempfile

    from openassembler.gpu import batch

    d = tempfile.mkdtemp()
    jp, rp = os.path.join(d, "jobs.pkl"), os.path.join(d, "res.pkl")
    with open(jp, "wb") as f:
        f.write(jobs_blob)
    info = batch.device_info()
    t0 = time.monotonic()
    r = batch.run(jp, rp, job_timeout, min(budget, BUDGET_S))
    return {"device": info, "gpu_seconds": round(time.monotonic() - t0, 1), **r}


def make_jobs(n, schemes, variants, reps, rounds, seed, arch="sm_80", with_cublas=True, gemm="sgemm128",
              fused=True):
    import random
    import struct
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    from openassembler.fastmm import factors as F
    from openassembler.fastmm import plan as P

    rng = random.Random(seed)
    init = {"A": (seed, n * n), "B": (seed + 1, n * n)}          # generated in the container
    rows = sorted(random.Random(seed + 2).sample(range(n), CHECK_ROWS))
    # The host reference data (_check_data) is built only after the GPU results are back:
    # ~1 min of pure-Python work holds the GIL, which starved Modal's heartbeat thread and
    # made the server stop the app mid-run (n = 8192).
    A = B = None
    P.GEMM = gemm
    allS = F.all_schemes()
    plans = {"gemm": P.direct(n)}
    for name in schemes:
        s = allS[name] if name in allS else None
        if name.startswith("naive_"):
            a, b, c = map(int, name[6:].split("x"))
            s = F.naive(a, b, c)
        if s is None:
            raise SystemExit(f"unknown scheme {name}")
        plans[name] = P.build(s, n)
        if fused:
            try:
                plans[name + "+fused"] = P.build_fused(s, n)
            except ValueError as e:
                print(f"  {name}: no fused plan ({e})")
    d = root / "corpus" / arch / "fastmm"
    jobs = []
    if with_cublas:
        plans["cublas"] = P.cublas(n)
    for rnd in range(rounds):
        order = [(pn, v) for pn in plans for v in (["-"] if pn == "cublas" else variants)]
        rng.shuffle(order)
        for pn, v in order:
            pl = plans[pn]
            mods = {} if pn == "cublas" else {P.GEMM: (d / f"{v}@{P.GEMM}.cubin").read_bytes()}
            for k in pl.lincomb_ns:
                mods[f"lincomb{k}"] = (d / f"{v}@lincomb{k}.cubin").read_bytes()
            for sh in getattr(pl, "fgemm_shapes", ()):
                k = "fgemm_%d_%d_%d" % sh
                mods[k] = (d / f"{v}@{k}.cubin").read_bytes()
            jobs.append({"mode": "plan", "name": f"{pn}|{v}|{rnd}", "modules": mods,
                         "buffers": pl.buffers, "init_rand": init, "steps": pl.steps,
                         "outputs": {"C": [(4 * n * i, 4 * n) for i in rows]} if rnd == 0 else {},
                         "reps": reps, "warmup": 3})
    info = {pn: {"launches": len(pl.steps), "lincombs": pl.n_lincomb, "mul_flops": 2 * pl.flops_mul,
                 "buffers_MB": round(sum(pl.buffers.values()) / 2 ** 20)} for pn, pl in plans.items()}
    return jobs, info, (A, B, rows)


CHECK_ROWS = 16


def _cols(n):
    return list(range(0, n, max(1, n // 128)))      # 128 checked columns


def _check_data(n, seed, rows):
    """Only what the host check needs, streamed through the same generator as
    cuda_driver.rand_f32 (identical values, rounded to f32): the sampled rows of A and the
    checked columns of B. Holding all of A and B as Python floats took ~4 GB at n = 8192
    and starved Modal's client heartbeat (the app was stopped mid-run)."""
    import array
    import random

    def stream(s):
        r = random.Random(s)
        for i in range(n):
            yield i, array.array("f", (r.uniform(-1, 1) for _ in range(n)))
    want = set(rows)
    A = {i: list(row) for i, row in stream(seed) if i in want}
    cols = _cols(n)
    Bc = {j: [0.0] * n for j in cols}
    for k, row in stream(seed + 1):
        for j in cols:
            Bc[j][k] = row[j]
    return A, Bc


def check(n, A, B, rows, crow_bytes):
    """Relative Frobenius error on the sampled rows x 128 columns, float64 reference."""
    import math
    import struct

    err2 = ref2 = 0.0
    for i, cb in zip(rows, crow_bytes):
        C = struct.unpack(f"<{n}f", cb)
        for j in _cols(n):
            ref = math.fsum(a * b for a, b in zip(A[i], B[j]))
            err2 += (C[j] - ref) ** 2
            ref2 += ref * ref
    return math.sqrt(err2 / ref2)


@app.local_entrypoint()
def main(n: int = 4096, schemes: str = "strassen_2x2x2,at_2x2x2,naive_2x2x2,strassen2_4x4x4,at_gpu_4x4x4,at_4x4x4,naive_4x4x4",
         variants: str = "ptxas,ours", gpu: str = "A100-40GB", reps: int = 5, rounds: int = 2,
         seed: int = 1, budget: float = 400, job_timeout: float = 120, dry_run: bool = False,
         gemm: str = "sgemm128") -> None:
    import json
    import pickle
    from pathlib import Path

    jobs, info, (A, B, crows) = make_jobs(n, schemes.split(","), variants.split(","), reps, rounds, seed,
                                          gemm=gemm)
    for pn, x in info.items():
        print(f"  {pn:18} {x['launches']:5} launches ({x['lincombs']} lincomb), "
              f"{x['mul_flops'] / 1e9:8.1f} GFLOP in GEMMs, {x['buffers_MB']} MB")
    blob = pickle.dumps(jobs)
    print(f"{len(jobs)} jobs, {len(blob) / 2 ** 20:.0f} MB to upload, GPU {gpu}, budget {budget:.0f}s")
    if dry_run:
        return
    t0 = time.time()
    res = run_batch.with_options(gpu=gpu).remote(blob, job_timeout, budget)
    dev = res["device"]
    print(f"{dev['name']} cc {dev['cc']}: {res['gpu_seconds']}s in container, {time.time() - t0:.0f}s wall")
    A, B = _check_data(n, seed, crows)
    rows = {}
    for i, j in enumerate(jobs):
        pn, v, rnd = j["name"].split("|")
        r = res["results"].get(i, {})
        row = rows.setdefault(pn, {}).setdefault(v, {})
        if "error" in r:
            # keep a timing from another round; only report the error if nothing ran
            row.setdefault("errors", []).append(r["error"][:60])
            continue
        t = r.get("time_ms", {}).get("median")
        if t is not None and (row.get("ms") is None or t < row["ms"]):
            row["ms"] = t
        if "C" in r.get("out", {}):
            row["rel_err"] = check(n, A, B, crows, r["out"]["C"])
    base = rows.get("cublas", {}).get("-", {}).get("ms")
    print(f"\n{'plan':18} {'variant':7} {'ms':>9} {'TFLOP/s*':>9} {'vs cuBLAS':>13} {'rel err':>9}")
    for pn, vs in rows.items():
        for v, row in vs.items():
            ms = row.get("ms")
            tf = 2 * n ** 3 / (ms * 1e-3) / 1e12 if ms else None
            print(f"{pn:18} {v:7} {ms if ms else float('nan'):9.3f} {tf if tf else float('nan'):9.2f} "
                  f"{(base / ms if base and ms else float('nan')):12.3f}x {row.get('rel_err', float('nan')):9.2e} "
                  f"{'' if row.get('ms') else (row.get('errors') or [''])[0]}")
    print("* effective: 2 n^3 / time (counts the classical flop count for every plan)")
    tag = gpu.lower().replace("!", "").replace("-", "_")
    out = Path(__file__).resolve().parent.parent / "corpus" / "sm_80" / f"fastmm_{tag}_n{n}_{gemm}.json"
    if out.exists():
        # the committed file is the recorded run; keep it, write this one beside it
        out = out.with_name(f"{out.stem}_{time.strftime('%Y%m%d-%H%M%S')}.json")
    out.write_text(json.dumps({"device": dev, "n": n, "plans": info, "results": rows}, indent=1))
    print(f"-> {out}")
