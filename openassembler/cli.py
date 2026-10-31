"""openassembler: compile PTX straight to SASS with our own backend, skipping
ptxas's code generation (but reusing its ELF container -- see README.md
"How it works"), and measure the result against ptxas on real hardware.

    openassembler compile kernel.ptx -o kernel.cubin [--opt-level opt]
    openassembler check --only add.s32
    openassembler bench
    openassembler fastmm build
    openassembler gpu run --ssh user@host
    openassembler status

Every subcommand here is a thin front door onto the scripts in tools/ and the
importable openassembler/ package; run `openassembler <subcommand> --help` for the
full flag set, or call the wrapped tools/*.py script directly if you want one.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"

OPT_LEVELS = ("safe", "mined", "sched", "opt")


def _require_in_repo(path: Path, what: str) -> Path:
    """ptxas/nvdisasm run inside the colima VM (env/cuda.sh), which only
    bind-mounts this repo -- any path they touch must resolve under ROOT."""
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError:
        print(f"error: {what} must be inside the repo ({ROOT}) -- ptxas/nvdisasm "
              f"run in a VM that only sees this tree. Got: {resolved}", file=sys.stderr)
        raise SystemExit(1)
    return resolved


def _run_tool(script: str, argv: list[str]) -> int:
    """Exec tools/<script> with the same interpreter, PYTHONPATH=ROOT."""
    import os

    env = dict(os.environ, PYTHONPATH=str(ROOT))
    return subprocess.run([sys.executable, str(TOOLS / script), *argv], cwd=ROOT, env=env).returncode


def cmd_compile(a: argparse.Namespace) -> int:
    """PTX file -> cubin, using our backend (openassembler.backend.ra) instead of ptxas's
    code generator. Needs ptxas on PATH (or via env/cuda.sh) once, to build the
    ELF container we splice our code into -- see README.md "How it works"."""
    from openassembler import oracle
    from openassembler.backend import ra
    from openassembler.backend.compile import Out, schedule
    from openassembler.backend.sched import list_schedule
    from openassembler.backend.unroll import unroll_loops
    from openassembler.container.cubin import required_regcount, splice_text
    from openassembler.isa.encode import Encoder
    from openassembler.ptx.parse import parse_module

    ptx_path = Path(a.ptx)
    if a.base_cubin is None:
        _require_in_repo(ptx_path, "the PTX file")
    ptx = ptx_path.read_text()
    module = parse_module(ptx)
    if a.kernel:
        if a.kernel not in module.kernels:
            print(f"error: kernel {a.kernel!r} not found; have {list(module.kernels)}", file=sys.stderr)
            return 1
        kname = a.kernel
    else:
        if len(module.kernels) != 1:
            print(f"error: {ptx_path} has {len(module.kernels)} kernels, pass --kernel NAME; "
                  f"have {list(module.kernels)}", file=sys.stderr)
            return 1
        (kname,) = module.kernels
    kernel = module.kernels[kname]

    if a.opt_level == "safe":
        code, st = ra.compile_kernel(kernel)
        labels = st["labels"]
    else:
        # Mirrors tools/bench_build.py:_reschedule_k, the reference pipeline for
        # the mined/sched/opt variants (list_schedule runs on the raw lowered
        # body, before it's split into labels + Out tuples for schedule()).
        k = unroll_loops(kernel) if a.opt_level == "opt" else kernel
        s = ra.Selector(k)
        ops = ra.coalesce(ra.dce(s.run()))
        phys = ra.allocate(ops, s.r1)
        lowered = ra.lower(ops, phys)
        if a.opt_level in ("sched", "opt"):
            lowered = list_schedule(lowered)
        labels, body = {}, []
        for t, vl, r, w in lowered:
            if t.startswith("<label "):
                labels[t[7:-1]] = len(body)
            else:
                body.append((t, vl, r, w))
        code = schedule([Out(t, vl, r, w) for t, vl, r, w in body], labels=labels, model="mined")

    enc = Encoder(a.arch)
    words = ra.encode_code(enc, code, labels)
    nop = enc.encode("NOP")
    words.append(enc.encode_branch("BRA `(.L_x_0)", 16 * len(words), 16 * len(words)))
    while len(words) % 8:
        words.append(nop)
    exits = [16 * i for i, (t, _ctl) in enumerate(code) if t.endswith("EXIT")]
    nregs = required_regcount([t for t, _ctl in code])

    base_path = Path(a.base_cubin) if a.base_cubin else None
    if base_path is None:
        # ptxas runs inside the colima VM, which only bind-mounts this repo, so
        # the scratch cubin has to live under ROOT (not system /tmp).
        import os
        import uuid

        scratch = ROOT / "corpus" / ".tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        base_path = scratch / f"base_{os.getpid()}_{uuid.uuid4().hex[:8]}.cubin"
        try:
            oracle.ptxas(ptx_path, base_path, arch=a.arch)
        except Exception:
            base_path.unlink(missing_ok=True)
            raise
    base = base_path.read_bytes()
    if a.base_cubin is None:
        base_path.unlink(missing_ok=True)
    cubin = splice_text(base, kname, words, nregs, exits)
    out_path = Path(a.out) if a.out else ptx_path.with_suffix(".cubin")
    if a.verify:
        out_path = _require_in_repo(out_path, "the output cubin (needed by --verify's nvdisasm step)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(cubin)
    n_insns = sum(1 for t, _ in code if not t.startswith("NOP") and not t.startswith("BRA `(.L_x_0)"))
    print(f"{out_path}  ({n_insns} instructions, opt-level={a.opt_level}, regs={nregs})")
    if a.verify:
        return cmd_verify(argparse.Namespace(cubin=str(out_path), ptx=str(ptx_path), kernel=kname, trials=a.trials))
    return 0


def cmd_verify(a: argparse.Namespace) -> int:
    """Run a compiled cubin's SASS through the emulator and compare against the
    PTX reference interpreter on random inputs (same check compile_check.py and
    bench_build.py use before trusting a build)."""
    import random

    from openassembler import oracle
    from openassembler.emu.probe import run_probe

    cubin_path = _require_in_repo(Path(a.cubin), "the cubin")
    sass = oracle.run_tool("nvdisasm", "-hex", "-c", str(cubin_path.relative_to(ROOT))).stdout
    ptx = Path(a.ptx).read_text()
    rng = random.Random(0)

    sys.path.insert(0, str(TOOLS))
    import diff_emu  # noqa: E402

    trials = getattr(a, "trials", 6) or 6
    for _ in range(trials):
        inputs = [tuple(diff_emu.rand32(rng) for _ in range(4)) for _ in range(32)]
        k, kf = diff_emu.rand32(rng), diff_emu.rand32(rng)
        got = run_probe(sass, inputs, k, kf)
        if run_probe(sass, inputs, k, kf, poison=0xDEADBEEF) != got:
            print("FAIL: output depends on uninitialized register contents", file=sys.stderr)
            return 1
        exp = diff_emu.run_ptx(ptx, a.kernel, inputs, k, kf)
        for lane, (g, (e, undef, approx)) in enumerate(zip(got, exp)):
            if not undef and not diff_emu.same("verify", g, e, approx):
                print(f"FAIL: lane {lane}: ours {g} vs PTX interpreter {e}", file=sys.stderr)
                return 1
    print(f"OK: {trials} random trials, bit-exact against the PTX reference interpreter")
    return 0


def cmd_check(a: argparse.Namespace) -> int:
    """Compile the full probe corpus, validate with nvdisasm, check semantics
    against the PTX interpreter. Needs the probe corpus (tools/build_corpus.py)
    and ptxas. Wraps tools/compile_check.py."""
    argv = []
    if a.only:
        argv += ["--only", a.only]
    argv += ["--trials", str(a.trials), "--compiler", a.compiler]
    return _run_tool("compile_check.py", argv)


def cmd_bench(a: argparse.Namespace) -> int:
    """Build the microbenchmark cubins at every optimization level
    (safe/mined/sched/opt), each checked in the emulator first. Wraps
    tools/bench_build.py. Pair with `openassembler gpu run` to time them."""
    return _run_tool("bench_build.py", ["--arch", a.arch])


def cmd_fastmm(a: argparse.Namespace) -> int:
    """Fast matrix multiplication: build Strassen/AlphaTensor-scheme GEMM
    kernels with our compiler (tools/fastmm_build.py), or run a plan on a GPU
    (tools/fastmm_run.py, Modal only -- for SSH/local use `gpu run` on the
    cubins fastmm build produces)."""
    if a.fastmm_cmd == "build":
        argv = ["--arch", a.arch, "--variant", a.variant]
        if a.only:
            argv += ["--only", a.only]
        return _run_tool("fastmm_build.py", argv)
    if a.fastmm_cmd == "run":
        argv = ["--n", str(a.n), "--schemes", a.schemes, "--gpu", a.gpu,
                "--reps", str(a.reps), "--rounds", str(a.rounds), "--budget", str(a.budget)]
        if a.dry_run:
            argv.append("--dry-run")
        return subprocess.run(["modal", "run", str(TOOLS / "fastmm_run.py"), *argv], cwd=ROOT).returncode
    return 1


def cmd_gpu(a: argparse.Namespace) -> int:
    """Run probe or built cubins on a real GPU and compare against the
    emulator / PTX spec. Default backend is any machine you can SSH into with
    an NVIDIA driver (no CUDA toolkit needed there) -- this is the
    reproduction path anyone without a cloud GPU account can use. `--backend
    modal` is what produced the numbers recorded in corpus/ for this repo."""
    if a.backend == "modal":
        argv = ["--gpu", a.gpu, "--which", a.which, "--trials", str(a.trials), "--budget", str(a.budget)]
        if a.only:
            argv += ["--only", a.only]
        if a.dry_run:
            argv.append("--dry-run")
        cmd = ["modal", "run", str(TOOLS / "gpu_modal.py"), *argv]
        print("$", " ".join(shlex.quote(c) for c in cmd))
        return subprocess.run(cmd, cwd=ROOT).returncode
    # ssh or local: same script, tools/gpu_run.py
    argv = ["--which", a.which, "--trials", str(a.trials), "--budget", str(a.budget)]
    if a.only:
        argv += ["--only", a.only]
    if a.backend == "local":
        argv.append("--local")
    else:
        argv += ["--ssh", a.backend]
    return _run_tool("gpu_run.py", argv)


def cmd_status(a: argparse.Namespace) -> int:
    """Print the headline, reproducible numbers from docs/STATUS.md."""
    status = ROOT / "docs" / "STATUS.md"
    text = status.read_text()
    lines = text.splitlines()
    # Print the "New since" table (first markdown table in the file) verbatim;
    # it's already the condensed, up-to-date headline.
    start = next(i for i, l in enumerate(lines) if l.startswith("| Area"))
    end = next(i for i in range(start, len(lines)) if not lines[i].startswith("|"))
    print("\n".join(lines[start:end]))
    print(f"\nFull detail, evidence files, and reproduction commands: {status.relative_to(ROOT)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="openassembler", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("compile", help="compile a PTX file to a cubin with our backend")
    pc.add_argument("ptx")
    pc.add_argument("-o", "--out", default=None)
    pc.add_argument("--kernel", default=None, help="kernel name, if the module has more than one")
    pc.add_argument("--arch", default="sm_80")
    pc.add_argument("--opt-level", choices=OPT_LEVELS, default="opt")
    pc.add_argument("--base-cubin", default=None,
                     help="ptxas-built cubin to splice into (default: build one with ptxas now)")
    pc.add_argument("--verify", action="store_true", help="check the result against the PTX interpreter")
    pc.add_argument("--trials", type=int, default=6)
    pc.set_defaults(func=cmd_compile)

    pv = sub.add_parser("verify", help="check a compiled cubin's SASS against the PTX reference interpreter")
    pv.add_argument("cubin")
    pv.add_argument("ptx")
    pv.add_argument("--kernel", required=True)
    pv.add_argument("--trials", type=int, default=6)
    pv.set_defaults(func=cmd_verify)

    pk = sub.add_parser("check", help="compile + validate the full probe corpus (tools/compile_check.py)")
    pk.add_argument("--only", default="")
    pk.add_argument("--trials", type=int, default=6)
    pk.add_argument("--compiler", choices=["ra", "naive"], default="ra")
    pk.set_defaults(func=cmd_check)

    pb = sub.add_parser("bench", help="build the optimization-level benchmark cubins (tools/bench_build.py)")
    pb.add_argument("--arch", default="sm_80")
    pb.set_defaults(func=cmd_bench)

    pf = sub.add_parser("fastmm", help="Strassen/AlphaTensor fast matrix multiplication")
    fsub = pf.add_subparsers(dest="fastmm_cmd", required=True)
    fb = fsub.add_parser("build", help="build the GEMM/lincomb kernels for every scheme")
    fb.add_argument("--arch", default="sm_80")
    fb.add_argument("--variant", choices=OPT_LEVELS, default="opt")
    fb.add_argument("--only", default="")
    fr = fsub.add_parser("run", help="run a fast-matmul plan on a Modal GPU")
    fr.add_argument("--n", type=int, default=4096)
    fr.add_argument("--schemes", default="strassen_2x2x2,at_2x2x2,naive_2x2x2")
    fr.add_argument("--gpu", default="A100-40GB")
    fr.add_argument("--reps", type=int, default=5)
    fr.add_argument("--rounds", type=int, default=2)
    fr.add_argument("--budget", type=float, default=400)
    fr.add_argument("--dry-run", action="store_true")
    pf.set_defaults(func=cmd_fastmm)

    pg = sub.add_parser("gpu", help="run cubins on a real GPU (SSH/local/Modal) and compare")
    gsub = pg.add_subparsers(dest="gpu_cmd", required=True)
    gr = gsub.add_parser("run", help="build jobs, run them on a GPU, compare against the emulator/PTX spec")
    gr.add_argument("--backend", default="local", metavar="local|modal|user@host",
                     help="'local' (this machine's GPU), 'modal' (cloud, needs `modal setup`), "
                          "or an SSH target user@host (any Linux box with an NVIDIA driver)")
    gr.add_argument("--which", default="ptxas", choices=["ptxas", "ours"])
    gr.add_argument("--only", default="")
    gr.add_argument("--trials", type=int, default=4)
    gr.add_argument("--budget", type=float, default=300)
    gr.add_argument("--gpu", default="A100-40GB", help="Modal GPU type (--backend modal only)")
    gr.add_argument("--dry-run", action="store_true", help="print the job count/cost estimate, run nothing")
    gr.set_defaults(func=cmd_gpu, backend_cmd="run")

    ps = sub.add_parser("status", help="print the headline results from docs/STATUS.md")
    ps.set_defaults(func=cmd_status)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "gpu" and args.gpu_cmd != "run":
        parser.error("gpu: only 'run' is implemented")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
