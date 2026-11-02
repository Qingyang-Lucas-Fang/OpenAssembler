"""openassembler: compile PTX straight to SASS with our own backend, skipping
ptxas's code generation (but reusing its ELF container -- see README.md
"How it works"), and measure the result against ptxas on real hardware.

    openassembler compile kernel.ptx|kernel.py|kernel.cu -o out.cubin   # also writes out.sass
    openassembler corpus                        # build the probe corpus (once)
    openassembler check --only add.s32
    openassembler bench
    openassembler fastmm build
    openassembler gpu run --backend user@host   # or: local | modal
    openassembler status
    openassembler                               # no arguments: interactive shell

Every subcommand here is a thin front door onto the scripts in tools/ and the
importable openassembler/ package; run `openassembler <subcommand> --help` for the
full flag set, or call the wrapped tools/*.py script directly if you want one.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"

OPT_LEVELS = ("safe", "mined", "sched", "opt")


def _run_tool(script: str, argv: list[str]) -> int:
    """Exec tools/<script> with the same interpreter, PYTHONPATH=ROOT."""
    import os

    env = dict(os.environ, PYTHONPATH=str(ROOT))
    return subprocess.run([sys.executable, str(TOOLS / script), *argv], cwd=ROOT, env=env).returncode


def cmd_compile(a: argparse.Namespace) -> int:
    """Source file -> cubin, using our backend (openassembler.backend.ra) instead of
    ptxas's code generator. Accepts .ptx (as-is), .py (openassembler.pyfront's small
    numeric subset), or .cu/.cpp (via nvcc --ptx, openassembler.cppfront) -- all three
    end up as PTX text handed to the same backend. Needs ptxas on PATH (or via
    env/cuda.sh) once, to build the ELF container we splice our code into --
    see README.md "How it works"."""
    from openassembler import oracle
    from openassembler.backend import ra
    from openassembler.backend.compile import Out, schedule
    from openassembler.backend.sched import list_schedule
    from openassembler.backend.unroll import unroll_loops
    from openassembler.container.cubin import required_regcount, splice_text
    from openassembler.frontend import FrontendError, to_ptx
    from openassembler.isa.encode import EncodeError, Encoder
    from openassembler.ptx.parse import parse_module

    src_path = Path(a.ptx)
    if not src_path.exists():
        print(f"error: {src_path} does not exist", file=sys.stderr)
        return 1
    try:
        ptx, kname = to_ptx(src_path, entry=a.kernel, arch=a.arch)
    except FrontendError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except Exception as e:  # pyfront.NotSupported / cppfront.CppFrontendError
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    module = parse_module(ptx)
    if kname is None:  # .ptx with no --kernel: fall back to the single-kernel rule
        if len(module.kernels) != 1:
            print(f"error: {src_path} has {len(module.kernels)} kernels, pass --kernel NAME; "
                  f"have {list(module.kernels)}", file=sys.stderr)
            return 1
        (kname,) = module.kernels
    if kname not in module.kernels:
        print(f"error: kernel {kname!r} not found; have {list(module.kernels)}", file=sys.stderr)
        return 1
    kernel = module.kernels[kname]
    if a.dump_ptx:
        print(ptx)

    def generate():
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
        return code, labels, words, exits, nregs

    try:
        code, labels, words, exits, nregs = generate()
    except ra.NotSupported as e:
        print(f"error: our backend can't compile this kernel yet: {e}\n"
              "  That PTX instruction isn't implemented in openassembler/backend/ra.py. Rewrite the "
              "kernel without it, or see docs/FRONTENDS.md for what is supported.",
              file=sys.stderr)
        return 1
    except EncodeError as e:
        print(f"error: our encoder can't encode this kernel: {e}\n"
              "  See docs/FRONTENDS.md (\"Python\" / \"CUDA C/C++\" sections) for what is supported.",
              file=sys.stderr)
        return 1

    base_path = Path(a.base_cubin) if a.base_cubin else None
    if base_path is None:
        # ptxas runs inside the colima VM, which only bind-mounts this repo, so
        # all scratch files (including the PTX itself, for .py/.cu sources
        # that didn't start as a .ptx file on disk) have to live under ROOT.
        import os
        import uuid

        scratch = ROOT / "corpus" / ".tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        tag = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
        scratch_ptx = scratch / f"base_{tag}.ptx"
        base_path = scratch / f"base_{tag}.cubin"
        scratch_ptx.write_text(ptx)
        try:
            oracle.ptxas(scratch_ptx, base_path, arch=a.arch)
        except subprocess.CalledProcessError as e:
            base_path.unlink(missing_ok=True)
            why = (e.stderr or "").strip() or f"exit code {e.returncode}"
            print("error: couldn't build the ELF container with ptxas (our backend splices its "
                  f"code into one; see README \"How it works\"):\n  {why}\n"
                  "  ptxas comes with the CUDA toolkit: run env/fetch_cuda.sh "
                  "(on macOS also `colima start`), or pass --base-cubin.", file=sys.stderr)
            return 1
        except (FileNotFoundError, oracle.OracleError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        finally:
            scratch_ptx.unlink(missing_ok=True)
    base = base_path.read_bytes()
    if a.base_cubin is None:
        base_path.unlink(missing_ok=True)
    cubin = splice_text(base, kname, words, nregs, exits)
    out_path = Path(a.out) if a.out else src_path.with_suffix(".cubin")
    if out_path.is_dir() or str(a.out or "").endswith(("/", os.sep)):
        out_path = out_path / (src_path.stem + ".cubin")      # -o some/dir  ->  some/dir/<name>.cubin
    sass_path = Path(a.sass) if a.sass else out_path.with_suffix(".sass")
    if not a.no_sass and sass_path.resolve() == out_path.resolve():
        print(f"error: the cubin and the .sass listing would both be written to {out_path}. "
              "Give -o a .cubin name (the .sass goes next to it), or pass --sass PATH.", file=sys.stderr)
        return 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(cubin)
    n_insns = sum(1 for t, _ in code if not t.startswith("NOP") and not t.startswith("BRA `(.L_x_0)"))
    print(f"{out_path}  ({n_insns} instructions, opt-level={a.opt_level}, regs={nregs})")

    if not a.no_sass:
        rc = _write_sass(cubin, sass_path, code)
        if rc:
            return rc
    if a.verify:
        rc = _verify_compiled(module, kname, code, labels, ptx, a.trials)
        if rc == 1:
            # A real mismatch (rc 2 means "couldn't check", which keeps the files). Don't leave
            # a cubin that just failed its own check where a later `gpu run` or a copy-paste
            # could pick it up as if it were good.
            out_path.unlink(missing_ok=True)
            if not a.no_sass:
                sass_path.unlink(missing_ok=True)
            print(f"(removed {out_path}: it did not pass --verify)", file=sys.stderr)
        return rc
    return 0


def _write_sass(cubin: bytes, sass_path: Path, code) -> int:
    """Disassemble the cubin we just wrote with nvdisasm and save it as a .sass
    listing (the same `nvdisasm -hex -c` format the rest of this repo reads:
    instruction text plus its raw 128-bit word). This is the real toolkit
    decoding our bytes, so it doubles as a check that what we encoded reads
    back as what we meant. nvdisasm runs natively on Linux and inside the
    colima VM on macOS (env/cuda.sh), so this works on both."""
    import os
    import re
    import uuid

    from openassembler import oracle

    # Disassemble from a scratch copy inside the repo: always reachable by the
    # tool (on macOS the VM only sees $HOME), wherever -o points.
    oracle.SCRATCH.mkdir(parents=True, exist_ok=True)
    tmp = oracle.SCRATCH / f"sass_{os.getpid()}_{uuid.uuid4().hex[:8]}.cubin"
    tmp.write_bytes(cubin)
    try:
        text = oracle.disasm_text(tmp)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
        detail = (getattr(e, "stderr", "") or str(e)).strip().splitlines()[-1:] or [""]
        print(f"error: couldn't disassemble the cubin into a .sass file: {detail[0]}\n"
              "  nvdisasm comes with the CUDA toolkit: run env/fetch_cuda.sh (on macOS also "
              "`colima start`), or pass --no-sass to skip this step.", file=sys.stderr)
        return 1
    finally:
        tmp.unlink(missing_ok=True)
    sass_path.parent.mkdir(parents=True, exist_ok=True)
    sass_path.write_text(text)
    print(f"{sass_path}  (SASS listing from nvdisasm)")

    # Soft sanity check, never fatal: every non-branch instruction nvdisasm
    # printed should be the one we encoded (branch targets are renamed by
    # nvdisasm, so those lines legitimately differ).
    norm = lambda s: re.sub(r"\s+", "", s)  # noqa: E731
    got = [i.text for i in oracle.parse_listing(text)]
    want = [t for t, _ in code]
    bad = [(w, g) for w, g in zip(want, got) if "BRA" not in w and "BSSY" not in w and norm(w) != norm(g)]
    if bad:
        print(f"warning: {len(bad)} instruction(s) disassemble differently than we encoded them, "
              f"e.g. {bad[0][0]!r} -> {bad[0][1]!r}", file=sys.stderr)
    return 0


_PROBE_PARAMS = (("u64", "out"), ("u64", "in"), ("u32", "k"), ("f32", "kf"))


def _verify_compiled(module, kname: str, code, labels, ptx_text: str, trials: int) -> int:
    """Picks the right checker for a just-compiled kernel: the original fixed
    probe-ABI harness for probe-shaped kernels (out/in/k/kf -- the corpus this
    repo's own tests use), or the generic one (openassembler.verify) for anything
    else, which covers pyfront/cppfront kernels and hand-written PTX with any
    signature."""
    kernel = module.kernels[kname]
    if tuple(kernel.params) == _PROBE_PARAMS:
        import random

        from openassembler.emu.probe import run_probe

        sys.path.insert(0, str(TOOLS))
        import diff_emu  # noqa: E402

        from bench_build import listing  # noqa: E402

        sass_text = listing(code, labels)
        rng = random.Random(0)
        for _ in range(trials):
            inputs = [tuple(diff_emu.rand32(rng) for _ in range(4)) for _ in range(32)]
            k, kf = diff_emu.rand32(rng), diff_emu.rand32(rng)
            got = run_probe(sass_text, inputs, k, kf)
            if run_probe(sass_text, inputs, k, kf, poison=0xDEADBEEF) != got:
                print("FAIL: output depends on uninitialized register contents", file=sys.stderr)
                return 1
            exp = diff_emu.run_ptx(ptx_text, kname, inputs, k, kf)
            for lane, (g, (e, undef, approx)) in enumerate(zip(got, exp)):
                if not undef and not diff_emu.same("verify", g, e, approx):
                    print(f"FAIL: lane {lane}: ours {g} vs PTX interpreter {e}", file=sys.stderr)
                    return 1
        print(f"OK: {trials} random trials (probe ABI), bit-exact against the PTX reference interpreter")
        return 0

    from openassembler.ptx.interp import NotModeled
    from openassembler.verify import run_and_diff

    for trial in range(trials):
        try:
            mismatches = run_and_diff(module, kname, code, labels, n_elems=64, seed=trial)
        except NotModeled as e:
            print(f"not verified: the PTX reference interpreter can't model this kernel ({e}). "
                  "The cubin compiled and was kept, but --verify cannot check it; write your own "
                  "test (see docs/FRONTENDS.md, \"What --verify proves\").", file=sys.stderr)
            return 2
        if mismatches:
            m = mismatches[0]
            print(f"FAIL: {m.param}[{m.index}]: ours {m.ours:#x} vs PTX interpreter {m.expected:#x} "
                  f"({len(mismatches)} of {len(mismatches)} total mismatches this trial)", file=sys.stderr)
            return 1
    print(f"OK: {trials} random trials (generic, {kname}'s own parameters), "
          "bit-exact against the PTX reference interpreter")
    return 0


def cmd_verify(a: argparse.Namespace) -> int:
    """Run an already-built cubin's SASS through the emulator and compare
    against the PTX reference interpreter on random inputs (same check
    compile_check.py and bench_build.py use before trusting a build). Only
    for kernels matching the probe corpus's fixed (out, in, k, kf) ABI --
    for your own kernel's own signature, use `compile --verify`, which picks
    this or the generic checker (openassembler.verify) automatically."""
    import random

    from openassembler import oracle
    from openassembler.emu.probe import run_probe

    try:
        sass = oracle.disasm_text(a.cubin)
    except (oracle.OracleError, subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"error: {getattr(e, 'stderr', '') or e}", file=sys.stderr)
        return 1
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
    if not any((ROOT / "corpus" / "sm_80" / "ptx").glob("*.ptx")):
        print("error: the probe corpus isn't built yet (corpus/ is build output, not in git).\n"
              "  Build it once:  openassembler corpus   (needs the CUDA toolkit, ~20 s)", file=sys.stderr)
        return 1
    argv = []
    if a.only:
        argv += ["--only", a.only]
    argv += ["--trials", str(a.trials), "--compiler", a.compiler]
    return _run_tool("compile_check.py", argv)


def cmd_corpus(a: argparse.Namespace) -> int:
    """Build the probe corpus (541 PTX probes -> ptxas -> cubin + SASS) under
    corpus/<arch>/. Needed once before `check`, `gpu run`, or the offline
    A100-replay tests; needs the CUDA toolkit. Wraps tools/build_corpus.py."""
    from openassembler import oracle

    if not oracle.available("ptxas"):
        print(f"error: {oracle.NEEDS_TOOLKIT}", file=sys.stderr)
        return 1
    return _run_tool("build_corpus.py", [a.arch])


def _need_toolkit() -> int | None:
    from openassembler import oracle

    if not oracle.available("ptxas"):
        print(f"error: {oracle.NEEDS_TOOLKIT}", file=sys.stderr)
        return 1
    return None


def cmd_bench(a: argparse.Namespace) -> int:
    """Build the microbenchmark cubins at every optimization level
    (safe/mined/sched/opt), each checked in the emulator first. Wraps
    tools/bench_build.py. Pair with `openassembler gpu run` to time them."""
    if (rc := _need_toolkit()) is not None:
        return rc
    return _run_tool("bench_build.py", ["--arch", a.arch])


def cmd_fastmm(a: argparse.Namespace) -> int:
    """Fast matrix multiplication: build Strassen/AlphaTensor-scheme GEMM
    kernels with our compiler (tools/fastmm_build.py), or run a plan on a GPU
    (tools/fastmm_run.py, Modal only -- for SSH/local use `gpu run` on the
    cubins fastmm build produces)."""
    if a.fastmm_cmd == "build":
        if (rc := _need_toolkit()) is not None:
            return rc
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
    if a.dry_run:
        # no GPU, no network: just say what would be sent (gpu_run.py itself has no --dry-run)
        sys.path[:0] = [str(ROOT), str(TOOLS)]
        import gpu_common as G

        try:
            jobs = G.build_jobs("sm_80", a.which, a.only, a.trials, 1)
        except SystemExit as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        print(f"dry run: {len(jobs)} {a.which} probes x {a.trials} trials would run on "
              f"{'this machine' if a.backend == 'local' else a.backend}; nothing was sent")
        return 0
    if a.backend == "local":
        import platform

        if platform.system() != "Linux":
            print("error: --backend local runs the probes on THIS machine's NVIDIA GPU, which needs "
                  f"Linux with an NVIDIA driver (this is {platform.system()}). Use --backend user@host "
                  "to run on a Linux GPU box over SSH, or --backend modal for a cloud GPU.", file=sys.stderr)
            return 1
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

    pc = sub.add_parser("compile", help="compile a .ptx/.py/.cu/.cpp file to a cubin with our backend")
    pc.add_argument("ptx", metavar="FILE", help=".ptx, .py (openassembler.pyfront), or .cu/.cpp (via nvcc)")
    pc.add_argument("-o", "--out", default=None)
    pc.add_argument("--kernel", default=None,
                     help="kernel/function name, if the source has more than one")
    pc.add_argument("--arch", default="sm_80")
    pc.add_argument("--opt-level", choices=OPT_LEVELS, default="opt")
    pc.add_argument("--base-cubin", default=None,
                     help="ptxas-built cubin to splice into (default: build one with ptxas now)")
    pc.add_argument("--verify", action="store_true", help="check the result against the PTX interpreter")
    pc.add_argument("--trials", type=int, default=6)
    pc.add_argument("--dump-ptx", action="store_true",
                     help="print the PTX our frontend generated (for .py/.cu sources) before compiling it")
    pc.add_argument("--sass", default=None, metavar="PATH",
                     help="where to write the SASS listing (default: next to the cubin, .sass)")
    pc.add_argument("--no-sass", action="store_true",
                     help="write only the cubin; skip the nvdisasm step")
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

    pco = sub.add_parser("corpus", help="build the probe corpus that check / gpu run / some tests need")
    pco.add_argument("--arch", default="sm_80")
    pco.set_defaults(func=cmd_corpus)

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
    raw = sys.argv[1:] if argv is None else argv
    if not raw:
        from openassembler.ui.repl import run as run_repl
        return run_repl()

    # First `openassembler <command>` typed directly into a given shell session
    # shows the logo once; later commands in that same session stay quiet
    # (openassembler/ui/session.py). The REPL shows it once at startup instead,
    # on its own, so this only applies to one-shot invocations. Skipped when
    # stdout isn't a terminal (scripts, CI, pipes) so output stays clean, and
    # for -h/--help so the banner doesn't eat into `| head` / pager output.
    if sys.stdout.isatty() and "-h" not in raw and "--help" not in raw:
        from openassembler.ui.banner import render
        from openassembler.ui.session import consume_show_once

        if consume_show_once():
            print(render())
            print()

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "gpu" and args.gpu_cmd != "run":
        parser.error("gpu: only 'run' is implemented")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
