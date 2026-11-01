# Reproducing every number in this repo

Every claim in [README.md](README.md) and [docs/STATUS.md](docs/STATUS.md) comes
from a script in this repo. This file lists the exact command behind each
category of result, split by what it costs and what it needs.

All commands run from the repo root. Offline sections need nothing but Python
3.12+; the oracle and GPU sections need NVIDIA's own tools and, for sections
C/D, a real GPU.

## A. Install

```sh
git clone <this-repo> && cd openassembler
pip install -e .                 # installs the `openassembler` command
# or, to match exactly how this repo's own tools/*.py scripts were run:
pip install uv
```

Every command below has two forms: `openassembler <subcommand> ...` (the CLI) or
`uv run --no-project python tools/<script>.py ...` (the script the CLI wraps
directly — useful if you want flags the CLI doesn't expose yet). Both do the
same thing; the CLI prints the equivalent `tools/` invocation it's running
before most GPU commands.

## B. Offline results ($0, no GPU, no NVIDIA tools needed)

| Result | Command | Time |
|---|---|---|
| All 164 tests (incl. replays of recorded A100 runs) | `uv run --with pytest==8.4.2 --no-project pytest -q` | ~30 s |
| Compile + verify one example kernel | `openassembler compile examples/div_by_const.ptx -o /tmp/x.cubin --verify` | ~1 s (plus the oracle setup in §C, since `compile` needs a `ptxas`-built ELF template) |
| Div-by-constant proofs (all 2³² inputs, exhaustive) | `uv run --no-project python tools/verify_divconst_exhaustive.py` | ~1 min |
| Strassen/AlphaTensor scheme verification (97 schemes, exact over ℚ) | `uv run --no-project --with numpy python tools/fastmm_import.py` then check `specs/fastmm/schemes.json.gz` loads via `openassembler.fastmm.factors.all_schemes()` | ~24 s |
| Emulator vs. PTX interpreter (offline, 541 probes × 8 trials) | `PYTHONPATH=. uv run --no-project python tools/diff_emu.py sm_80 --trials 8` | ~5 min |
| Emulator vs. **recorded** A100 runs (no GPU needed — replays `corpus/sm_80/gpu_a100_40gb_v2.raw.json.gz`) | `PYTHONPATH=. uv run --no-project python tools/gpu_replay.py corpus/sm_80/gpu_a100_40gb_v2.raw.json.gz` | ~2 min |
| Tensor-core numerics model fit (recorded lanes) | `PYTHONPATH=. uv run --no-project python tools/gpu_replay.py corpus/sm_80/gpu_a100_40gb_tensor.raw.json.gz`; `... tools/fit_mma_nominal.py` | ~1 / ~10 min |

`...` = `PYTHONPATH=. uv run --no-project python`. These all read files already
in `corpus/sm_80/*.json(.gz)` — no network, no GPU, nothing to set up.

## C. Setting up the oracle (needed for §D and to build *new* cubins)

Our compiler reuses a `ptxas`-built cubin as the ELF container (see
README.md "How it works"), so compiling anything new — not just replaying
recorded data — needs `ptxas`, `nvdisasm`, and `cuobjdump` on hand once.

**NVIDIA's CUDA toolkit is never vendored in this repo** (see [NOTICE](NOTICE)).
`env/fetch_cuda.sh` downloads it straight from NVIDIA's own servers and
verifies it against NVIDIA's own published manifest (`redistrib_12.8.0.json`),
the same way NVIDIA's installers do internally.

```sh
# If you're not on Linux (e.g. macOS): a throwaway Linux VM to run the
# toolkit in. Any Linux VM works; this is what the recorded results used.
brew install colima docker
colima start --cpu 4 --memory 6 --disk 40

env/fetch_cuda.sh                       # downloads + sha256-verifies CUDA 12.8 linux-sbsa
env/cuda.sh ptxas --version              # must print "release 12.8"
```

If you're already on Linux with the CUDA toolkit on `PATH`, you don't need
`env/cuda.sh` or colima at all — `openassembler.oracle` only shells out through
`env/cuda.sh` because this repo was developed on a Mac. Point
`OPENASSEMBLER_CUDA_HOME` at an existing install, or adapt `env/cuda.sh`'s one-line
wrapper to just exec the command directly.

With the oracle set up:

| Result | Command | Time |
|---|---|---|
| Rebuild the 541-probe PTX/SASS/cubin corpus | `openassembler check` (or `uv run --no-project python tools/build_corpus.py sm_80`) | ~20 s |
| Our compiler, full corpus, all 4 opt levels | `openassembler check --compiler ra` | ~3 min |
| Naive compiler v0 (A/B baseline) | `openassembler check --compiler naive` | ~3 min |
| Build the 7 benchmark kernels at every opt level | `openassembler bench` | ~10 s |
| Build the 73 fast-matmul kernels (all schemes) | `openassembler fastmm build` | ~15 min (each kernel is checked in the emulator before being written) |
| Bit-field map (bit-flip fuzzing) | `uv run --no-project python tools/bitflip.py sm_80 > corpus/sm_80/bitflip_report.txt` | ~5 s |
| Encoder round-trip (corpus + randomized register/offset rewrites, count varies per run) | `uv run --no-project python tools/encoder_eval.py` | ~1 min |

## D. Real GPU — the part that costs money (or doesn't, if you have one)

This is where README.md's and STATUS.md's headline numbers come from.
**Any NVIDIA GPU works** — you don't need Modal or any cloud account.

### D1. A GPU you can SSH into, or this machine (recommended path for anyone reproducing this)

```sh
# Smoke test: 3 probes, a few seconds
openassembler gpu run --backend user@your-gpu-box --only add.s32 --trials 2

# Full probe corpus against ptxas's own cubins (sanity check: should mostly match)
openassembler gpu run --backend user@your-gpu-box --trials 4

# Our compiler's cubins, the actual result this repo is about
openassembler gpu run --backend user@your-gpu-box --which ours --trials 4

# This machine, if it has an NVIDIA GPU + driver (no CUDA toolkit needed here)
openassembler gpu run --backend local --which ours
```

The remote machine needs only a Linux NVIDIA driver (`libcuda.so.1`) and
`python3 >= 3.8` — no CUDA toolkit, no compiler. `openassembler gpu run` copies
over a small stdlib-only package (`openassembler/gpu/`), runs it with a hard
timeout even if the SSH connection drops, copies results back, and deletes
the remote scratch directory.

Benchmark timing and fast-matmul plans need the richer `tools/` scripts
directly (the CLI's `gpu run` covers the probe-correctness path; the timing
variants aren't wired into it yet):

```sh
PYTHONPATH=. uv run --no-project python tools/bench_build.py
python tools/gpu_run.py --ssh user@your-gpu-box --which ours --tag mybox
```

### D2. Modal (cloud A100, what produced the numbers in this README)

Convenience, not a requirement — use this if you don't have GPU access
otherwise. Needs a [Modal](https://modal.com) account.

```sh
pip install modal==1.6.2 && modal setup

openassembler gpu run --backend modal --dry-run              # job count + cost, $0 spent
openassembler gpu run --backend modal --only add.s32 --trials 2   # ~$0.01, few seconds
openassembler gpu run --backend modal --which ours --trials 4     # our compiler's cubins

modal app list; modal container list      # must show nothing running when you're done
```

**Billing safety**, inherited from `tools/gpu_modal.py`: `modal run` is
ephemeral and stops when the local process exits; if it's killed hard,
`modal app list` / `modal app stop -y <id>` cleans up. Three layers of
timeout (per-kernel watchdog, in-container `--budget`, Modal's hard function
timeout) bound a single run even if a kernel hangs. **Never use `modal
shell`** for this — it stays billing until you remember to close it.

Timing and fast-matmul runs on Modal (outside the `openassembler` CLI for now,
same safety model):

```sh
modal run tools/gpu_modal.py --which dir:corpus/sm_80/bench --trials 2 --timing --grid 4096 --reps 30 --iters 4096 --budget 300
modal run tools/fastmm_run.py --n 8192 --schemes strassen_2x2x2,at_2x2x2,strassen2_4x4x4,at_gpu_4x4x4 --rounds 2 --reps 3 --budget 580 --job-timeout 200
```

The n = 8192 fast-matmul run above costs roughly $0.35 (≈10 A100-minutes at
Modal's A100-40GB rate — check `openassembler fastmm run --dry-run` for a current
estimate, prices drift). Every run in this repo so far has cost under $2
total; `--dry-run` always prints the estimate before anything is billed.

## E. What ties it together

`openassembler <subcommand> --help` for the full flag reference. The subcommands
map onto the sections above:

| CLI | wraps | section |
|---|---|---|
| `compile`, `verify` | `openassembler.backend.ra`, `openassembler.container.cubin` directly | B/C |
| `check` | `tools/compile_check.py` | C |
| `bench` | `tools/bench_build.py` | C |
| `fastmm build` / `fastmm run` | `tools/fastmm_build.py` / `tools/fastmm_run.py` | C / D2 |
| `gpu run --backend local\|user@host\|modal` | `tools/gpu_run.py` / `tools/gpu_modal.py` | D1 / D2 |
| `status` | prints `docs/STATUS.md`'s headline table | — |
| *(no subcommand)* | `openassembler.ui.repl` — interactive shell over every command above | — |
