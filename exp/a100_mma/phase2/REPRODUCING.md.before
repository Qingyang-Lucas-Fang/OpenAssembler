# Reproducing every number in this repo

Every claim in [README.md](README.md) and [docs/STATUS.md](docs/STATUS.md) comes from a
script in this repo. This file gives the command behind each one, grouped by what it needs:

| Section | Needs | Costs |
|---|---|---|
| [B. Offline](#b-offline-checks-no-nvidia-software-no-gpu) | Python 3.12+ | $0 |
| [C. With NVIDIA's toolkit](#c-with-nvidias-toolkit) (compile, build the corpus, full tests) | + CUDA toolkit (free download, no GPU) | $0 |
| [D. On a real GPU](#d-real-gpu) (costs money on Modal, free on your own) | + a Linux machine with an NVIDIA GPU, or a Modal account | $0 on your own GPU; cents on Modal |

All commands run from the repo root. Times are approximate: Linux x86_64 first, then the 4-core Linux VM on an Apple-silicon Mac (typically 2-4x slower for anything that calls the NVIDIA tools).

## A. Install

Requires **Python 3.12+**. The package has no Python dependencies.

```sh
git clone https://github.com/Qingyang-Lucas-Fang/OpenAssembler.git && cd OpenAssembler
python3 -m venv .venv && source .venv/bin/activate     # a venv avoids a system pip on an older Python
pip install -e .            # installs the `openassembler` command
```

The scripts in `tools/` also run directly (`python3 tools/<script>.py`) without installing;
they find the package themselves. `uv` is optional: every `python3 ...` below works as
`uv run --no-project python ...` too.

| Machine | Status |
|---|---|
| Linux x86_64 / ARM64 | native; the toolkit runs directly |
| macOS, Apple Silicon or Intel | supported; the toolkit is Linux-only, so it runs inside a small [colima](https://github.com/abiosoft/colima) VM that `env/cuda.sh` drives. **Keep the repo under your home directory** (the VM mounts only `$HOME`). |

Windows is not supported.

No NVIDIA GPU is needed for anything except section D.

## B. Offline checks (no NVIDIA software, no GPU)

| Result | Command | Time |
|---|---|---|
| Unit tests | `pip install pytest && python3 -m pytest -q` | ~5 s |
| Division/remainder-by-constant lowering is correct for **all 2³² inputs** (needs a C compiler: `clang` or `gcc`) | `python3 tools/verify_divconst_exhaustive.py` | 1-2 min |
| All 97 Strassen/AlphaTensor schemes are exact over ℚ | `python3 -c "from openassembler.fastmm import factors as F; s=F.all_schemes(); print(len(s), all(F.verify(v) for v in s.values()))"` | ~30 s |

On a fresh clone, tests that require NVIDIA tools or generated corpus data skip with a
specific reason (`pytest -rs`). The CPU-only compiler, frontend, and CLI tests run
without a GPU. With the toolkit set up, run the full suite with `uv run pytest -q`.

## C. With NVIDIA's toolkit

Our backend splices its code into an ELF container that `ptxas` builds for the same
kernel ([how it works](README.md#how-it-works)), and `nvdisasm` produces the `.sass`
listing, so compiling needs NVIDIA's tools once. They are never bundled: NVIDIA's license
forbids redistribution, so `env/fetch_cuda.sh` downloads CUDA 12.8 from NVIDIA and checks it
against NVIDIA's own published checksums. It picks the Linux x86_64 or ARM64 build to match
the machine that will run it.

```sh
# macOS only: the VM the tools run in
brew install colima && colima start

env/fetch_cuda.sh                 # ~300 MB into env/cuda-12.8/ (gitignored); needs curl, tar, xz, python3
env/cuda.sh ptxas --version       # must print "release 12.8"
```

Already have CUDA on Linux? Skip the download: have `ptxas`, `nvdisasm`, `nvcc` on `PATH`, or set
`OPENASSEMBLER_CUDA_HOME`. The measurements here are with 12.8; other versions may disassemble
slightly differently.

Now compiling works, and the probe corpus can be built (`corpus/` is generated output, so a
fresh clone has none):

| Result | Command | Time |
|---|---|---|
| Compile + verify one kernel (writes `output/cubin/out.cubin` and `output/sass/out.sass`) | `openassembler compile examples/div_by_const.ptx -o out --verify` | seconds |
| Same from Python or CUDA C | `openassembler compile examples/vector_add.py -o out --verify` (or `.cu`) | seconds |
| **Build the 541-probe corpus** (needed by the rows below) | `openassembler corpus` | ~20 s |
| Our compiler on every probe, validated against `nvdisasm` and the PTX interpreter | `openassembler check` | ~3 min |
| Naive v0 compiler, for comparison | `openassembler check --compiler naive` | ~3 min |
| Emulator vs. the **recorded** A100 runs (no GPU; replays `corpus/sm_80/gpu_a100_40gb_v2.raw.json.gz`). Compares bit-exactly and ends with a summary line: ~520 of 541 match, the rest are listed as expected (approximate instructions, clocks, probes since fixed) | `python3 tools/gpu_replay.py corpus/sm_80/gpu_a100_40gb_v2.raw.json.gz` | 10 s - 2 min |
| The same check as pass/fail (applies the tolerances) | `python3 -m pytest tests/test_gpu_truth.py -q` | ~10 s |
| Emulator vs. PTX interpreter, 541 probes x 8 trials | `python3 tools/diff_emu.py sm_80 --trials 8` | ~5 min |
| Tensor-core numerics (recorded lanes), then the model fit | `python3 tools/gpu_replay.py corpus/sm_80/gpu_a100_40gb_tensor.raw.json.gz`, `python3 tools/fit_mma_nominal.py` | ~1 / ~10 min |
| Benchmark kernels at four optimization levels | `openassembler bench` | ~1 min |
| Fast-matmul kernels for every scheme | `openassembler fastmm build` | ~15 min |
| Bit-field map (bit-flip fuzzing) | `python3 tools/bitflip.py sm_80 > corpus/sm_80/bitflip_report.txt` | ~5 s |
| Encoder round-trip | `python3 tools/encoder_eval.py` | ~1 min |
| Full test suite | `uv run pytest -q` | depends on toolkit/corpus availability |

If a step fails, the message says what is missing; the usual causes:

| Message | Fix |
|---|---|
| `colima not found` / `the colima VM isn't running` | `brew install colima && colima start` |
| `is outside $HOME` / `outside your home directory` | move the repo under your home directory (macOS VM mount) |
| `ptxas: command not found` / `couldn't build the ELF container` | `env/fetch_cuda.sh` |
| `the probe corpus isn't built yet` / `no ours cubins` | `openassembler corpus`, then `openassembler check` |
| `unsupported OS` | use Linux or macOS; Windows is not supported |
| `need xz` | `sudo apt install xz-utils` (NVIDIA ships .tar.xz) |
| `nvcc fatal ... gcc` compiling a `.cu` | install a host C++ compiler: `sudo apt install g++` |

## D. Real GPU

The headline numbers were measured on an A100-SXM4-40GB. Anything with an NVIDIA GPU can
run the correctness side; the timing tables currently need Modal (below).

### D1. Your own GPU, over SSH or locally: correctness of the probe corpus

`gpu run` sends the probe cubins to the GPU, runs them, and compares every result with the
emulator and the PTX spec. It needs the corpus from §C first:

```sh
openassembler corpus                                         # once
openassembler check                                          # once; builds the cubins of OUR compiler (--which ours)

openassembler gpu run --backend user@your-gpu-box --only add.s32 --trials 2     # smoke test, seconds
openassembler gpu run --backend user@your-gpu-box --trials 4                    # ptxas's cubins (sanity)
openassembler gpu run --backend user@your-gpu-box --which ours --trials 4       # our compiler's cubins
openassembler gpu run --backend local --which ours                              # this machine's own GPU (Linux)
```

What the GPU host needs: Linux, an NVIDIA driver (`libcuda.so.1`), `python3` 3.8+, and GNU
`timeout` (standard on Linux). No CUDA toolkit, no compiler, nothing installed: the tool
copies a small stdlib-only package over `ssh`/`scp`, runs it under a hard timeout, copies the
results back, and deletes its scratch directory. You need passwordless `ssh` to the host.
`--backend local` is Linux-only (it uses GNU `timeout`, and macOS has no NVIDIA GPU anyway).

Results land in `corpus/sm_80/gpu_<host>.json` (`gpu_<host>_ours.json` with `--which ours`; `gpu_local.json` for `--backend local`), plus a `.raw.json.gz` you can replay offline as in §C.

**Honest status of this path:** it is tested end to end against a simulated driver, and `--backend local` / Modal against real A100s, but it has **not been run against a real SSH host** yet. If it misbehaves on yours, that is a bug worth reporting.

### D2. Modal: a cloud A100, and where the timing numbers come from

Use this if you have no GPU. You need a [Modal](https://modal.com) account:

```sh
pip install modal==1.6.2 && modal setup

openassembler gpu run --backend modal --dry-run                      # job count and cost, $0
openassembler gpu run --backend modal --only add.s32 --trials 2      # ~$0.01
openassembler gpu run --backend modal --which ours --trials 4        # our cubins; ~$0.01
modal app list; modal container list                                 # must show nothing running afterwards
```

The **timing** tables (README "Results") use Modal-only scripts today: the SSH/local path checks
correctness but has no `--timing` mode. Each needs its cubins built first (§C):

```sh
# ptxas vs. our compiler, 7 loop kernels (README table 1). Run `openassembler bench` first.
modal run tools/gpu_modal.py --which dir:corpus/sm_80/bench --trials 2 --timing --grid 4096 --reps 30 --iters 4096 --budget 300

# Strassen/AlphaTensor vs. cuBLAS at n = 8192 (README table 2). Run `openassembler fastmm build` first (~15 min).
modal run tools/fastmm_run.py --n 8192 --schemes strassen_2x2x2,at_2x2x2,strassen2_4x4x4,at_gpu_4x4x4 --rounds 2 --reps 3 --budget 580 --job-timeout 200
modal run tools/fastmm_run.py --n 8192 --schemes strassen_2x2x2,at_2x2x2,strassen2_4x4x4,at_gpu_4x4x4 --dry-run   # the plan, no GPU, no price
```

The n = 8192 run is about 10 A100-minutes, roughly \$0.35 at Modal's A100-40GB rate (prices
drift). `gpu run --backend modal --dry-run` and `mlp_run.py --dry-run` print a dollar estimate before anything is billed; `fastmm_run.py --dry-run` prints only the launch plan, job count and upload size (no price), and needs `openassembler fastmm build` first. Everything in this
repo so far cost under \$2 in total.

**Billing safety.** `modal run` is ephemeral and stops when the local process exits; if it is
killed hard, `modal app list` then `modal app stop -y <id>` cleans up. Three layers of timeout
(per-kernel watchdog, in-container `--budget`, Modal's hard limit) bound a run even if a kernel
hangs. Never use `modal shell` for this: it bills until you close it.


### Modal reproduction

This is the exact path for reproducing a project `run` or `cnr` on a cloud GPU. Modal is billed while a container is running, so begin with the dry run.

```sh
pip install modal
modal setup
openassembler compile examples/vector_add.py -o vector_add --out-dir output
openassembler run output --backend modal --dry-run --n 1024 --reps 0 --budget 60
openassembler run output --backend modal --n 1024 --reps 3 --budget 60 --job-timeout 20
```

The dry run prints the jobs, upload size, GPU choice, budget, timeout, and a price estimate; it starts no container. `--budget` is the total in-container limit (capped by the Modal adapter), `--job-timeout` is the watchdog for one kernel job, and Modal adds a hard function timeout. Use a small `--n`, few repetitions, and a single kernel for a first smoke test. For real workloads, pass `--spec SPEC.json` with explicit buffers and launch geometry.

The command writes `output/run.json`. Timestamped raw Modal evidence belongs outside the committed output tree so a later run cannot overwrite a recorded result. On completion check for leaked resources:

```sh
modal app list
modal container list
modal app stop APP_ID       # only if an app remains after a failed client process
```

Do not use `modal shell` for this workflow: an interactive container can continue billing until it is closed. A failed client process can leave an app alive; the cleanup commands above are the required recovery.

### D3. Your own kernel on a GPU: the MLP example

`compile` writes the cubin, SASS listing, PTX, ptxas reference cubin, and manifest under
`output/`. `run output` executes the manifest directly; `cnr SRC` compiles and then runs the
kernels built in that invocation. The tested end-to-end example is a 2-layer MLP written in Python
([`examples/mlp/kernels.py`](examples/mlp/kernels.py)), built with both compilers, run on an A100, and
checked against a float64 reference:

```sh
python3 tools/mlp_build.py                         # builds corpus/sm_80/mlp/{ptxas,ours}@*.cubin
modal run tools/mlp_run.py --dry-run               # job size, $0
modal run tools/mlp_run.py --batch 64 --in-dim 128 --hidden 256 --out-dim 64    # ~$0.01
```

[`tools/mlp_run.py`](tools/mlp_run.py) shows how to describe cubins, buffers and launch geometry as a job
([`run_plan_job`](openassembler/gpu/cuda_driver.py) documents the format, and it also works over SSH).
More on what each frontend accepts: [docs/FRONTENDS.md](docs/FRONTENDS.md).

## E. What the CLI maps to

`openassembler <command> --help` has every flag.

| Command | Does | Section |
|---|---|---|
| `compile FILE` or `compile REPO/` | file/repository to `output/cubin/`, `output/sass/`, and manifest (`--out-dir`, `--verify`, `--fallback`) | C |
| `run OUTDIR` | run an existing manifest on local/SSH/Modal, compare with ptxas; accepts `--spec` and `--dry-run` | D |
| `cnr SRC` | compile a file/repository, then run only what was built | D |
| `verify CUBIN PTX --kernel NAME` | check a cubin of the probe signature `(out, in, k, kf)` against the PTX interpreter | C |
| `corpus` | build the 541-probe corpus (`tools/build_corpus.py`) | C |
| `check` | `tools/compile_check.py` | C |
| `bench` | `tools/bench_build.py` | C |
| `fastmm build` / `fastmm run` | `tools/fastmm_build.py` / `tools/fastmm_run.py` (Modal) | C / D2 |
| `nn_study.py` | compile and measure ReLU, GELU, softmax, LayerNorm, Adam, cross-entropy, matmul, and attention baselines | D2 |
| `gpu run --backend local\|user@host\|modal` | `tools/gpu_run.py` / `tools/gpu_modal.py` | D1 / D2 |
| `status` | the headline results from `docs/STATUS.md` | — |
| *(none)* | interactive shell over all of the above | — |
