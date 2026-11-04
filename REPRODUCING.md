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

The current CLI/application/tensor integration matrix, fresh GPU outcomes and
additional reproduction commands are in
[docs/INTEGRATION_REVIEW.md](docs/INTEGRATION_REVIEW.md). It also records archive
limits, per-output build locks, installed-wheel checks and unsupported scope.

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
drift). `gpu run --backend modal --dry-run` and `mlp_run.py --dry-run` print a dollar estimate before anything is billed; `fastmm_run.py --dry-run` prints only the launch plan, job count and upload size (no price), and needs `openassembler fastmm build` first. These historical cost estimates
exclude the newer MMA compiler and hardware-profiling experiments; they are not
current project spending totals.

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

### Native A100 tensor CLI

The installed CLI detects supported SM80 tensor PTX automatically. Generate the
validated full-tile plan, or compile its annotated source through the ordinary
compile path:

```sh
openassembler gemm --m 128 --n 128 --k 128 --dtype f16 --out-dir output
openassembler run output --backend modal --budget 180 --job-timeout 60
openassembler compile output/.build/generated/gemm.ptx --out-dir recompiled

# Larger measured plan; compilation still needs no GPU.
openassembler gemm --m 4096 --n 16384 --k 4096 --dtype bf16 --out-dir output_bf16

# Production backend/generator/CLI tests, including exact words and poison checks.
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  tests/test_backend_tensor.py tests/test_gemm_codegen.py tests/test_cli_gemm.py
```

The operation, alignment, supported shapes and control-flow subset are documented
in [TENSOR_GEMM.md](docs/TENSOR_GEMM.md). Production code has no `exp/` dependency.
`run` reads the generated launch contract, uses FP16/BF16 small-integer inputs
and compares every FP32 output bitwise with ptxas. This is a smoke check, not the
full floating validation or CUDA graph benchmark below. Tensor kernels without
a validated generated contract require `--spec`.

Full native CLI promotion evidence can be reproduced with the retained builders:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/a100_mma/phase4/native_build.py
PYTHONDONTWRITEBYTECODE=1 python3 -m modal run exp/a100_mma/modal_run.py \
  --suite phase4/native_suite.json
```

The builder verifies all six production instruction streams against the retained
phase-three binaries before the GPU run. Read
[the integration report](exp/a100_mma/phase4/REPORT.md) for current validation.

### Historical isolated A100 MMA compiler

This retained experiment originally ran separately from the CLI. Its validated
SM80 selector and GEMM schedule are now integrated as described above. Historical
scripts use their isolated tensor selector, encoder and cubin splicer;
ptxas builds its reference and ELF container. The six retained FP16/BF16 GEMM
results are in [exp/a100_mma/REPORT.md](exp/a100_mma/REPORT.md). No end-to-end
transformer training acceleration has been established.

```sh
# Compile only: CUDA toolkit, Python 3.12+, no GPU needed.
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/a100_mma/final_workloads.py

# Bounded A100 run: optional authenticated Modal client (tested with 1.6.2).
PYTHONDONTWRITEBYTECODE=1 python3 -m modal run exp/a100_mma/modal_run.py --suite validation-suite.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/a100_mma/summarize.py

# Tensor backend contracts and semantic checks, isolated temporary outputs.
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q exp/a100_mma/test_experiment.py \
  --basetemp exp/a100_mma/pytest-reproduce -o cache_dir=exp/a100_mma/pytest-reproduce-cache
```

The kernels require their documented full-tile dimensions and `A @ B.T` layout.
They accumulate and output FP32. The GPU harness checks every output against
cuBLAS for eight input cases, checks output canaries, and compares sampled
elements with a CPU float64 oracle. Random-case checks use relative L2 ≤ 2e-5;
zero and small-integer cases are exact. Timing uses seven shuffled rounds of
20-node CUDA graph replays. Unlocked clocks can change absolute timings.

The function timeout is 600 seconds, with a 480-second batch budget and
120-second worker watchdog. This suite is billed independently of the older
historical runs; previous project cost totals are not estimates for this suite.
Inspect `modal app list` after completion. Results include binary hashes and
compiler settings, and newer invocations archive immutable evidence under
`exp/a100_mma/runs/` and binaries under `exp/a100_mma/binaries/`.

The adapter also accepts `cnr`; for an existing generated small pipeline and
its packed-input launch specification:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/a100_mma/oa.py \
  --ssa-locals --interlocked-tensor --tensor-issue 8 --mined-scalar --ldsm-completion-wars \
  cnr exp/a100_mma/ptx/pipe_64_64_32_32_32_f16.ptx \
  --out-dir exp/a100_mma/build/reproduce-cli --spec exp/a100_mma/cli-spec.json \
  --backend modal --budget 60 --job-timeout 20
```

This command needs the small pipeline source/spec already present in the
experiment checkout; `final_workloads.py` generates the large validation sources.
The adapter rejects output paths outside the experiment. The `safe` level in
reused CLI manifests does not describe every experimental schedule switch;
`experiment.json` records the actual tensor schedule.

### A100 hardware profiling and follow-on experiments

Read [the phase-two report](exp/a100_mma/phase2/REPORT.md) before interpreting
results. These commands write experimental code artifacts and evidence under
`exp/a100_mma/`, including Nsight reports. Run the builders with the project
Python and the Modal client with the Python that has `modal==1.6.2` installed.

```sh
# Hardware counters: needs the retained passing build or final_workloads.py first.
PYTHONDONTWRITEBYTECODE=1 python3 -m modal run exp/a100_mma/phase2/profile_modal.py

# Clocked exact-SASS HMMA/LDSM probes, with checksums.
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/a100_mma/phase2/clocked_exact.py
PYTHONDONTWRITEBYTECODE=1 python3 -m modal run exp/a100_mma/phase2/clock_modal.py \
  --suite-path clocked-exact/suite.json --output clocked-exact/results-clean.json

# Two/three/four-stage comparison; every output checked before timing.
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/a100_mma/phase2/build_multistage.py --large
PYTHONDONTWRITEBYTECODE=1 python3 -m modal run exp/a100_mma/modal_run.py \
  --suite phase2/multistage-large-suite.json

# Paired compiler improvement on square/MLP/tall FP16 and BF16 workloads.
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/a100_mma/phase2/validate_pointer.py
PYTHONDONTWRITEBYTECODE=1 python3 -m modal run exp/a100_mma/modal_run.py \
  --suite phase2/pointer-validation-suite.json

# Fused-bias linear layer, nonaligned dimensions, dX/dW/db and SGD.
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/a100_mma/phase2/build_integration.py
PYTHONDONTWRITEBYTECODE=1 python3 -m modal run exp/a100_mma/phase2/integration_modal.py
```

Nsight Compute runs require permission to read GPU performance counters; the
tested Modal A100 exposed this permission. The profiler uses cache flushing and
kernel replay; its timing is not interchangeable with CUDA graph timing. Tensor
activity and stall percentages are scoped to the named counters, not whole-model
utilization. An unavailable metric or counter permission must be reported instead
of replaced by a synthetic estimate. The profiler function is capped at 540 s;
clock probes at 180 s; integration at 240 s. Each is an ephemeral, single-container
Modal app. Review the scripts for the chosen workload before running them.

The autograd experiment uses explicit padding and layout copies. It measures a
linear-layer training step and is slower than its matched reference in the
retained small case; it is not an optimized whole-transformer training path.
Native sparse-MMA capability probes use ptxas as an oracle, with no OpenAssembler
sparse code-generation or dense-model accuracy claim. AlphaTensor plans are
verified over rationals and contain constant factors, but no new low-rank GPU
speedup is claimed. See the [literature/feasibility table](exp/a100_mma/phase2/LITERATURE.md).

### Further A100 MMA experiments

[Phase three](exp/a100_mma/phase3/REPORT.md) contains the repeated paired-fragment
and vector-epilogue improvements, actual cuBLAS dispatch/SASS observations,
cuBLASLt heuristic comparisons, five-stage/producer-warp experiments, LDGSTS
immediate-field proofs and fused low-rank staging. Everything generated remains
under `exp/a100_mma/`. The compiler still runs in Python; the default compiler
backend is unchanged. Use Python3.11+ for local builds (`tomllib` is required).

```sh
# Build scalar/row references, then the six-workload composition suite.
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/a100_mma/phase3/epilogue_confirm.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/a100_mma/phase3/search_combined.py
PYTHONDONTWRITEBYTECODE=1 python3 -m modal run exp/a100_mma/modal_run.py \
  --suite phase3/search_combined_suite.json

# Stage specialization and final three-schedule comparison.
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/a100_mma/phase3/address_probe.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/a100_mma/phase3/address_shared.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/a100_mma/phase3/address_stage.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/a100_mma/phase3/address_stage_confirm.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/a100_mma/phase3/benchmark_suites.py
PYTHONDONTWRITEBYTECODE=1 python3 -m modal run exp/a100_mma/modal_run.py \
  --suite phase3/final_suite.json
# Earlier fixed-schedule confirmation, with more warmup and fewer alternatives.
PYTHONDONTWRITEBYTECODE=1 python3 -m modal run exp/a100_mma/modal_run.py \
  --suite phase3/confirmation_suite.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/a100_mma/phase3/evidence.py

# Exact encoding/dependency and full-output semantic checks, without a GPU.
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q exp/a100_mma/phase3 \
  -o cache_dir=exp/a100_mma/phase3/pytest_cache

# Nsight hardware counters for scalar and combined schedules plus matched ptxas.
PYTHONDONTWRITEBYTECODE=1 python3 -m modal run exp/a100_mma/phase3/profile_combined.py

# Actual matched cuBLAS dispatch, launch resources and resident SASS statistics.
PYTHONDONTWRITEBYTECODE=1 python3 -m modal run exp/a100_mma/phase3/cublas_modal.py
PYTHONDONTWRITEBYTECODE=1 python3 -m modal run exp/a100_mma/phase3/cublas_probe.py

# Broader vendor algorithms; native controls require the combined builds above.
PYTHONDONTWRITEBYTECODE=1 python3 -m modal run exp/a100_mma/phase3/lt_modal.py \
  --baseline-suite phase3/lt_controls_suite.json

# Larger tiles / up to five stages, correctness gated before timing.
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/a100_mma/phase3/geometry.py --large
PYTHONDONTWRITEBYTECODE=1 python3 -m modal run exp/a100_mma/modal_run.py \
  --suite phase3/geometry_large_suite.json

# Compile-time fused Strassen transforms and matched rank-8 control.
PYTHONDONTWRITEBYTECODE=1 python3 -m modal run exp/a100_mma/phase3/lowrank_modal.py
```

Individual reports describe the producer, address-field and stage-specialization
builders. cuBLASLt heuristics and timings depend on the installed library, GPU and
workspace; a timing-selected heuristic needs independent confirmation. The
low-rank microprototype has its own explicit FP16-transform rounding policy,
and its near-launch-floor timings do not establish faster large GEMM or training.
Profiler replay durations remain separate from the CUDA graph comparisons.

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
