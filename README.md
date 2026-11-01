  <p align="center"><img src="figs/openassembler_logo.png" width="820"></p>

<div align="center">

# OpenAssembler

**A from-scratch NVIDIA SASS compiler, built entirely by reverse-engineering `ptxas` as a black box.**

No NVIDIA source. No decompiled binaries. Every encoding bit, every instruction
latency, and every ELF quirk below was *measured* — by compiling probe kernels,
diffing `nvdisasm` output, flipping bits and watching what breaks, and running
the result on a real A100.

[![Tests](https://img.shields.io/badge/tests-164%20passing-brightgreen)](#)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Arch](https://img.shields.io/badge/arch-sm__80%20(Ampere)-76b900)](#)

[What this is](#what-this-is) ·
[Quick start](#quick-start) ·
[What we found](#what-we-found-on-real-hardware) ·
[How it works](#how-it-works) ·
[Results](#results) ·
[Reproduce it](REPRODUCING.md)

</div>

---

## What this is

`ptxas`, NVIDIA's PTX-to-SASS compiler, is closed source. **OpenAssembler is a
second implementation of it**, built by treating `ptxas` as a black box and
reverse-engineering its instruction encoding, scheduling, and ELF container
format from the outside — then replacing its code generator with our own.

Concretely, this repo contains:

- **A SASS encoder/decoder for sm_80 (Ampere)** — opcodes, operand forms,
  control bits (stall, yield, scoreboards, wait masks) — recovered by bit-flip
  fuzzing against `nvdisasm` ([docs/findings-sm80-encoding.md](docs/findings-sm80-encoding.md)).
- **A PTX → SASS compiler** (register allocation, scheduling, 4× loop
  unrolling) that emits code **1,103 instructions vs. ptxas's 1,182** on a
  110-probe corpus, and beats `ptxas`'s own output by **3–36%** on measured
  A100 wall-clock time for several kernel shapes.
- **A cycle-accurate-ish SASS emulator**, fitted to real A100 behavior:
  tensor-core (HMMA/IMMA/DMMA) numerics bit-exact on 512/512 recorded lanes,
  instruction latencies measured directly on hardware (not guessed from
  `nvdisasm`'s mined minimum distances, which are provably wrong — see
  below).
- **Division/remainder by a constant**, replacing `ptxas`'s always-runtime
  reciprocal sequence with a [z3-proven](openassembler/opt/divconst.py)
  multiply-high lowering: **13 → 2–3 instructions**.
- **Strassen / AlphaTensor fast matrix multiplication**: 97 published
  low-rank factorizations of matrix multiply (AlphaTensor, DeepMind/Nature
  2022) imported, verified exactly over the rationals, compiled by our
  backend, and measured against cuBLAS on an A100 at n = 8192.
- **`openassembler`, a CLI** that ties all of this together: compile a PTX
  file straight to a cubin with our backend, verify it against a reference
  interpreter, and benchmark it on any NVIDIA GPU you can reach — your own
  box, an SSH host, or a cloud GPU.

Every number in this README is reproducible; see [REPRODUCING.md](REPRODUCING.md)
and [docs/STATUS.md](docs/STATUS.md) for the exact command behind each one.

> **Honesty up front.** OpenAssembler does not replace `ptxas` end to end: it
> still uses a `ptxas`-built cubin as the ELF container (symbol table,
> relocations, constant banks) and only replaces the *code* inside it — see
> [How it works](#how-it-works). It also doesn't cover all of PTX yet: no
> f16/f64 arithmetic, no warp shuffles, no register spilling. See
> [Gaps](#gaps) and [docs/STATUS.md](docs/STATUS.md) for the honest list.

---

## Quick start

### Install
```bash
git clone github.com/Qingyang-Lucas-Fang/OpenAssembler && cd openassembler
pip install -e .   
```
One-time: fetch NVIDIA's own CUDA toolkit as a black-box oracle (never vendored in this repo — see NOTICE). Needs a Linux box or a VM (e.g. colima on macOS); see REPRODUCING.md if you're not on Linux.


### Setting up the oracle
```bash
env/fetch_cuda.sh  
env/cuda.sh ptxas --version              # must print "release 12.8"
```
### Run OpenAssembler
```bash
openassembler
```
and then it should display properly on your terminal
<img width="943" height="318" alt="Image" src="https://github.com/user-attachments/assets/6411bddf-d711-4c01-913f-bb30cee27e6f" />

---

### Commands for different devices
Run it on a real GPU — any Linux box with an NVIDIA driver, no CUDA toolkit needed htere:
```sh
openassembler gpu run --backend user@your-gpu-box --which ours
```

...or on this machine, if it has an NVIDIA GPU:
```sh
openassembler gpu run --backend local --which ours
```

...or on a cloud A100 via Modal (what produced the numbers below):
```sh
openassembler gpu run --backend modal --dry-run    # cost estimate, $0 spent
openassembler gpu run --backend modal
```

Run `openassembler --help` for the full command list (`compile`, `verify`,
`check`, `bench`, `fastmm build|run`, `gpu run`, `status`).

---

## What we found on real hardware

Everything here was measured on an A100-SXM4-40GB rented by the minute
through [Modal](https://modal.com) (see [REPRODUCING.md](REPRODUCING.md) for
cost and the exact commands). A few findings that surprised us:

- **`ptxas`'s own "mined minimum distances" are not real latencies.** Two
  instructions that `ptxas` sometimes schedules 4 cycles apart actually need
  5 on real hardware; using 4 gives wrong results. We measured the real
  numbers directly: **4 cycles within a pipe, 5 across pipes**, and a
  **13-cycle minimum** from a predicate write to any instruction that guards
  on it — confirmed by sweeping every `stall` field from 1–15 against the
  A100 (`corpus/sm_80/hw_latency.json`).
- **The register count declared in the cubin isn't just advisory.** Writing
  register `Rd` faults with `CUDA_ILLEGAL_INSTRUCTION` unless the declared
  count is ≥ `Rd + 3` — and that count lives in **two places**: the
  `EIATTR_REGCOUNT` record *and* bits 24–31 of the `.text.<kernel>` section's
  ELF `sh_info`. We only found the second one by faulting a real GPU
  (`corpus/sm_80/gpu_a100_40gb_bisect5.json`, 77 variants).
- **Tensor-core accumulation is truncating, not rounding, per term**, and the
  model below matches **100% of 512 recorded A100 lanes** per data type
  (f16/bf16/tf32/s8/f64) — the next-best model we tried only matched ~70%:

  <p align="center"><img src="figs/nv_tiling.png" width="820" alt="How an HMMA/IMMA tensor-core instruction actually builds its f32 sum: multiply at nominal exponent, align, truncate to a 24-bit window below the max exponent, sum exactly, round once."></p>

- **A hidden bit survives disassembly.** A template we synthesized by
  flipping bits kept 0xad of an old immediate in `ISETP` bits 32–39.
  `nvdisasm` doesn't print those bits — but the hardware reads them, and a
  stale value made every affected loop run one extra iteration.

The full list, each with the measured file behind it, is in
[docs/STATUS.md](docs/STATUS.md#what-the-a100-taught-us-each-finding-has-a-recorded-run-behind-it).

---

## Results

### Our compiler vs. `ptxas`, same PTX, A100-measured wall-clock

4096 CTAs × 32 threads, 4096 loop iterations, best median of 3 shuffled
timing rounds (a fixed order biases results — see
[docs/STATUS.md](docs/STATUS.md), "Timing method"):

| kernel | ptxas (ms) | ours, best variant (ms) | **ptxas / ours** |
|---|---|---|---|
| `loop.div7` (division by 7) | 0.629 | 0.464 | **1.36×** |
| `loop.rem1000` | 0.373 | 0.312 | **1.19×** |
| `loop.smem` | 0.748 | 0.713 | 1.05× |
| `ilp4.imad` | 0.257 | 0.250 | 1.03× |
| `ilp4.ffma` | 0.257 | 0.254 | 1.01× |
| `chain.imad_lop3` (dependent chain) | 0.129 | 0.135 | 0.95× |
| `chain.ffma` (dependent chain) | 0.089 | 0.097 | 0.92× |

All 49 cubins from this table are bit-exact against each other across runs.
We win on division (magic-number lowering vs. `ptxas`'s always-runtime
reciprocal) and on loops with enough work to hide scheduling latency; `ptxas`
still wins tight dependent chains by 5–8%, because it hides the loop counter
and branch inside the chain and we pay a 13-cycle predicate wait per
unrolled trip (see [Gaps](#gaps)).

### Strassen / AlphaTensor fast matmul vs. cuBLAS, fp32, n = 8192, A100

| plan | our compiler (ms) | vs. cuBLAS | rel. error |
|---|---|---|---|
| cuBLAS SGEMM (reference) | 57.6 | 1.000× | 1.6e-6 |
| our direct GEMM | 60.0 | 0.960× | 1.6e-6 |
| **Strassen ⟨2,2,2⟩, rank 7** | **58.4** | 0.986× | 2.3e-6 |
| **AlphaTensor ⟨2,2,2⟩, rank 7** | **58.3** | 0.988× | 2.7e-6 |
| AlphaTensor 4×4×4, rank 49 | 71.6 | 0.804× | 5.7e-6 |

One level of Strassen's rank-7 decomposition trades 12.5% fewer block
multiplications for a 2.7% speedup over our own direct GEMM — the classic
result (DeepMind, *Nature* 2022), reproduced end to end on our own backend.
4×4×4 schemes lose at this size: the block GEMMs become too small to fill
108 SMs. Full table, every scheme, and the n = 2048/4096 crossover in
[docs/STATUS.md](docs/STATUS.md#fast-matrix-multiplication-alphatensor-schemes--our-compiler--a100).

### Correctness, not just speed

| | |
|---|---|
| Our compiler's cubins on real hardware | **110/110 bit-exact** on an A100 (4 trials × 32 threads) |
| Emulator vs. real A100 (`ptxas` cubins) | **532/541 probes** bit-exact or within PTX spec tolerance |
| Tensor cores (HMMA/IMMA/DMMA, LDSM, LDGSTS) | bit-exact on 512/512 recorded lanes, every dtype |
| Division/remainder-by-constant lowering | proved equal to `/` and `%` by z3 (exhaustive for 32-bit) |
| Strassen/AlphaTensor factorizations | all 97 verified exactly over ℚ |
| Offline tests | 164 passing |

---

## How it works

```
 PTX source
     │
     ▼
 ┌─────────────┐   ┌────────────────────────────────────────────--──┐
 │   ptxas     │   │  openassembler (this repo)                     │
 │ (reference) │   │                                                │
 │             │   │  parse → select IR → DCE → coalesce →          │
 │  produces   │   │  linear-scan allocate → lower → [unroll ×4] →  │
 │  the ELF    │   │  [list-schedule] → stall/yield/scoreboard →    │
 │  container  │   │  encode (our own encoder, from the recovered   │
 └──────┬──────┘   │  spec, not ptxas's)                            │
        │          └───────────────────┬────────────────────────┬--─┘
        │ cubin (symbols, relocs,      │ new .text bytes,       │
        │ constant banks — unchanged)  │ regcount, EXIT offsets │
        ▼                              ▼                        │
 ┌───────────────────────────────────────────-─┐                │
 │  splice_text(): patch the cubin in place    │ ◄──────────────┘
 │  (openassembler/container/cubin.py)         │
 └──────────────────────┬──────────────────────┘
                        ▼
                 a cubin the driver
                 runs exactly like
                 any other — but every
                 instruction inside it
                 is ours
```

We don't build an ELF from scratch. `ptxas`'s cubin for the same kernel
*signature* already has a correct symbol table, relocations, and constant
banks — reverse-engineering that container format byte-for-byte would be a
separate, much less interesting project. So we reuse it as a template and
only replace what we actually built: the instruction words, the declared
register count (patched in **two** places — see above), and the exit-offset
table. `openassembler compile` without `--base-cubin` runs `ptxas` once, on
the same PTX, purely to get this template; no `ptxas` code generation ends
up in the output.

**The compiler pipeline** (`openassembler/backend/ra.py`):

1. **Select**: PTX → an IR over virtual registers. `mov`/pack/unpack cost
   nothing (pure renaming). `mul.wide` + `add.s64` fuse into one
   `IMAD.WIDE`, matching `ptxas`'s own idiom.
2. **DCE, then coalesce**: delete copies that don't conflict.
3. **Linear-scan register allocation** over live intervals, with
   copy-affinity hints, iterated to a fixed point.
4. **Lower** to SASS text with physical registers.
5. *(optional)* **Unroll** counted loops 4× (`backend/unroll.py`) and
   **list-schedule** each basic block by critical path over A100-measured
   latencies, not `ptxas`'s mined (and sometimes wrong) ones
   (`backend/sched.py`).
6. **Encode** with our own bit-level encoder, built from the spec in
   `specs/sm_80/`, independently of `ptxas`.

Four optimization levels are available end to end
(`--opt-level safe|mined|sched|opt`), matching the `safe/mined/sched/opt`
columns in the benchmark table above.

---

## Future Improvements

Here are a list of possible further improvements:

- **Dependent instruction chains**: `ptxas` still wins by 5–8%. It hides the
  loop counter increment and branch test inside the dependent chain; our
  unrolled loop pays a 13-cycle predicate wait per trip. Software-pipelining
  the next trip's predicate one iteration early should close this.
- **Base GEMM is ~4% behind cuBLAS** at n = 8192, which every fast-matmul
  plan inherits. A tensor-core (TF32/HMMA) GEMM using our bit-exact HMMA
  model, plus `cp.async` pipelining, is the next step — with a GEMM at
  parity, the measured 2.7% Strassen gain would put us *ahead* of cuBLAS.
- **4×4×4 schemes lose at n = 8192**: the block GEMMs are too small to fill
  108 SMs (only 2.4 waves). Larger n, or fusing two Strassen levels, is
  untested.
- **Fused multi-term GEMM kernels are slower than unfused** — memory-bound on
  the extra loads, and our scheduler loses to `ptxas` here specifically
  (they need 110+ live registers).
- **No f16/f64 arithmetic, no warp shuffles/votes, no register spilling, no
  `ld.global.v2`, no more than 7 live predicates** in the compiler yet.
- **sm_86/89/90/100/120**: the plan runner and timing harness are already
  arch-independent; the encoder and latency model are sm_80 (Ampere) only.

The living version of this list, with what's measured vs. hypothesized, is
[docs/STATUS.md § Gaps and next steps](docs/STATUS.md#gaps-and-next-steps).

---

## Repository layout

```
openassembler/        the package: backend/ (compiler),
                      isa/ (encoder),
                      emu/ (SASS emulator),
                      container/ (ELF splicing),
                      ptx/ (parser + reference interpreter),
                      opt/ (div-by-const),
                      fastmm/ (Strassen/AlphaTensor),
                      gpu/ (transport-agnostic GPU job runner),
                      cli.py

tools/                build/benchmark/fuzz scripts the CLI wraps; also usable directly

docs/         `       STATUS.md (every measured result),
                      WORKFLOW.md (method),
                      findings-*.md (encoding + div-by-const writeups),
                      research/

specs/                recovered encoding tables + AlphaTensor-derived GEMM schemes (JSON)

corpus/               generated probe kernels + evidence JSON (regenerated by tools/; the small summary files are kept, the bulk is gitignored)

third_party/          AlphaTensor factorizations (DeepMind, Apache-2.0 — see NOTICE)

env/                  env/fetch_cuda.sh + env/cuda.sh: the oracle wrapper (toolkit itself is never vendored, see NOTICE)

tests/                164 offline pytest tests, including replays of recorded A100 runs
```

## License

Apache License 2.0 — see [LICENSE](LICENSE). Third-party attributions (the
AlphaTensor import under `third_party/`) are in [NOTICE](NOTICE). This project
reverse-engineers NVIDIA's `ptxas` by black-box observation only; no NVIDIA
source code or binaries are included or redistributed.
