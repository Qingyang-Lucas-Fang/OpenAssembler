# Status (sm_80; CUDA 12.8 oracle; real-hardware validation on A100 via Modal)

2026-11-04 update: the standard CLI now compiles supported SM80 tensor PTX.
`openassembler gemm --m M --n N --k K --dtype f16|bf16` generates the validated
full-tile schedule, and `run` reads its launch contract. Six native instruction
streams match the measured kernels exactly and pass full-output A100 checks;
452 repository tests pass. See [the public contract](TENSOR_GEMM.md) and
[integration evidence](../exp/a100_mma/phase4/REPORT.md). Historical measurements
below retain their original scope; no whole-training-step speedup is claimed.

The [November 4 integration review](INTEGRATION_REVIEW.md) adds current CLI,
mixed-manifest, writer-lock, FP16/BF16 spec, application replay and wheel checks.
Its [machine-readable evidence](evidence/integration-20261104.json) records the
latest full-suite result and fresh A100 execution. Historical test counts below
refer to their original snapshots.

Updated 2026-11-03. Every number below comes from a script in this repo. Section
"Reproduce" lists the exact command for each one.

## New since 2026-10-25 (details in the sections below)

| Area | Result |
|---|---|
| **Current integration (2026-11-04)** | **486 tests pass**, no failures/skips; 262 supported scalar probes match semantics. Fresh A100 scalar/barrier/FP16/BF16 runs pass; original Transformer and nested-model applications replay with zero GPU code generation. Installed wheel passes outside-checkout checks. [Review](INTEGRATION_REVIEW.md), [evidence](evidence/integration-20261104.json). |
| **Native tensor CLI** | `compile` selects supported SM80 tensor PTX; `gemm` generates the validated FP16/BF16 full-tile plan. `run` uses retained launch metadata or an explicit FP16/BF16 `--spec`. PyTorch capture remains a separate backend. [Contract](TENSOR_GEMM.md). |
| **Measured speed, our compiler vs ptxas** (A100, 7 loop kernels, best median of 3 shuffled rounds) | ptxas time / our time: div7 **1.36×**, rem1000 **1.19×**, smem 1.05×, ilp4.imad 1.03×, ilp4.ffma 1.01×, chain.imad_lop3 0.95×, chain.ffma 0.92×. All 49 bench cubins bit-exact. `corpus/sm_80/gpu_a100_40gb_bench_times.json` |
| **Fast matmul (Strassen / AlphaTensor) compiled by our compiler** | 97 schemes imported and **all verified exactly** over the rationals. At n = 8192 fp32 on the A100: cuBLAS 57.6 ms; our direct GEMM 60.0 ms; **one Strassen / AlphaTensor-2×2×2 level 58.4 / 58.3 ms** (2.7% faster than our direct GEMM, 1.4% slower than cuBLAS); 4×4×4 schemes 69–72 ms. Relative error 2–6e-6 (cuBLAS 1.6e-6). `corpus/sm_80/fastmm_a100_40gb_n8192_sgemm128.json` |
| Hardware latencies (A100, measured, not mined) | Fixed-latency RAW: **4 cycles within a pipe, 5 across** (IMAD/FFMA pipe vs IADD3/LOP3/SHF pipe). **Predicate → any `@P`-guarded op or branch: 13 cycles.** SETP → `SEL …, P`: 4. ALU → STG data: 1–2. 33 pairs × stalls 1–15. `corpus/sm_80/hw_latency.json` |
| Compiler features | Branches, loops, predication (`@p` ops update in place), f32 ops, f32 immediates, shared memory (v1/v2/v4), `bar.sync 0`, `st.global.v4`, 4× loop unrolling, basic-block list scheduling, measured-latency stalls |
| Emulator | Multi-warp CTAs with cooperative `BAR.SYNC`; uniform-datapath ops execute once per warp (were executed per thread) |
| Tests | Run `uv run pytest -q`; toolkit and corpus tests skip with an explicit reason when unavailable. The backend regression suite covers arithmetic, pointer loops, shared-memory barriers, CLI manifests, and the GPU runner. |
| 0.2.0 backend regression | 65 arithmetic fragments (including division, sqrt, casts, approximate math, integer div/rem) and pointer-loop fixtures pass emulator/interpreter/encoder checks. 372 tests pass in the current toolkit environment. |
| Modal validation (2026-11-02 to 2026-11-03) | Python vector-add: bit-exact on A100-SXM4-80GB. Arithmetic adversarial replay: 65 compiled fragments, 320 input pairs each, bit-exact on A100-SXM4-40GB. CUDA pointer-loop pipeline: dot rows, grid-stride SAXPY, naive matmul, nested loops: bit-exact on A100-SXM4-40GB. A 64-thread cross-warp shared-memory barrier matched ptxas on A100-SXM4-80GB. NN baseline and evidence: [study](NN_OPTIMIZATION_STUDY.md). |

## Headline results (2026-10-25)

| Area | Result |
|---|---|
| **Our compiler on real hardware** | **110/110 cubins built by our compiler (v1) are bit-exact on an A100-SXM4-40GB** (4 trials × 32 threads each; `gpu_a100_40gb_ours_ra.json`) |
| Our compiler size | On those 110 probes: **1,103 SASS instructions vs ptxas 12.8's 1,182 (−6.7%)**; on the 80 both of our compilers handled then: v1 808, naive v0 2,281, ptxas 887. **Re-measured 2026-11-01** (`openassembler check`): on today's 203 probes, ours 2,132 vs ptxas 2,249 (−5.2%); on the 87 probes v0 and v1 both handle: v1 878, naive v0 2,477, ptxas 962. These are instruction counts for the historical probe set. No timing was collected on those 203 probes; selected microbenchmark timing is summarized separately above. |
| Our compiler coverage | The original probe corpus count remains a historical 203-kernel compiler run. The 0.2.0 arithmetic and pointer-loop regressions are tracked separately above; unsupported constructs remain explicit diagnostics. |
| Emulator vs A100 (ptxas cubins) | **532/541 probes reproduce the A100**, bit-exact or within the PTX spec's tolerance for `.approx` ops. The other 9: 7 nondeterministic (clock, globaltimer, smid, nsmid, warpid, gridid, cvta.shared) plus sin/cos lanes with \|x\| > 100π, where the spec gives no accuracy bound. |
| Tensor cores | HMMA f16/bf16/tf32, IMMA s8, DMMA f64, LDSM and LDGSTS are emulated **bit-exact on 512/512 recorded A100 lanes each**. |
| Emulator vs PTX interpreter (offline) | 518/541 bit-exact, 0 mismatches. The other 23 use ops the interpreter doesn't model (mma, ldmatrix, cp.async, timers); those are verified against hardware instead. |
| M2 stall model | 89.2% exact stall-field match on held-out kernels |
| Encoder | 1,239/1,239 randomized register/offset rewrites disassemble to the intended text |
| Tests | 372 passing in the 2026-11-03 toolkit run; fresh clones skip only tests whose external toolkit/corpus is unavailable. |
| GPU spend | About $1.5 of the $10 cap so far, by runner estimates (mostly the n = 8192 fast-matmul runs, about 10 A100-minutes each, including two runs whose results were lost when Modal stopped the app). Exact figure on the Modal usage page. Every run ended with zero containers; one leftover container was stopped by hand with `modal app stop`. |

## What the A100 taught us (each finding has a recorded run behind it)

| Finding | Evidence | Where it lives now |
|---|---|---|
| **Register-count rule:** writing Rd faults with `CUDA_ERROR_ILLEGAL_INSTRUCTION` unless the declared count ≥ Rd + 3. The count lives in `.text.<k>` `sh_info` bits 24–31, not only in `EIATTR_REGCOUNT`. | 77-variant sweep, `gpu_a100_40gb_bisect5.json`; ptxas itself always declares max-reg + 3 or more | `openassembler/container/cubin.py: required_regcount` |
| Our ELF writer is sound: same bytes, grown `.text`, raised register count and re-encoded words all run correctly | `gpu_a100_40gb_bisect.json` (15/15 variants match) | `openassembler/container/` |
| **Tensor-core numerics:** per step of 8 products (f16/bf16) or 4 (tf32), each product is aligned at its *nominal* exponent e(a)+e(b) together with the f32 accumulator; terms are truncated toward zero 24 bits below the largest exponent, summed exactly, and rounded to f32 toward zero. Matches 100% of lanes; the next-best model matches about 70%. | `gpu_a100_40gb_tensor.raw.json.gz`, `tools/fit_mma_nominal.py` | `openassembler/emu/tensor.py: MMA_MODEL` |
| f64 arithmetic keeps the input NaN's payload; F2F f32↔f64 keeps NaN sign and payload; NaN→int is `1<<(w−1)` for an f64 source or a 64-bit destination; `RSQ(−0) = −inf` | first full run `gpu_a100_40gb.json` | `openassembler/emu/fp.py`, `ops.py` |
| `FSEL` selects raw bits (no NaN canonicalization). ptxas uses it on f64 high words in min/max.f64. | v2 replay | `ops.py: _fsel` |
| `BRX` target = table entry: ptxas loads absolute `.text` offsets from `.nv.constant2.<k>` and cancels the pc. | v2 replay of `brx` | `ops.py: _brx`, `probe.py: cubin_consts` |
| Emulator bugs found by hardware: 128-bit register reads truncated to 32 bits (every `STS.128`/`STG.E.128`); `RSQ64H` underflowed for huge inputs | ldmatrix / sqrt.rn.f64 replays | fixed in `machine.py`, `ops.py` |
| Compiler bug found by hardware: 64-bit mov from a symbol aliased an unwritten register pair | `gpu_a100_40gb_ours.json` (cvta.shared) | refused in compiler; **poison-register check** in `compile_check.py` now catches this offline |
| **Predicate latency:** an `@P0` guard (BRA, IADD3, IMAD, FADD all tested) issued 6–12 cycles after the ISETP/FSETP that writes P0 uses the *old* P0. 13 cycles is the measured minimum, the same as ptxas's minimum in all its code. Loops ran one extra iteration; the emulator, which has no timing, was right. | `tools/lat_sweep.py` → `gpu_a100_40gb_latsweep.json` (`*.g` kernels, stalls 1–15) | `compile.py: _branch_pred_latency`, `PRED_TO_GUARD = 13` |
| **ptxas's mined minimum distances are not latencies.** IMAD → LOP3 at 4 cycles (ptxas uses 4 in some places) gives wrong results; the real value is 5. Hardware: 4 within the FMA pipe (IMAD/FFMA/FADD/FMUL) or the ALU pipe (IADD3/LOP3/SHF/IMNMX/MOV/SEL/ISETP), 5 across. | same sweep, 30 pairs × stalls 1–8 | `compile.py: hw_latency` (used by the "mined" scheduler) |
| **Hidden bits:** a template synthesized by bit flips kept 0xad of an old immediate in ISETP bits 32–39. nvdisasm does not print those bits, but the hardware reads them (every loop ran one iteration more). Bits whose flip leaves the text unchanged are now set to the majority value of real ptxas words with the same opcode. | bench run before the fix (`DIFFER` k+1) | `encode.py: minimize` |
| `@p add %r, …` in PTX is a partial definition: the old value must survive when the guard is false. The compiler wrote a fresh register. | `lat.ISETP>IADD3.g` on the A100 | `ra.py: run` (guarded writes get a fixed home) |
| Uniform datapath: `UIADD3 UR4, UR4, -0x3` was executed once per thread (32×) by the emulator | ptxas's unrolled loops never terminated in the emulator | `ops.py: execute` |
| **Timing method:** two byte-identical cubins timed 0.125 and 0.098 ms because jobs ran in a fixed order while the GPU ramped up. Timing jobs now run in several shuffled rounds; the best median is kept. | `gpu_a100_40gb_bench_times.json` | `tools/gpu_modal.py --rounds` |
| Stall 12–15 is only legal with the yield bit set (nvdisasm rejects it otherwise; ptxas writes `S13 Y` before branches) | nvdisasm error on our cubin | `compile.py` |
| Probe bugs found by hardware (10, all fixed): misaligned vector stores and loads, shared atomics racing on one word, `bar.sync 1,64` deadlock, three kinds of uninitialized-smem reads, racy `st.shared` | runs above | `tools/ptx_probes.py` |

## How the compiler works (v1, `openassembler/backend/ra.py`)

1. **Selection** to an IR over 32-bit virtual registers. PTX `mov` and pack/unpack are renamings, so they cost no instructions. Kernel params stay `c[0x0][…]` operands. `mul.wide` + `add.s64` fuse into one `IMAD.WIDE` (ptxas's idiom). Division and remainder by a constant use the exhaustively proven multiply-high sequences.
2. **Wide operands** (64/128-bit) are register groups that must be consecutive and aligned. Where a group is built from arbitrary values, copies are inserted first.
3. **DCE**, then **coalescing** (deletes copies with no conflict), then **constant CSE**.
4. **Linear-scan allocation** over per-value live intervals with copy-affinity hints, iterated until stable. A value last read by an instruction may share a register with that instruction's result.
5. **Scheduling** (`compile.py: schedule`): a write scoreboard on variable-latency results, and a read scoreboard on memory ops' source registers so later writes wait (WAR safety under register reuse). Stalls: `safe` = 6 after every fixed-latency op; `mined` = distance to the next dependent op using the **A100-measured** latencies (`hw_latency`), 13 cycles before any `@P` use of a fresh predicate.
6. Template **encoder**, then **splice** into ptxas's ELF skeleton for the same signature with the measured register count, EXIT offsets and symbol size.

Control flow (since 2026-10-30): PTX registers assigned more than once in a kernel with branches or guards get a fixed home register; liveness is a CFG dataflow, so values carried around back edges stay allocated; predicates are allocated to P0–P6 by interval coloring. Optional passes, used by the `sched` / `opt` bench variants:

- `openassembler/backend/sched.py`: basic-block list scheduler (critical path over measured latencies; memory ops keep their order).
- `openassembler/backend/unroll.py`: 4× unrolling of counted `do { B } while (++c < n)` loops, with a remainder loop. If B does not read c, one `add c, 4` per trip, with the exit test moved ahead of the body.

### Bench variants (`tools/bench_build.py`, `corpus/sm_80/bench/`)

| kernel (ms, A100, 4096 CTAs × 32 thr, 4096 iters) | ptxas | safe | mined | sched | opt | ptxas/opt |
|---|---|---|---|---|---|---|
| chain.ffma | 0.089 | 0.242 | 0.208 | 0.212 | 0.097 | 0.92× |
| chain.imad_lop3 | 0.129 | 0.286 | 0.246 | 0.224 | 0.135 | 0.95× |
| ilp4.ffma | 0.257 | 0.386 | 0.368 | 0.328 | 0.254 | 1.01× |
| ilp4.imad | 0.257 | 0.357 | 0.334 | 0.315 | 0.250 | 1.03× |
| loop.div7 | 0.629 | 0.687 | 0.646 | 0.566 | 0.464 | **1.36×** |
| loop.rem1000 | 0.373 | 0.457 | 0.421 | 0.388 | 0.312 | **1.19×** |
| loop.smem | 0.748 | 0.828 | 0.736 | 0.725 | 0.713 | 1.05× |

`safe` → `opt` is entirely scheduling and unrolling; the instructions per iteration are the same. ptxas still wins the dependent chains by 5–8%. Its unrolled loop body hides the counter update and branch inside the chain; ours ends each 4× trip with a 13-cycle predicate wait.

## Fast matrix multiplication: AlphaTensor schemes → our compiler → A100

**What AlphaTensor gives.** AlphaTensor (Fawzi et al., Nature 2022) found low-rank decompositions of the matrix-multiplication tensor. A rank-R scheme for ⟨n,m,p⟩ multiplies n×m by m×p blocks with R block products instead of nmp. Examples: Strassen ⟨2,2,2⟩ uses 7 instead of 8; ⟨4,4,4⟩ uses 49 instead of 64 over the reals (47 only in arithmetic mod 2, which floats cannot use); ⟨3,3,3⟩ uses 23. Applied recursively, this gives O(N^log_n R) arithmetic, for example N^2.807 for Strassen. The saving is in block multiplications. The price is extra block additions and memory traffic, plus somewhat larger rounding error. On a GPU, any gain only appears for large N and when the additions are cheap. DeepMind reported 8.5% over `jnp.dot` on a V100 at 8192 for their GPU-tuned ⟨4,4,4⟩ scheme, measured against XLA's GEMM, not cuBLAS.

**What is in the repo.**

| Piece | File |
|---|---|
| Upstream data (DeepMind repo, Apache-2.0) | `third_party/alphatensor/` (fetched through a Modal container; GitHub was unreachable from the Mac) |
| Import to a numpy-free spec (97 schemes incl. Strassen, the V100- and TPU-tuned 4×4×4, Strassen²) | `tools/fastmm_import.py` → `specs/fastmm/schemes.json.gz` |
| Exact verification of every scheme (Σ_r u⊗v⊗w = T over ℚ) | `openassembler/fastmm/factors.py: verify` (97/97 pass, 24 s) |
| Kernels as PTX: `sgemm128` (128×128 tile, 8×8 per thread, double-buffered smem), `lincombK` (K-term block sums), `fgemm_a_b_c` (fused: A/B sums formed while tiles load, `C_o += w_o·M` in the epilogue, no temporaries) | `openassembler/fastmm/kernels.py` |
| Our compiler builds all of them (73 kernels), each checked in the emulator (multi-warp, poisoned registers) against a Python reference | `tools/fastmm_build.py` → `corpus/sm_80/fastmm/ours@*.cubin` (ptxas builds next to them as `ptxas@*`) |
| Plans: one recursion level → launch list; unfused (lincomb → GEMM → lincomb) and fused (fgemm, ≤ 4 terms per side) | `openassembler/fastmm/plan.py` |
| GPU runner: generic multi-launch "plan" jobs, cuBLAS reference, host float64 check on 16 rows × 128 columns | `openassembler/gpu/cuda_driver.py: run_plan_job`, `tools/fastmm_run.py` |

**Measured, A100-SXM4-40GB, fp32, n = 8192** (best median of 3 reps × 2 shuffled rounds; `TFLOP/s*` = 2n³/time):

| plan | launches | ours (ms) | ptxas-compiled (ms) | ours vs cuBLAS | rel. error |
|---|---|---|---|---|---|
| cuBLAS SGEMM (reference) | 1 | 57.6 (19.1 TFLOP/s*) | – | 1.000× | 1.6e-6 |
| direct GEMM, `sgemm128` | 1 | 60.0 | 62.1 | 0.960× | 1.6e-6 |
| **Strassen ⟨2,2,2⟩ rank 7** | 21 | **58.4** | 60.6 (previous run; cut by budget in this one) | 0.986× | 2.3e-6 |
| **AlphaTensor ⟨2,2,2⟩ rank 7** | 19 | **58.3** | 60.6 | 0.988× | 2.7e-6 |
| Strassen² ⟨4,4,4⟩ rank 49 | 163 | 69.5 | 71.8 | 0.829× | 3.2e-6 |
| AlphaTensor V100-tuned ⟨4,4,4⟩ rank 49 | 174 | 71.6 | 74.0 | 0.804× | 5.7e-6 |
| fused Strassen ⟨2,2,2⟩ | 8 | 72.2 | – | 0.798× | 2.3e-6 |
| fused AlphaTensor ⟨2,2,2⟩ | 8 | 76.5 | 68.0 | 0.753× | 2.7e-6 |

Reading this:

- **One 2×2×2 level saves 12.5% of the multiplications and turns that into a 2.7% speedup over our own direct GEMM** (60.0 → 58.4 ms). The block additions eat the rest. That is still 1.4% short of cuBLAS, because our base GEMM is 4% slower than cuBLAS's.
- **4×4×4 schemes lose at this size.** They save 23% of the multiplications, but each of the 49 block GEMMs is 2048×2048 and gets only 256 CTAs (2.4 waves on 108 SMs), and 114–125 extra addition kernels run. The paper's 8.5% was against a slower baseline.
- **Our compiler's code is faster than ptxas's for the same PTX** in every unfused plan (by 2–4%). In the fused kernels ptxas wins: they need 110+ registers, and ptxas's scheduling of the 32 global loads per k-tile is better than ours.
- Errors grow with the scheme's coefficient sizes, as expected (V100 scheme 5.7e-6 vs 1.6e-6 for GEMM), and stay at single-precision level.

At n = 2048, all of these plans lose (cuBLAS 0.98 ms vs 1.9 ms for Strassen): the blocks are too small to fill the GPU.

## Reproduce

Every command behind every number here is in **[REPRODUCING.md](../REPRODUCING.md)**, which works on
Linux and macOS and says what each step needs (nothing / NVIDIA's toolkit / a GPU). It is the one
place the commands live; this file used to carry its own Mac-only copy, which drifted.

Files named in the tables above (`corpus/sm_80/*.json`, `*.raw.json.gz`) are committed evidence from
the recorded A100 runs; `corpus/sm_80/{ptx,sass,cubin,ours}/` are build output you regenerate with
`openassembler corpus` and `openassembler check`.

`third_party/alphatensor` was fetched from `raw.githubusercontent.com/google-deepmind/alphatensor/main/`
(files: `algorithms/factorizations_{r,f2}.npz`, `benchmarking/{factorizations,utils,run_gpu_benchmark}.py`,
READMEs). From the Mac, GitHub timed out, so the files were downloaded inside a Modal container.

Modal lessons from these runs:

- The client heartbeat runs in a Python thread. A minute of pure-Python work in the local entrypoint (building float64 references for n = 8192) starved it, and Modal stopped the app mid-run. One stopped app even left a container running until `modal app stop`. Host-side checks now run only after results are back, and with about 400 MB of memory.
- `--budget` cuts the remaining jobs: the n = 8192 run used all 580 s, so a few ptxas cells are missing. Raise the budget or run fewer schemes.

### D. Any SSH GPU host (or a local Linux GPU)

`python tools/gpu_run.py --ssh user@host [--which ours]` or `--local`. It needs only an
NVIDIA driver and python3 ≥ 3.8 on the remote. It has been tested with a simulated driver
but not yet against a real SSH host.

## Gaps and next steps

- **Broader tensor GEMM coverage:** the native FP16/BF16 full-tile SM80 path is now available through `gemm`. TF32, arbitrary tails/layouts and integration into PyTorch forward/backward graphs remain future work. Historical FP32 fast-matmul gains do not predict speedups against current tensor-core cuBLAS.
- **Fused fgemm kernels are slower than unfused.** They are memory-bound on the 2–4 extra 128-bit loads per k-tile, and the epilogue's read-modify-write of up to 4 C blocks adds traffic. Our scheduler also loses to ptxas on them (76.5 vs 68.0 ms).
- **4×4×4 schemes:** the block GEMMs are too small for 108 SMs at n = 8192. Try n = 16384 (needs 2 GB more buffers, OK on 40 GB), or two Strassen levels with fused additions.
- **Dependent chains:** ptxas still wins by 5–8%. It hides the loop counter and branch inside the chain; we pay a 13-cycle predicate wait per unrolled trip. Fix: compute the next trip's predicate one trip early (software pipelining).
- **Other GPUs:** the loader, plan runner and timing harness are arch-independent (`--gpu L4|A10|H100|…`), but our encoder and latency model are sm_80 only. The A10 (sm_86) shares the sm_80 encoding for everything we emit, though that is not yet verified. sm_89/sm_90 need their own corpus, synthesized templates and a latency sweep, all produced by the same tools.
- **Compiler gaps:** warp shuffles/votes, f16/f64 arithmetic, `ld.global.v2`, atomics, more than 7 live predicates, register spilling, and some non-default float rounding/encoding forms. The supported 0.2.0 arithmetic and pointer-loop paths are covered by measured fragments and regression tests.
- **Unmeasured hypotheses:** tensor-core handling of inf/NaN/subnormal inputs, DMMA order (matched on finite inputs), `QSPC` on real shared/local pointers.
