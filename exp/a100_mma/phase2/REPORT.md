# A100 hardware-counter and compiler optimization round

2026-11-03. Production compiler/encoder/specification files remain unchanged.
Root README.md and REPRODUCING.md were updated as explicitly requested; the linked
pytest-fixture README was left alone. All implementation experiments stay under
`exp/a100_mma/`. The first-round [report](../REPORT.md) remains the historical baseline.

## Measured progress

A compiler change now updates loop-carried 64-bit pointers directly in their
allocated homes with carry-linked IADD3 instructions. It removes redundant
moves/materialization around pointer updates, reducing the chosen kernel from
1,898 to 1,812 static non-NOP instructions. It does not change the MMA algorithm,
precision, shared layout, tile or output contract. Carry and wrap cases were
tested independently, then all six projections were compared in the same warmed
GPU worker on identical inputs with complete output verification.

| M × N × K | Input | Baseline ms | Pointer optimization ms | Paired speedup | cuBLAS ms |
|---|---|---:|---:|---:|---:|
| 4096 × 4096 × 4096 | f16 | 0.842 | 0.802 | 1.051× | 0.582 |
| 4096 × 16384 × 4096 | f16 | 3.136 | 2.975 | 1.054× | 2.386 |
| 16384 × 4096 × 4096 | f16 | 3.700 | 3.645 | 1.015× | 2.306 |
| 4096 × 4096 × 4096 | bf16 | 0.817 | 0.777 | 1.052× | 0.544 |
| 4096 × 16384 × 4096 | bf16 | 3.093 | 2.951 | 1.048× | 2.093 |
| 16384 × 4096 × 4096 | bf16 | 3.660 | 3.629 | 1.009× | 2.097 |

The square/MLP gain is about 5%; tall-shape gains are 0.9–1.5%. Absolute timings
vary with clocks. This table is paired within each worker, with 16 additional
warm-up replay rounds, seven shuffled timing rounds, and 20 nodes per graph.
It is not a claimed transformer-training speedup, and cuBLAS remains faster.

A subsequent tensor-order/reuse experiment uses the same complete B-fragment
set, traverses output columns first, and hints reuse of the B operand by the next
HMMA. A broad scheduling comparison reached 0.757 ms versus 0.783 ms for the
pointer-optimized row-order baseline in that worker. Shape-specific confirmation
is below; timing noise is material for small differences.

| Shape / input | Pointer baseline ms | Column-order/reuse ms | Paired speedup |
|---|---:|---:|---:|
| square / f16 | 0.817 | 0.802 | 1.018× |
| mlp / f16 | 3.030 | 2.992 | 1.013× |
| tall / f16 | 3.716 | 3.694 | 1.006× |
| square / bf16 | 0.794 | 0.784 | 1.013× |
| mlp / bf16 | 2.996 | 2.964 | 1.011× |
| tall / bf16 | 3.692 | 3.682 | 1.003× |

## Hardware counters

Real Nsight Compute 2025.1 reports were collected on an A100-SXM4-40GB.
`profiles/*/profile.json` records commands, selected metrics, return codes and
binary provenance; `.ncu-rep`, raw CSV and parsed summaries are retained.

For the original passing 128×128×64 kernel:

| Counter | OpenAssembler | ptxas on identical PTX |
|---|---:|---:|
| Tensor-pipe cycles active / peak sustained active | 59.43% | 78.26% |
| Active warps / peak sustained active | 11.91% | 12.12% |
| Short-scoreboard stall percentage | 13.79% | 1.15% |
| Long-scoreboard stall percentage | 11.94% | 12.05% |
| Wait-stall percentage | 43.89% | 37.17% |
| Barrier-stall percentage | 2.73% | 3.34% |
| Executed instructions | 90.85 million | 69.62 million |

These counters support a data-supply/dependency and instruction-overhead
problem. They do not identify every operand-collector or register-bank event;
we did not invent unavailable direct register-bank counters. Profiling replay
flushes caches and uses different timing conditions from graph benchmarking,
so the Nsight duration is not substituted for the performance table.

For a 64×32 warp tile, bounding live B fragments reduced registers 132→126.
Nsight active warps rose from 12.49% to 23.93%; tensor activity rose from 49.45%
to 58.69%. In a paired graph run that window was 1.205× faster than its own
132-register variant, but only comparable to the best existing 64×64 warp tile.
This validates occupancy as a useful mechanism, not a new overall winner.

## SASS scheduling, clocked probes and synchronization

Exact-SASS probes avoid a flaw in the first attempt: generic PTX lowering can
insert fragment copies between successive MMAs. The rejected first clock probe
measured 32 cycles per intended operation because it timed those extra moves.
The final exact sequence has verified checksums, saved SASS and binary hashes.

* One dependent accumulator stream and eight independent streams both show an
  **8-cycle/op elapsed slope** over 16/64/128/256 HMMA instructions, even with
  encoded S2/S4. This is one warp on one CTA, not peak full-SM throughput or
  a proof that every HMMA RAW latency is eight cycles.
* A serialized LDSM x4 → dependent LOP3 → dependent address IADD3 chain shows
  **41 cycles/iteration**. This includes both ALU consumers and scoreboard
  handling; it is not a 41-cycle pure shared-memory latency claim.
* A one-field mutation of the passing 4096³ cubin tested BAR S1/2/4/6/8 and
  DEPBAR S1/2/3/4/6. BAR variants passed. DEPBAR S1/2/3 failed full outputs;
  S4/6 passed. This corrects the earlier joint diagnosis: the experiment does
  not establish BAR≥6 as necessary. The default retains it conservatively.
* Reusing a scoreboard ID without explicit retirement passed this limited
  experiment, but did not establish a useful improvement; the undocumented
  hypothesis remains disabled. Blindly deleting address read dependencies
  already caused failures. The passing alternative binds address WARs to load
  completion, including multiple outstanding readers.

## Pipeline depth, memory layout and shape matching

Two, three and four async shared stages were implemented and validated for
32/64-wide K tiles. Three/four stages use wait-group thresholds 1/2; their
instruction fields were bit-probed and round-tripped. Ring buffers have peeled
tails, with no speculative input loads. All six large variants passed; in the
64-wide K family, two stages took 1.003 ms versus 1.178/1.180 ms for three/four
in those runs. Added shared allocation and control work did not pay off.

Shared-memory A/B XOR masks 7/7, 7/3, 3/7 and 0/0 were tested in one worker.
7/7 took 0.807 ms; 7/3 0.836 ms; 3/7 0.911 ms; no swizzle 2.255 ms.
The no-swizzle profile showed 234.9 million shared-load bank conflicts and
65.35% short-scoreboard stalls, compared with 0.170 million and 13.79% in the
original swizzled profile. A100 physical banks were not changed; software
address layouts changed. The asymmetric layouts tested did not win.

Virtual CTA grouping 2/4/8/16 changed L2-oriented tile order while preserving
the exact launch extent. None beat the original square order in the paired
run. This is not a hardware cluster and provides no guarantee of SM co-location
or access to another SM’s shared memory. Warp tiles 32×32, 32×64, 64×32 and
64×64 have now been measured: smaller register counts alone do not imply higher
throughput, because they change reuse, CTA geometry and dynamic load work.

## Layouts, tails, epilogue and autograd

`integration_worker.py` exposes an experimental `torch.autograd.Function` for
a linear layer. The forward cubin fuses bias. Backward uses generated GEMMs for
dX and dW and a PyTorch bias reduction. Logical dimensions 130×198×257 exercise
all three tails and transposed input views. Explicit contiguous copies and zero
padding adapt them to the native RCR full-tile kernel; **native masked tails and
direct arbitrary-stride layouts are still pending**.

For both FP16 and BF16, full forward/dX/dW comparisons with a CPU float64
oracle passed, followed by three squared-loss/SGD steps with final output,
weight, bias and losses checked against a matched explicit reference. Small-step
host wall timing includes Python/ctypes dispatch, allocations, padding, casts,
transposes and optimizer work: about 0.98 ms for ours versus 0.68 ms reference.
The reference uses explicit FP32 products with TF32 disabled and the same
low-precision dY policy; it is not an optimized production-autocast baseline.
This establishes an integration hook, not a transformer-step acceleration.

## Numerical behavior, sparsity and hardware feasibility

A 2025 formal-model paper distinguishes alignment truncation from global IEEE
rounding. The discriminant `32 - 2**-40` returned FP32 32 in both our generated
cubin and ptxas’s cubin; IEEE RTZ would return its FP32 predecessor. This
supports correcting the README’s broad RTZ wording; it does not prove the
repo’s complete fitted model, which still needs discriminating guard-bit and
subnormal tests.

Three native `mma.sp.m16n8k32` capability probes (positive, negative and zero)
passed and disassembled as `HMMA.SP.16832`. They use **ptxas oracle code**,
not a new OpenAssembler sparse selector, and establish no sparse training
speedup. FP8 MMA, TMA, DPX and thread-block clusters require hardware that A100
does not have. Our [literature/feasibility table](LITERATURE.md) evaluates
software substitutes and the real cost of each.

Compile-time Strassen/AlphaTensor plans now verify the real-arithmetic identities
and emit constant transform DAGs with a single live product tile. No coefficients
would be device data. New GPU low-rank lowering is deliberately gated on the
direct kernel’s competitiveness and transform cost. Rank 7 saves 12.5% of block
products before additions; it cannot by itself remove the currently measured
direct-kernel gap. SubCuber is the directly relevant 2026 compiler comparison.

## Validation, artifacts and remaining work

`summary.json` joins counter, clock, mutation, correctness and paired-timing
records. Workload JSONs include binary/source hashes and snapshots; failed
variants remain evidence. GPU data files are under the parent `results/`, with
immutable run copies in `runs/`. `final-tests.log` records the selected CLI,
backend and experimental regression checks. `tree-check.json` verifies that
only the two explicitly authorized root docs changed outside `exp/`.

Priorities now are: profile the best schedule’s remaining short-scoreboard and
wait stalls; pipeline fragment loads without dropping WAR edges; build a
resource-aware selector across measured shapes; add native masked/strided
lowering; and integrate a complete transformer step. Only then should the
low-rank GPU experiment be judged under matched numerical policy. Counter
access worked here, so no hardware simulator was substituted for measurements.

Reproduction commands are in [the main guide](../../../REPRODUCING.md#a100-hardware-profiling-and-follow-on-experiments).
The research survey links each technical proposal to primary sources and marks
architectural proposals separately from features implementable on existing A100.
