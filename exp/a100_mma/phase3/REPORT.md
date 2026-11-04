# A100 phase three: measured tensor-pipeline improvements

Later native CLI promotion and further experiments are in
[`phase4/REPORT.md`](../phase4/REPORT.md). This report retains the phase-three
research state and its performance comparisons.

2026-11-04. Experiments remain under `exp/a100_mma/`; the production compiler
is unchanged. This is an experimental PTX→OpenAssembler register allocation,
SASS scheduling/encoding→cubin-body pipeline. NVIDIA ptxas supplies the ELF
container and a reference compilation of the same PTX. The measured OpenAssembler
kernel body is independently generated; cuBLAS is a comparator.

## Repeated projection results

Pair two B fragments in an `LDSM.x4` load, use its aligned register halves
directly in HMMA, store adjacent FP32 accumulator elements together with one
output-row pointer, and specialize the alternating shared stages in a two-tile
loop. The final same-worker comparison below measures the old direct schedule,
the paired-fragment/epilogue combination and the stage-specialized combination.
The individual improvements have separate confirmation runs. We retain gap8
everywhere; we do not select a different winner for each shape.

| M × N × K | Inputs | Prior direct baseline ms | Combined ms | Less time | Matched cuBLAS ms | cuBLAS throughput fraction |
|---|---|---:|---:|---:|---:|---:|
| 4096 × 4096 × 4096 | FP16 | 0.810598 | 0.641690 | 20.84% | 0.595354 | 92.78% |
| 4096 × 16384 × 4096 | FP16 | 2.967808 | 2.460621 | 17.09% | 2.433741 | 98.91% |
| 16384 × 4096 × 4096 | FP16 | 3.656602 | 2.726656 | 25.43% | 2.350592 | 86.21% |
| 4096 × 4096 × 4096 | BF16 | 0.783462 | 0.628378 | 19.79% | 0.555878 | 88.46% |
| 4096 × 16384 × 4096 | BF16 | 2.908570 | 2.439322 | 16.13% | 2.136166 | 87.57% |
| 16384 × 4096 × 4096 | BF16 | 3.634995 | 2.664806 | 26.69% | 2.125466 | 79.76% |

Exact values and result-file hashes are recomputed by `evidence.py` into
[`evidence-summary.json`](evidence-summary.json). The first search run gave
15.38–18.70% less time and 71.85–96.38% of cuBLAS throughput for the same fixed
combination without stage specialization. Its separate confirmation repeats
14.63–18.45% less time. Stage specialization then improves all six shapes in
another run: about 2% square/MLP and 13.5–13.8% tall. The final comparison above
includes all three schedules with24 warmup rounds, making the reported total
improvement a direct same-worker measurement. Clocks are unlocked; run-to-run
clock differences preclude mixing absolute numbers between runs.

The table uses actual A100-SXM4-40GB records, FP16/BF16 inputs and FP32
accumulation/output, `C=A@B.T`, with A and B contiguous in K. It compares to
`cublasGemmEx` with compute32F and its tensor-op default algorithm. CUDA graphs
exclude Python launch overhead: seven shuffled rounds,20 launches per graph,
median per launch. This is neither whole-step training speed nor a universal
cuBLAS parity result. A separate cuBLASLt heuristic sweep tests the vendor
baseline more broadly.

Every timed candidate first passes eight input cases: zero/integer/normal/scaled
data and two seeds. Every output element is checked against the matched cuBLAS
FP32 output, plus sampled CPU FP64 references; integer/zero checks are exact.
Normal/scaled relative L2 tolerance is2e-5. Output guards and finite/unwritten
checks remain active. See the worker and retained `checks` rows for the exact
contract. Final six-case comparison has40 check rows per case, each covering the
entire matrix. Full integer SASS emulator checks also run with poisoned registers.

## Why the combination works

[`search_REPORT.md`](search_REPORT.md) records30 candidates per size across
two dtypes, including aligned register permutations, role-aware placement,
reuse controls, shared swizzles, fragment order and K-fragment overlap.
[`search_combined_REPORT.md`](search_combined_REPORT.md) documents the36-kernel
composition suite and exact cubin word/control/operand checks.

* B pairing reduces total static LDSM instructions96→64, keeping256 HMMA.
  Both halves of the loaded quad remain explicit dependency operands, including
  the scalar operand hidden by the printed SASS group base.
* The row epilogue reduces FP32 stores128→64 and repeated wide address
  calculations144→24. Adjacent accumulator registers feed the64-bit store.
* Combined static scheduled instructions fall1820→1128; registers rise228→232.
  The lower instruction count is descriptive evidence, not the timing argument.
* Moving the paired-address adjustment out of the loop reduces instruction
  count further but loses performance. Most register-placement perturbations
  and asymmetric shared swizzles also lose. We retain all results.

Real Nsight counters support a better supplied tensor pipeline: baseline→combined
tensor-pipe activity63.51→75.51%; short-scoreboard stall fraction14.05→11.12%;
shared-load bank conflicts170,940→57,238; active warps remain approximately12%.
Tensor instruction count is unchanged, and shared load wavefront count changes
little. The wait-stall fraction rises, so these fractions are not additive cycle
savings. Matched-PTX ptxas still reaches85.44% tensor activity and3.995%
short-scoreboard stalls, leaving a concrete dependency/scheduling target.
Profiler replay durations are not substituted for graph benchmark times.
Metric definitions, full tables and retained reports are in
[`PROFILE_REPORT.md`](PROFILE_REPORT.md).

## Studying the actual vendor kernels

`cublas_target.py`, `cublas_modal.py` and `cublas_probe.py` observe the installed
library's actual matched-precision dispatch, launch resources, counters and
resident SASS using NVIDIA tools. Vendor disassembly stays in the ephemeral
container; local records retain derived instruction mixes and launch data.
No vendor kernel module is used as an OpenAssembler implementation.

FP16 chooses a CUTLASS-named SM80 128×128 kernel; BF16 chooses a128×256 kernel.
The vendor tensor-pipe activity is roughly86–90%, with much lower short-scoreboard
stall fractions. Both use LDSM.x4 fragment loads, vector stores, and uniform
integer operations. This observation motivated the independent pairing and
layout experiments. A kernel name alone does not prove its stage protocol or
algorithm. In particular, the FP16 grid has a third dimension of2; split-K is
an inference until connected to an algorithm configuration. See
[`CUBLAS_OBSERVATION.md`](CUBLAS_OBSERVATION.md) for the exact library version,
full launch sequence and limits of these inferences.

The public cuBLASLt sweep returns 10 unique FP16 and 13 BF16 algorithms across
zero/32-MiB workspace queries. All 108 full-output checks pass, including native
controls, and algorithm attribute reads pass ABI validation. This sweep uses
the paired-fragment/epilogue control before stage specialization. In the same process,
FP16 OpenAssembler takes 0.671693 ms versus selected Lt 0.576461 ms;
BF16 takes 0.655462 ms versus 0.563917 ms: native time is 16.52%/16.23% longer.
Both selected algorithms have split-K 1 and zero workspace. Native is slower in
all seven rounds. The small selected Lt gains over default GemmEx need independent
confirmation; selection and assessment share the same seven rounds. A 32-MiB
query can omit a faster zero-workspace candidate, so the union of feasible
algorithms is distinguished from the result of one query. Exact attributes,
checks and paired samples are in [`lt_README.md`](lt_README.md) and
`lt_audit.json`. These observations strengthen the benchmark target.

## Aggressive candidates with negative or limited results

* **Software producer warps:** five variants received semantic checks. Four
  pass A100 full-output smoke and large checks; the544-thread variant exceeds
  resources and is retained as a failure. The remaining variants take1.019–1.943ms
  at4096³, slower than the unified kernel. CTA barriers, inactive warp groups
  and register/occupancy costs remain material. This rejects this implementation,
  not every possible producer/consumer pipeline. See
  [`producer_README.md`](producer_README.md).
* **Larger tiles/deeper stages:** independently test128×256 and256×128 tiles,
  BK32/64 and up to five stages, retaining B pairing and the row epilogue.
  All14 smoke and14 large binaries pass. At4096³ the combined128²/BK64/two-stage
  kernel takes0.633/0.631ms in this run; the larger BK64 tiles take0.688–0.715ms,
  and the BK32 deeper pipelines take0.771–1.122ms. Five-stage support uses a
  separately disassembled DEPBAR threshold3 and a poisoned full-output
  fill/wrap/drain test. Vendor geometry cannot simply be copied into a different
  scheduler and assumed faster.
* **LDGSTS immediate addressing:** all128 single-bit mutations plus89 multi-field
  roundtrips establish signed12-bit global byte offsets and signed20-bit shared
  byte offsets for the exact tested instruction form. K4096 chunk offsets
  exceed the global range. Shared offsets fit and enable232→216 registers,
  with full semantic/disassembly proofs. Shared folding alone loses about 7%
  despite fewer registers/instructions. Combining it with two-tile stage
  specialization wins 2.67% FP16 / 2.12% BF16 against the already improved
  row/B-pair kernel in the first paired square run. Six-projection confirmation
  passes, and the tall improvement repeats in the final suite. The unrolled body
  keeps the complete wait/barrier
  protocol and peels the final two tiles. See [`address_report.md`](address_report.md).
* **Joint warp layout/register threshold:** for both eight-warp layouts,
  shared folding reduces 136→128 registers and crosses a residency threshold.
  A100 counters confirm active warps 12.48→23.99% and tensor activity
  59.09→72.98% for the profiled FP16 64×32 case. Large-kernel time improves
  about 20% over its unfolded control, yet remains slower than the four-warp
  reference. The selected shared-memory carveout also changes, so this is not
  an isolated register-file experiment. Small smoke shapes favor the eight-warp
  layouts; shape-dependent dispatch remains relevant. See
  [`address_occupancy_report.md`](address_occupancy_report.md).
* **Fused low-rank staging:** a native OpenAssembler Strassen microprototype
  uses compile-time ±1 factors, packed FP16 transforms, cached base quadrants,
  one live product and FP32 reconstruction. Rank7 and matched rank8 pass12
  hardware checks over96×96×32. Rank7 uses14 instead of16 static HMMA but
 940 instead of879 instructions and94 instead of74 registers. Its measured
 1.3% advantage is near the launch floor, not evidence for faster large GEMM.
  FP16 transform rounding changes the direct numerical contract; the explicit
  matched transform oracle records this difference. This is a functional start
  toward SubCuber-style fusion, not a cuBLAS or training speed claim. See
  [`lowrank_README.md`](lowrank_README.md).

## Reproduce and continue

The final combined local suite passes 46 tests together, including exact
binary/control/implicit-operand checks and poisoned full-output oracles. Test
output is in `pytest-final.log` (240 seconds). Hardware result records retain binary hashes,
source metadata, all seven timing samples and complete-output check counts;
`runs/<timestamp>/` preserves immutable copies. Negative and failed-resource
experiments remain visible.

`final_suite.json` reproduces the headline table after the builders have run;
`confirmation_suite.json` retains the earlier two-component comparison.
Use [`REPRODUCING.md`](../../../REPRODUCING.md#further-a100-mma-experiments) and
the individual reports for exact commands. Local compilation uses the repo's
Python3.11+ environment; Modal uses the authenticated system client. The compiler
remains Python. Rewriting host code cannot improve an already captured device
instruction stream; compiler speed and device speed are separate measurements.

The evidence now prioritizes reducing the remaining LDSM/scoreboard gap and
register/address lifetimes together. For the tall stage-pair improvement,
counters show unchanged occupancy and short-scoreboard stalls, reduced long
scoreboard stalls and DRAM traffic, and a higher L2 hit rate. This supports a
memory-supply/cache explanation without proving a single cause. Fixing the
shared carveout in folded/unfolded controls is a useful next ablation before
assigning a gain solely to registers. See [`PROFILE_FOLLOWUP.md`](PROFILE_FOLLOWUP.md).
Then add native transpose/strided/boundary lowering for dX/dW and a fused
forward/backward path. The phase-two padded hook is a correctness integration
probe; a real training-step benchmark is still required. For low-rank work,
continue with larger streamed tiles and matched precision after direct schedule
and resource improvements stabilize.

Primary literature remains in [`phase2/LITERATURE.md`](../phase2/LITERATURE.md),
including SubCuber's closest prior art. Additional NVIDIA primary documentation
and the pinned CUTLASS SM80 mainloop are retained with hashes in
[`sources/retrieval.json`](sources/retrieval.json). These distinguish physical
hardware features from software approximations and measured observations from
algorithm-name inferences.
