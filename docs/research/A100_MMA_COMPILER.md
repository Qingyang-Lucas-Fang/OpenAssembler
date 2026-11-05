# Current A100 GEMM Compiler

This document describes the production SM80 path after the experiments
consolidated in [A100_MMA_RESULTS.md](A100_MMA_RESULTS.md). The public contract
is [TENSOR_GEMM.md](../TENSOR_GEMM.md); this research note records why the
current schedule is selected and what the latest evidence does and does not
establish.

## Execution model

`openassembler gemm` generates shape-specialized PTX. The standard tensor
backend parses and selects instructions, allocates register groups, computes
dependency scoreboards, applies HMMA issue controls and operand reuse hints,
encodes SM80 words, then splices them into ptxas's ELF container. `nvdisasm`
checks the resulting cubin. The GPU executes the cubin; Python is compile-time
orchestration and does not interpret each device instruction.

The validated direct schedule is a 128×128×64 CTA tile, four 64×64 warp tiles,
128 threads, two 32KiB shared-memory operand regions and 64KiB dynamic shared
memory. It uses FP16/BF16 inputs, FP32 accumulators/output, paired B-fragment
LDSM loads, XOR-swizzled shared operands, 16-byte asynchronous copies, a
specialized two-tile loop with a peeled drain and vectorized FP32 stores.
The scheduler preserves asynchronous completion, LDSM address WAR protection,
HMMA-to-scalar drains, and ordinary-memory scoreboards.

## Explicit options

The default `m_fast` CTA order and `none` epilogue retain the established
contract. `morton8` is a logical output-tile permutation over complete physical
8×8 CTA groups; M and N must be multiples of1024. It does not promise that
adjacent CTAs run on the same SM. The corrected mapping was revalidated on
November 5 after changing its group decode; those results supersede the
November 4 flattened-grid Morton timings.

`bias_residual` computes `(A @ B.T + Bias) + Residual`; `bias_residual_relu`
adds finite-value ReLU. Bias is contiguous FP32 `[N]`, residual is contiguous
FP32 `[M,N]`, and the kernel signature adds both pointers. Fusion is available
only with `stagepair`. The generic CLI smoke path zero-initializes auxiliaries;
the retained A100 test uses nonzero inputs and full-output checks.

Both controls are explicit because results are shape-specific. No tuner
automatically chooses Morton or fusion. In the tested square/smoke/MLP fusion
set, square wins while smoke regresses about20% and MLP is slightly slower.
The tested MLP Morton candidates also regressed.

## Latest matched A100 run

The current-code full-output run is on A100-SXM4-40GB. Each candidate passes
zero, integer and normal inputs; all output elements and guard regions are
checked. The normal FP32 reference error is below `6.7e-6` relative L2. Raw
samples and device metadata are retained in
[`validation_20261105T074142.327676Z.json`](../evidence/a100-mma/current-validation/validation_20261105T074142.327676Z.json).

| Candidate | Candidate ms | Matched OpenAssembler control ms | Relative time reduction | ptxas control ms |
|---|---:|---:|---:|---:|
| Tall FP16 Morton8, 16384×4096×4096 | 2.528 | m_fast 2.728 | 7.34% | 2.369 |
| Tall BF16 Morton8, 16384×4096×4096 | 2.478 | m_fast 2.652 | 6.54% | 2.327 |
| Square FP16 bias+residual, 4096³ | 0.743 | OA separate 0.781 | 4.95% | ptxas fused 0.639; separate 0.742 |
| Square BF16 bias+residual, 4096³ | 0.729 | OA separate 0.770 | 5.26% | ptxas fused 0.627; separate 0.733 |
| Square FP16 bias+residual+ReLU, 4096³ | 0.741 | OA separate 0.783 | 5.34% | ptxas fused 0.630; separate 0.746 |
| Square BF16 bias+residual+ReLU, 4096³ | 0.729 | OA separate 0.768 | 5.16% | ptxas fused 0.624; separate 0.734 |

The current Morton implementation is about6.5–7.3% faster than the current
OpenAssembler m_fast control, but remains slower than ptxas by about6.7–6.9%.
Fused OpenAssembler square operations are about5% faster than the matched
OpenAssembler two-kernel path and are roughly even with the ptxas two-kernel
path; the ptxas fused cubin remains faster. These results improve the
OpenAssembler baseline, not cuBLAS-level performance.

The November 4 run remains recorded as a superseded result because its Morton
address decode flattened rectangular CTA grids differently. It must not be
combined with or cited instead of the November 5 current-source run.

## Scope not covered

The path is full-tile RCR GEMM, not a general transformer compiler. It does not
support arbitrary tails, transposed dX/dW layouts, TF32, arbitrary alpha/beta,
full attention, optimizer state, or a complete forward/backward training step.
No throughput or tokens-per-second claim follows from these kernel timings.
