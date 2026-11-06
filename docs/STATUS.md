# OpenAssembler Status

Updated 2026-11-06. This page is the short operational status; the detailed
history, failed experiments and superseded measurements are in the
[A100 research ledger](research/A100_MMA_RESULTS.md).

## Current production paths

The CLI supports the validated SM80 scalar compiler and the native A100 tensor
path. The tensor path generates full-tile `C = A @ B.T` kernels for FP16/BF16
inputs, FP32 accumulation/output, and M/N/K dimensions divisible by 128. It
uses paired `LDSM`, `cp.async` staging, XOR-swizzled shared memory, HMMA and a
vectorized FP32 epilogue. The production contract and unsupported forms are in
[TENSOR_GEMM.md](TENSOR_GEMM.md) and [A100_MMA_COMPILER.md](research/A100_MMA_COMPILER.md).

Two optimizations are explicit CLI choices:

- `--cta-order morton8` uses complete aligned 8x8 CTA groups and currently
  requires M/N multiples of 1024.
- `--epilogue bias_residual` and `bias_residual_relu` fuse FP32 bias/residual
  work for the supported square path.

Neither option is an automatic policy. Their measured scope is one A100 shape
family, and neither establishes whole-transformer or whole-training speedup.

## Latest A100 validation

The current-source run passed full-output zero, integer and normal inputs,
guard canaries and matched-control checks for two Morton and four fused
kernels. CUDA-graph medians from nine shuffled rounds were:

| Candidate | OA candidate | OA control | Reduction | ptxas reference |
|---|---:|---:|---:|---:|
| Tall FP16 Morton8, 16384x4096x4096 | 2.528 ms | 2.728 ms | 7.34% | 2.369 ms |
| Tall BF16 Morton8, 16384x4096x4096 | 2.478 ms | 2.652 ms | 6.54% | 2.327 ms |
| Square FP16 bias+residual | 0.743 ms | 0.781 ms | 4.95% | fused 0.639 ms |
| Square BF16 bias+residual | 0.729 ms | 0.770 ms | 5.26% | fused 0.627 ms |
| Square FP16 bias+residual+ReLU | 0.741 ms | 0.783 ms | 5.34% | fused 0.630 ms |
| Square BF16 bias+residual+ReLU | 0.729 ms | 0.768 ms | 5.16% | fused 0.624 ms |

The complete JSON, validator, manifests and matched cubins are retained under
[`docs/evidence/a100-mma/current-validation`](evidence/a100-mma/current-validation/).
The first Morton run used a flattened rectangular-grid decode and is retained
only as superseded evidence.

## Scalar compiler and tests

The scalar backend has measured constant division/remainder lowering,
control-flow liveness, register allocation, A100 latency-aware scheduling and
SM80 cubin splicing. The current repository test suite is the authority for
the exact count in a given checkout; toolkit- or corpus-dependent tests report
an explicit skip when their external dependency is unavailable. Run:

```sh
uv run pytest -q
openassembler check
```

The scalar A100 latency evidence is the corrected 42-case sweep in
[docs/evidence/a100-latency](evidence/a100-latency/README.md). Its 4-cycle
within-pipe, 5-cycle cross-pipe and 13-cycle predicate-to-guard thresholds are
measured classes for tested encodings, not a universal latency table.

## What remains open

The isolated November 6 [whole-stack study](research/WHOLE_STACK_RESULTS.md)
measures ten transformer configurations in two process-order rounds, plus
native-GEMM hybrid MLP steps and scalar stages. Nine transformer configurations
pass both numerical comparisons; BF16 training fails both. The application
gains come through the Inductor/Triton frontend. This is research evidence,
not a new production native-transformer contract.

- Counter-guided operand-collector, shared-bank and scoreboard work is still
  needed to close the direct GEMM gap to ptxas/cuBLAS.
- Full dX/dW transposes, tails, TF32, fused attention/normalization and a
  complete forward/backward/optimizer step are outside the current contract.
- Three/four-stage pipelines, producer/consumer warps, tested fragment windows,
  L2 persistence settings and Hilbert traversal did not beat the selected
  direct schedule on the measured workloads.
- AlphaTensor and Strassen require compile-time factors, matched FP32 output,
  bounded live products and a competitive mainloop before another large-shape
  comparison is meaningful.
- A100 lacks Hopper TMA, thread-block clusters, DPX and the H100 FP8 Tensor
  Core path. These cannot be enabled by a compiler flag.

See [REPRODUCING.md](../REPRODUCING.md) for current commands and
[research/README.md](research/README.md) for the complete evidence map.
