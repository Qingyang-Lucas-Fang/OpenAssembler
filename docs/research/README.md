# Research Index

This folder separates durable research conclusions from generated builds and
phase-by-phase experiment workspaces. The current consolidated record is in
these documents:

| Document | Scope |
|---|---|
| [A100 MMA results](A100_MMA_RESULTS.md) | Full chronological ledger of A100 tensor-core research, including positive, negative, failed and superseded results. |
| [Scalar compiler results](SCALAR_COMPILER_RESULTS.md) | Scalar A100 timing, cubin/encoding findings, compiler scheduling evidence and FP32 AlphaTensor study. |
| [A100 compiler contract](A100_MMA_COMPILER.md) | Current production SM80 GEMM compiler behavior, measurements, supported options and limits. |
| [SASS reverse engineering](REVERSE_ENGINEERING.md) | Methods, verified findings, evidence levels and links to the detailed encoding/cubin/bridge notes. |
| [Reproduction guide](../../REPRODUCING.md) | Current reproducible commands, retained evidence and provenance policy. |
| [Primary references](REFERENCES.md) | Literature and vendor documentation, with distinctions between hardware capability and measured result. |

Operational product documentation remains in [FRONTENDS.md](../FRONTENDS.md),
[TENSOR_GEMM.md](../TENSOR_GEMM.md), [STATUS.md](../STATUS.md), and
[REPRODUCING.md](../../REPRODUCING.md). Compact machine-readable A100 evidence
is indexed in [docs/evidence/a100-mma](../evidence/a100-mma/README.md). The
corrected instruction-latency record is in
[docs/evidence/a100-latency](../evidence/a100-latency/README.md).

The old experiment workspaces and generated build trees are retired as
documentation entry points. Their outcomes have been consolidated into the
two results ledgers; retained raw records are listed in the evidence indexes.
