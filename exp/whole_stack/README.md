# Whole-stack and native-stage evidence

The complete original cross-cutting study is
[WHOLE_STACK_RESULTS.md](../../docs/research/WHOLE_STACK_RESULTS.md). Per-feature
methods, commands and limitations now have canonical dossiers:

| Subject | Canonical dossier | Principal drivers |
|---|---|---|
| Synthetic transformer suite | [Whole-transformer evaluation](../../docs/research/ideas/whole-transformer-evaluation.md) | `run.py`, `model.py`, `analyze.py` |
| Configured CLI capture/replay | [Application capture](../../docs/research/ideas/pytorch-application-capture.md) | `cli_proof/prove.py` |
| Native forward/dX/dW MLP | [Hybrid MLP](../../docs/research/ideas/native-hybrid-mlp.md) | `native_mlp.py` |
| Native tensor counters | [Counter profiling](../../docs/research/ideas/tensor-counter-profiling.md) | `native_counters.py` |
| Serial scalar stages | [ReLU](../../docs/research/ideas/native-relu.md), [GELU](../../docs/research/ideas/native-gelu.md), [Adam-like update](../../docs/research/ideas/fused-adam-like-update.md) | `native_ops.py` |
| BF16 sensitivity | [Adam epsilon diagnostic](../../docs/research/ideas/adam-epsilon-sensitivity.md) | `bf16_diagnostic.py` |

`runs/` retains successful, failed and superseded JSON, and `native_static.json`
retains the native instruction inventory. The dossiers label which records are
admitted; a JSON timing beside a failed gate is not a speedup. Historical
snapshot manifests and superseded prose remain traceable through
[documentation provenance](../../docs/evidence/documentation-history/README.md).
Raw source/measurement fingerprints are retained. This directory no longer
contains duplicate native/attempt/operator narrative reports.
