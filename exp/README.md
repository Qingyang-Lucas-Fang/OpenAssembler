# Experiment drivers and evidence

This directory contains isolated research drivers and raw results. It is not an
alternative documentation tree and is not imported by the production compiler.
The canonical findings and exact reproduction commands are in the
[per-idea catalogue](../docs/research/README.md).

| Workspace | Use |
|---|---|
| [a100_counters](a100_counters/README.md) | Matched native/library counter and graph controls |
| [whole_stack](whole_stack/README.md) | Original transformer and hybrid-MLP application/stage study |
| [weaknesses](weaknesses/README.md) | Validated compiler/operator follow-ups and rejected controls |
| [nn_round2](nn_round2/README.md) | Native arithmetic, dense training products, composed networks, legal source-template candidates and counter-guided follow-ups |

Keep raw samples, failed gates and identities; do not create another phase
report that duplicates a dossier. Generated build/payload caches remain
noncanonical. [Development conventions](../docs/DEVELOPMENT.md) explain evidence
levels, promotion requirements and retained documentation provenance.
