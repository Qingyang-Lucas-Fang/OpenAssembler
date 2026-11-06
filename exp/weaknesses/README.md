# Validated weakness-remediation experiments

The canonical findings, reproduction commands and negative results are now in
individual dossiers rather than round-specific reports.

| Subject | Canonical dossier | Drivers and durable records |
|---|---|---|
| GEMM pointer homes | [Direct pointers](../../docs/research/ideas/direct-pointer-homes.md) | `gemm_pointer_homes.py`, three primary run JSON, static/summary/counter JSON |
| Copy forwarding control | [Post-allocation forwarding](../../docs/research/ideas/post-allocation-pointer-forwarding.md) | `gemm_pointer_forwarding.py` and its identical-stream result |
| Whole-step impact | [Native hybrid MLP](../../docs/research/ideas/native-hybrid-mlp.md) | `gemm_pointer_homes_mlp.py`, two paired device runs |
| Cooperative softmax | [Softmax](../../docs/research/ideas/cooperative-softmax.md) | `reductions_a100.py`, `reductions_runs/` |
| Cooperative LayerNorm | [LayerNorm](../../docs/research/ideas/cooperative-layernorm.md) | Same checked numerical/graph driver; per-operator contract |
| Scalar dependencies/warp sync | [Correctness fixes](../../docs/research/ideas/scalar-dependency-correctness.md) | `reductions_warpsync_template.py`, oracle evidence, scalar regression and production identity |
| Complete state checking | [Diagnostics](../../docs/research/ideas/state-verification-diagnostics.md) | `bf16_verification.py`, pinned A100 validation |
| Rejected precision repair | [Cast preservation](../../docs/research/ideas/bf16-cast-preservation.md) | `bf16_precision_casts.py`, failed unchanged BF16 gate |

The actual devices include A100-SXM4-40GB and 80GB; comparisons are paired
within each run and absolute times are not pooled. The git baseline for the
recorded pre-fix controls is `3507e6198a9323060241ab05c4558052d7bac2b3`.
Generated `*_build*` and CLI build trees are disposable caches, not reports.
Production modules do not import the experimental monkeypatches.

`reductions_production_identity.json` pins all 18 complete native streams and
the final generator/backend/spec fingerprints. `cli_rowwise_smoke.json` is the
separate ordinary-CLI, NaN-poisoned zero-input proof. `validation.json` records
completed checks; it is not a new benchmark for this documentation reorganization.

All `reductions_runs/a100-*.json`, including failed candidates, remain. Early
intermediate revisions were not all archived, so they are not positive
reproducible controls. Later source versions are preserved by hash in
`reductions_history/`; current source/stream identity is pinned independently.
The scalar paired-regression driver enforces the retained pre-fix binary hash
before GPU submission, rather than relabeling a current build as a baseline.

`evidence_manifest.json` checks current artifact/documentation custody through
`python3 exp/weaknesses/check_evidence.py`. Its earlier narrative fingerprints
are preserved in the [documentation snapshot](../../docs/evidence/documentation-history/README.md).
Raw hardware result values and production source identities remain unchanged.
