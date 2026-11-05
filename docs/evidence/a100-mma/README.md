# A100 MMA Evidence Index

This directory retains the selected raw records cited by the
[A100 research ledger](../../research/A100_MMA_RESULTS.md). The ledger states
comparison scope and conclusions; these files keep device identity, checks,
samples, source hashes and candidate-level details. Removed build caches and
repeated per-candidate PTX/SASS are not included except for the current
production validation and the archived initial low-rank artifact bundle.

## Retained records

| Path | Coverage |
|---|---|
| `legacy-overview.json` | Original 191-entry evidence summary for the first backend and early variants. |
| `phase2/summary.json` | Counters, clocked/synchronization sweeps, paired improvements, integration hooks, sparse oracle check and numeric observations. |
| `phase2/integration-results.json` | Padded/autograd integration timing and checks. |
| `phase2/clocked-exact.json` | Corrected exact-SASS dependency clock probes. |
| `phase2/literature-discovery.json` | Hardware capability and literature discovery record. |
| `phase2/sparse-results.json` | Sparse MMA feasibility; ptxas oracle only. |
| `phase3/evidence-summary.json` | Direct GEMM candidate measurements, stage specialization and address evidence. |
| `phase3/profile-followup.json` | Nsight counter summaries and occupancy follow-up. |
| `phase3/search-summary.json` | Search candidates, timing and full-output status. |
| `phase3/epilogue-summary.json` | Vector store and output address checks. |
| `phase3/address-confirmation.json` | Shared-stage/address confirmation. |
| `phase3/cublas-observation.json` | cuBLAS dispatch/SASS observations; no kernel extraction. |
| `phase3/cublas-dispatch.json` | Dispatch metadata from the vendor observation. |
| `phase3/cublaslt-results.json` | Queried cuBLASLt algorithm attributes and selected timings. |
| `phase3/lowrank-result.json` | Compile-time rank-7 microbenchmark and numeric measurements. |
| `phase3/lowrank-emulation.json` | Low-rank emulation checks. |
| `phase3/producer-emulation.json` | Producer/consumer candidate emulation checks. |
| `phase4/native-results.json` | CLI promotion checks and six A100 workload timings. |
| `phase4/native-build-evidence.json` | Production build and instruction-stream provenance. |
| `phase4/cache-results.json` | Cache hint correctness and mixed performance results. |
| `phase4/fragment-results.json` | B-fragment window correctness and negative timing result. |
| `phase5/layout-results.json` | Grouped, Morton and Hilbert CTA order runs. The original flattened Morton path is superseded. |
| `phase5/layout-counters.json` | CTA order counters and profiling limitations. |
| `phase5/l2-results.json` | Persisting-L2 policy data. |
| `phase5/l2-refine-results.json` | Refined L2 reservation sweep. |
| `phase5/l2-summary.json` | L2 decision summary. |
| `phase5/fusion-summary.json` | Initial, optimized and confirmation fusion totals. |
| `phase5/fusion-square-results.json` | Square fusion measurements and checks. |
| `phase5/fusion-square-confirmation.json` | Independent square confirmation results. |
| `phase5/fusion-smoke-results.json` | Small-shape fusion results, including regressions. |
| `phase5/fusion-mlp-results.json` | MLP fusion results, including regressions. |
| `phase5/fusion-failed-harness.json` | Failed harness record; not a valid performance result. |
| `phase5/barrier-single-slot.json` | Correct single-slot mbarrier test; slower than CTA barrier. |
| `phase5/barrier-two-slot.json` | Two-slot one-CTA overlap microbenchmark, not dense GEMM. |
| `phase5/barrier-accounting.json` | Arrival-count/parity accounting. |
| `phase5/barrier-layout-check.json` | Barrier/operand layout check and repaired lane-map issue. |
| `phase5/barrier-pipeline-static-check.json` | Static pipeline protocol checks. |
| `phase5/cuda-l2-contracts.json` | Sampled upstream kernel precision/layout contracts. |
| `phase5/cuda-l2-source-manifest.json` | Pinned upstream URLs and SHA-256s; copied upstream source is not retained. |
| `phase6-validation-20261104-superseded.json` | First production validation. Morton decode was subsequently corrected. |
| `current-validation/` | Current validator, six candidate manifests/cubins, matched ptxas/OpenAssembler controls and the 2026-11-05 full-output A100 record. |
| `encoding/` | Exact JSON inputs whose hashes are embedded in the production tensor encoder. |
| `initial-triton-lowrank-results.json` | Early Triton low-rank timings and SASS/resource metadata; precision mismatch is explicit. |
| `initial-triton-lowrank-artifacts-metadata.json` | Metadata for the archived early Triton PTX/SASS corpus. |
| `initial-triton-lowrank-artifacts.tar.gz` | Compressed original PTX/SASS files; SHA-256 is in `manifest.json`. |

Scalar NN baseline outputs are in the single canonical file
[`docs/evidence/nn-study-a100-20261102.json`](../nn-study-a100-20261102.json).
The system integration snapshot is
[`docs/evidence/integration-20261104.json`](../integration-20261104.json).
Corrected scalar instruction latency evidence is indexed separately under
[`docs/evidence/a100-latency`](../a100-latency/README.md).

`manifest.json` lists the SHA-256 and byte size for each retained file, plus
the original experiment path when known. It does not hash itself.
