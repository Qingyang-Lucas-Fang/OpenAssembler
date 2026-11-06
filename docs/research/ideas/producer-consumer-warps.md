# Software producer/consumer warp specialization

Status: tested-not-selected

Status applies to the stated experiment and contract, not to every possible version of the idea.

## Idea in plain language

Let some warps load future operand tiles while others perform MMA, using ordinary A100 synchronization. The intention is better overlap, but dedicated producer warps also occupy threads, registers and shared storage.

## Findings and implementation decision

The tested role-split variants passed four large-case A100 correctness checks but measured about 1.019–1.943 ms and lost to the unified-warp schedule. A 544-thread candidate exceeded launch resources. No producer/consumer dense-GEMM plan was promoted.

This rejects those designs, not software warp specialization in general. The separate two-slot mbarrier microbenchmark is positive only for its tiny one-CTA overlap workload; it must not be counted as a dense GEMM win.

## Method and validity checks

Specify which warps arrive, produce, consume and advance each phase. Check count/parity, shared-object separation, resource reports, full outputs and canaries. Never rely on unscheduled CTAs or a nonexistent A100 hardware cluster to make progress.

## Reproduction and command set

Run from the repository root. Use Python 3.12+ and an installed test environment. Commands that compile require the CUDA 12.8 toolkit; GPU commands additionally require an NVIDIA driver or an authenticated Modal client. Explicit skips mean a dependency/corpus is unavailable, not a passing hardware result.

```sh
.venv/bin/python -m pytest -q tests/test_backend_tensor.py tests/test_gemm_codegen.py
.venv/bin/python -m openassembler gemm --m 4096 --n 4096 --k 4096 --out-dir output/unified-warp-control
```
No public producer-warp option exists. The old role-split generator was retired; its measured outcome and emulation records remain below. A fresh design needs a new experimental driver.

## Code and retained evidence

- [Role-split checks and timing](../../evidence/a100-mma/phase2/summary.json)
- [Producer emulation record](../../evidence/a100-mma/phase3/producer-emulation.json)
- [mbarrier overlap dossier](mbarrier-overlap.md)

## What remains open

Try role allocation jointly with stage depth and register liveness only after a complete protocol/resource model exists. Do not repeatedly time a launch-invalid candidate as a slow kernel.
