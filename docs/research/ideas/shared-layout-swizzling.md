# Shared-memory swizzling and bank conflicts

Status: implemented

Status applies to the stated experiment and contract, not to every possible version of the idea.

## Idea in plain language

Change the mapping of matrix elements to shared-memory addresses so simultaneously loaded values spread across banks. This changes software layout, not the physical organization of A100's memory banks.

## Findings and implementation decision

The selected GEMM uses the tested symmetric 7/7 XOR swizzle. In the archived comparison, unswizzled shared loads reported 234.9 million conflicts and 2.255 ms, versus about 0.170 million and 0.807 ms for 7/7. Asymmetric 7/3 and 3/7 variants did not beat the selected layout.

Those large differences apply to that profiled schedule/run. Later matched small MLP profiles had zero shared-bank conflicts on both OA and ptxas; further swizzle changes alone therefore do not explain their remaining gap.

## Method and validity checks

Check that producer stores and consumer LDSM lane maps agree, that address maps are in bounds, and that each matrix element reaches its intended MMA lane. Validate all outputs before timing. Collect bank traffic and tensor/stall counters separately from graph medians.

## Reproduction and command set

Run from the repository root. Use Python 3.12+ and an installed test environment. Commands that compile require the CUDA 12.8 toolkit; GPU commands additionally require an NVIDIA driver or an authenticated Modal client. Explicit skips mean a dependency/corpus is unavailable, not a passing hardware result.

```sh
.venv/bin/python -m pytest -q tests/test_gemm_codegen.py tests/test_backend_tensor.py
.venv/bin/python -m openassembler gemm --m 128 --n 128 --k 256 --out-dir output/swizzle
```
The command builds the production swizzle. The old asymmetric/unswizzled sweep generator was retired, so the archived result is not a currently executable complete search. A new sweep must live under `exp/` and pass the same gates.

## Code and retained evidence

- [Swizzled address emission](../../../openassembler/kernels/gemm.py)
- [Historical layout/counter summary](../../evidence/a100-mma/phase2/summary.json)
- [Current source checks](../../../tests/test_gemm_codegen.py)
- [Joint-search proposal](joint-layout-register-search.md)

## What remains open

Search shared addresses jointly with fragment ownership and register lifetimes. Arbitrary hardware bank remapping or runtime asymmetric banks are not compiler capabilities established by these tests.
