# Software fragment caching in registers

Status: tested-not-selected

Status applies to the stated experiment and contract, not to every possible version of the idea.

## Idea in plain language

Keep useful operand fragments in registers for reuse instead of reloading them, while carefully limiting how long they stay live. This is a software storage decision, not a newly enabled physical register-file cache.

## Findings and implementation decision

Bounded B-fragment windows of two/four and prefetch variants passed correctness but were slower across seven paired rounds on the tested workloads. The selected kernel retains its current fragment lifetimes. More live fragments can increase resource pressure or interfere with the intended staging/compute overlap.

The compiler already uses explicit registers and supported operand reuse hints. Neither is a user-configurable general RF cache. These negative window tests reject the measured implementations, not every possible software caching or fragment-reuse strategy.

## Method and validity checks

Track every implicit fragment reader and last use, inspect register/local-memory resources, and verify that delayed overwrites preserve MMA/LDSM dependencies. Validate full outputs/canaries before timing. Compare fixed window sizes and load orders under matched precision and include occupancy/counter changes in the interpretation.

## Reproduction and command set

Run from the repository root. Use Python 3.12+ and an installed test environment. Commands that compile require the CUDA 12.8 toolkit; GPU commands additionally require an NVIDIA driver or an authenticated Modal client. Explicit skips mean a dependency/corpus is unavailable, not a passing hardware result.

```sh
python3 -m json.tool docs/evidence/a100-mma/phase4/fragment-results.json
.venv/bin/python -m pytest -q tests/test_backend_tensor.py tests/test_gemm_codegen.py
.venv/bin/python -m openassembler gemm --m 4096 --n 4096 --k 4096 --out-dir output/register-cache-control
```
This reproduces the selected control and reads the rejected window sweep. There is no `--register-cache` switch or retained complete old window generator. A new caching policy must first be implemented under `exp/`.

## Code and retained evidence

- [Fragment-window checks/timing](../../evidence/a100-mma/phase4/fragment-results.json)
- [Register/fragment selector](../../../openassembler/backend/tensor.py)
- [Existing reuse hints](operand-reuse-hints.md)
- [Joint lifetime/layout search](joint-layout-register-search.md)

## What remains open

Search reuse distances, fragment ownership and producer/consumer timing together. Do not assume higher register occupancy, fewer loads or an attractive static instruction count implies a faster kernel.
