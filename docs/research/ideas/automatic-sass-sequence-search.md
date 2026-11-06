# Automatic equivalent SASS sequence search

Status: untested-or-hardware-unavailable

Status applies to the stated experiment and contract, not to every possible version of the idea.

## Idea in plain language

Automatically generate different native instruction sequences for the same work: address arithmetic, fragment ordering, reuse hints and controls. Keep only sequences with the same semantics, then measure performance.

## Findings and implementation decision

Historical candidate searches explored several epilogue orders, fragment windows, pre-addressing, operand orders and scheduler variants; selected changes contributed to the current GEMM. However, a general reusable search system with proved equivalence and full hardware admission is not integrated into the compiler.

The proposal is not to relax known dependencies until a listing looks shorter. Removing address-WAR or weakening copy completion already caused failures. Compile-time/backend policy is explicit and immutable for the current validated tensor path.

## Method and validity checks

Proposed gates are exact/structural equivalence, valid disassembly, complete reference outputs, canaries/resource checks, then randomized paired timing. Retain every attempted candidate and avoid choosing a winner from one noisy sweep. Separate host search/compile time from device runtime.

## Reproduction and command set

Run from the repository root. Use Python 3.12+ and an installed test environment. Commands that compile require the CUDA 12.8 toolkit; GPU commands additionally require an NVIDIA driver or an authenticated Modal client. Explicit skips mean a dependency/corpus is unavailable, not a passing hardware result.

```sh
.venv/bin/python -m pytest -q tests/test_backend_tensor.py tests/test_gemm_codegen.py
python3 -m json.tool docs/evidence/a100-mma/phase3/search-summary.json
```
These reproduce the current control and inspect historical search outcomes. There is no public `openassembler autotune-sass` command or complete retained general search driver.

## Code and retained evidence

- [Search outcome record](../../evidence/a100-mma/phase3/search-summary.json)
- [Current native policy](../../../openassembler/backend/tensor.py)
- [Instruction/codegen regressions](../../../tests/test_backend_tensor.py)
- [Joint layout search](joint-layout-register-search.md)

## What remains open

Develop a bounded candidate grammar, explicit semantic checker and an evidence-aware objective that includes end-to-end cost. Host-side search optimization should follow a measured compile profile, not a language rewrite assumption.
