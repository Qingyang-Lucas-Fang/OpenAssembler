# Recovered arithmetic instruction sequences and rounding policy

Status: implemented

Status applies to the stated experiment and contract, not to every possible version of the idea.

## Idea in plain language

Represent a supported PTX operation with its actual native instruction sequence, including conversions, predicates and exceptional-case helpers when necessary. Then allocate and schedule the sequence independently of ptxas.

## Findings and implementation decision

The standard backend imports retained CUDA 12.8 arithmetic fragments and generates its own words. This covers the supported floating/integer conversion, division and approximation paths, alongside direct integer/bitwise selection. Unsupported types/modifiers fail explicitly.

Equivalent real-number formulas need not be floating-point equivalent: separate multiply/add and fused FMA can differ by ULPs. The early neural-kernel study required exact agreement and recorded GELU/normalization/Adam mismatches rather than loosening the gate afterward. Approximate instructions have their own stated contracts.

## Method and validity checks

Generate a small PTX probe per operation, prevent folding with lane-varying inputs, inspect the native sequence, then compare PTX interpretation, SASS emulation and hardware. Keep exceptional paths and selected rounding/FTZ modifiers in the contract.

## Reproduction and command set

Run from the repository root. Use Python 3.12+ and an installed test environment. Commands that compile require the CUDA 12.8 toolkit; GPU commands additionally require an NVIDIA driver or an authenticated Modal client. Explicit skips mean a dependency/corpus is unavailable, not a passing hardware result.

```sh
.venv/bin/python -m pytest -q tests/test_backend_ops.py tests/test_pyfront_semantics.py tests/test_verify.py
.venv/bin/python -m openassembler compile examples/nn_ops.py --out-dir output/arithmetic-ops
```
A successful build does not establish all special-value semantics. For real neural inputs use separately declared numerical gates; do not reinterpret a failed exact comparison as an exact success.

## Code and retained evidence

- [Fragment lowering](../../../openassembler/backend/fragments.py)
- [Retained arithmetic fragments](../../../openassembler/backend/ptxas_fragments.json)
- [Arithmetic instruction tests](../../../tests/test_backend_ops.py)
- [Early exact neural study](../../evidence/nn-study-a100-20261102.json)

## What remains open

Broader precision/contraction policies need explicit user-visible semantics and matched tests. Preserve NaN/subnormal/overflow behavior only where it has actually been implemented and validated.
