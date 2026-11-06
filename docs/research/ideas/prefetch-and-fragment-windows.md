# Copy prefetch hints and bounded fragment windows

Status: tested-not-selected

Status applies to the stated experiment and contract, not to every possible version of the idea.

## Idea in plain language

Start fetching the next fragment earlier or retain a bounded window of fragments in registers. A cache-prefetch hint can also influence data movement, but it is different from allocating more shared-memory stages.

## Findings and implementation decision

The production follow-up tested cp.async L2-prefetch modifiers, B-fragment windows of two/four and bounded prefetch. Candidates passed correctness, but prefetch hints produced small inconsistent effects and fragment windows were slower over all seven paired rounds on the tested workloads. None was promoted.

Keeping values live longer can raise register pressure or delay useful work. The negative result does not mean every next-fragment ping-pong schedule is impossible; it limits the exact tested variants.

## Method and validity checks

Keep arithmetic and output precision fixed, validate every result and canary, report registers/local/shared memory, and compare paired graph medians. Do not substitute an attractive static load count for measured overlap. Separate cache-hint effects from register-window effects.

## Reproduction and command set

Run from the repository root. Use Python 3.12+ and an installed test environment. Commands that compile require the CUDA 12.8 toolkit; GPU commands additionally require an NVIDIA driver or an authenticated Modal client. Explicit skips mean a dependency/corpus is unavailable, not a passing hardware result.

```sh
.venv/bin/python -m pytest -q tests/test_backend_tensor.py tests/test_gemm_codegen.py
python3 -m json.tool docs/evidence/a100-mma/phase4/cache-results.json
python3 -m json.tool docs/evidence/a100-mma/phase4/fragment-results.json
```
The public generator exposes the selected schedule only. Retired prefetch/window drivers are not present; these commands preserve the current control and inspect the actual old comparison.

## Code and retained evidence

- [Cache-hint measurements](../../evidence/a100-mma/phase4/cache-results.json)
- [Fragment-window full result](../../evidence/a100-mma/phase4/fragment-results.json)
- [Current operand staging](../../../openassembler/kernels/gemm.py)
- [Joint-search proposal](joint-layout-register-search.md)

## What remains open

Search lifetime/order and shared layout together, guided by counters. A new candidate should show repeatable benefit at matched precision without losing the established synchronization edges.
