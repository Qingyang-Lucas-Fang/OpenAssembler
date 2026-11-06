# Asymmetric shared-memory address layouts

Status: tested-not-selected

Status applies to the stated experiment and contract, not to every possible version of the idea.

## Idea in plain language

Map the two matrix operands differently across shared-memory addresses. An asymmetric software swizzle might suit different operand loads, but it does not change A100's physical bank hardware.

## Findings and implementation decision

The historical shared-layout sweep tested asymmetric 7/3 and 3/7 swizzles against symmetric 7/7. Those tested asymmetric layouts did not beat the selected symmetric layout; no asymmetric option was promoted. The unswizzled design had much higher conflict traffic, showing why the address map matters.

This is a negative result for particular software mappings while largely retaining register organization. It neither disproves all asymmetric layouts nor establishes a compiler-controllable physical bank partition. A runtime flag cannot turn A100's bank circuitry into a different arrangement.

## Method and validity checks

Keep producer stores, LDSM lane ownership and consumer register fragments consistent. Verify a complete address map and all numerical outputs before collecting bank-conflict/throughput counters. Compare the same precision and mainloop with shuffled paired timings. Search physical-register ownership jointly rather than treating swizzle parameters alone as the whole experiment.

## Reproduction and command set

Run from the repository root. Use Python 3.12+ and an installed test environment. Commands that compile require the CUDA 12.8 toolkit; GPU commands additionally require an NVIDIA driver or an authenticated Modal client. Explicit skips mean a dependency/corpus is unavailable, not a passing hardware result.

```sh
python3 -m json.tool docs/evidence/a100-mma/phase2/summary.json
.venv/bin/python -m pytest -q tests/test_gemm_codegen.py tests/test_backend_tensor.py
.venv/bin/python -m openassembler gemm --m 128 --n 128 --k 256 --out-dir output/asymmetric-layout-control
```
The command builds the current symmetric control. The old asymmetric generator was retired and no `--asymmetric-banks` option exists. New layouts need a new experimental implementation rather than changing only the allocation size.

## Code and retained evidence

- [Swizzle sweep/counters](../../evidence/a100-mma/phase2/summary.json)
- [Current address mapping](../../../openassembler/kernels/gemm.py)
- [Selected swizzle dossier](shared-layout-swizzling.md)
- [Joint search proposal](joint-layout-register-search.md)

## What remains open

Test asymmetric lane/register ownership with the shared address map under the same established dependencies. Treat physical bank reconfiguration as unsupported unless a real target mechanism and independent evidence are identified.
