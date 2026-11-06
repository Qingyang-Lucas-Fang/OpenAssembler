# Runtime-factor Triton low-rank lowering

Status: tested-not-selected

Status applies to the stated experiment and contract, not to every possible version of the idea.

## Idea in plain language

Fuse rank-seven block products and their linear transforms into one Triton kernel. Fewer mathematical products could save work, but generic runtime factors and many live intermediates can overwhelm the savings.

## Findings and implementation decision

The early A10040GB prototype was far slower: at 4096³ FP16, cuBLAS was about 0.511 ms, direct Triton 1.325 ms, Strassen 11.611 ms and AlphaTensor 13.041 ms. Across the six tested large shape/dtype cases, rank-seven candidates were about 21–42× slower than cuBLAS and 8–15× slower than direct Triton.

The archived PTX explicitly narrows FP32 accumulators to FP16/BF16 output and rounds transformed operands to low precision. It is not matched to today's native FP32-output contract. SASS shows roughly 205/209 registers, thousands of instructions, many barriers/scalar loads and slow-path calls in low-rank variants. Counts were corrected after an earlier pass included unrelated code. This rejects this lowering, not low-rank multiplication itself.

## Method and validity checks

Keep original dtype/output precision and transformations explicit, inspect the actual PTX stores, recover resource counts from disassembly when wrapper attributes are missing, validate numerical error and capture matched timing rounds. Do not call a compiler-lowering comparison algorithm-only.

## Reproduction and command set

Run from the repository root. Use Python 3.12+ and an installed test environment. Commands that compile require the CUDA 12.8 toolkit; GPU commands additionally require an NVIDIA driver or an authenticated Modal client. Explicit skips mean a dependency/corpus is unavailable, not a passing hardware result.

```sh
python3 -m json.tool docs/evidence/a100-mma/initial-triton-lowrank-results.json
python3 -m tarfile -l docs/evidence/a100-mma/initial-triton-lowrank-artifacts.tar.gz
```
The original Triton driver was retired, so these inspect retained results/PTX/SASS rather than rerun compilation. The archive enables auditing output conversions and instructions. A new runner must match the archived environment and contract before remeasurement.

## Code and retained evidence

- [Original device/timing/error record](../../evidence/a100-mma/initial-triton-lowrank-results.json)
- [Original artifact metadata](../../evidence/a100-mma/initial-triton-lowrank-artifacts-metadata.json)
- [PTX/SASS archive](../../evidence/a100-mma/initial-triton-lowrank-artifacts.tar.gz)
- [Compile-time native follow-up](compile-time-low-rank-mma.md)

## What remains open

Remove runtime factor machinery, consume products promptly and reuse transformed tiles. Test matched FP32 output and distinguish transformation rounding from direct multiplication error.
