# Compile-time rank-seven versus rank-eight native MMA

Status: tested-not-selected

Status applies to the stated experiment and contract, not to every possible version of the idea.

## Idea in plain language

Bake known low-rank coefficients into code and compare seven transformed block products with the ordinary eight-product construction. This avoids the runtime coefficient overhead of the earlier Triton prototype.

## Findings and implementation decision

A native 96×96×32 microbenchmark on A10080GB passed complete integer, floating and cancellation checks with canaries for its explicitly matched transformed-operand oracle. Rank seven used more registers/instructions and ran near the launch floor. It did not establish a competitive large GEMM or training speedup.

Transform precision is part of correctness: the floating transformed oracle differs from the untransformed direct product by about 0.00556 maximum absolute error, and a cancellation case by about 0.02214. Passing the transformed policy therefore does not mean bitwise or arbitrarily tight agreement with untransformed GEMM.

## Method and validity checks

Compare compile-time rank-seven and rank-eight products at equal input/transform/accumulation/output policy. Check every element against both the matched transformation oracle and the untransformed direct control, report resource counts, and separate timing resolution/launch cost from useful compute.

## Reproduction and command set

Run from the repository root. Use Python 3.12+ and an installed test environment. Commands that compile require the CUDA 12.8 toolkit; GPU commands additionally require an NVIDIA driver or an authenticated Modal client. Explicit skips mean a dependency/corpus is unavailable, not a passing hardware result.

```sh
python3 -m json.tool docs/evidence/a100-mma/phase3/lowrank-result.json
python3 -m json.tool docs/evidence/a100-mma/phase3/lowrank-emulation.json
.venv/bin/python -m pytest -q tests/test_fastmm.py
```
The original native rank-seven PTX emitter was retired. `test_fastmm` checks available factor/plan infrastructure, not a reconstruction of this tensor microbenchmark. No production `gemm --alphatensor` flag exists.

## Code and retained evidence

- [Full numerical/microtiming result](../../evidence/a100-mma/phase3/lowrank-result.json)
- [Native low-rank emulation](../../evidence/a100-mma/phase3/lowrank-emulation.json)
- [Exact factor infrastructure](../../../openassembler/fastmm/factors.py)
- [Transform-fusion proposal](low-rank-transform-fusion.md)

## What remains open

A meaningful large-shape comparison needs bounded live products, common transformed tiles, matched FP32 output and an equally optimized mainloop. The microbenchmark cannot be extrapolated to large throughput.
