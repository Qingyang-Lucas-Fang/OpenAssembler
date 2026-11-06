# Constant integer division and remainder lowering

Status: implemented

Status applies to the stated experiment and contract, not to every possible version of the idea.

## Idea in plain language

Replace division by a known integer with a short sequence of shifts and multiply-high instructions. The constants are chosen so the result is exact over the supported integer domain; this is not approximate floating-point division.

## Findings and implementation decision

The implemented 32-bit lowering removes a reciprocal-based path from direct PTX division/remainder sites. The CUDA 12.8 survey found the tested immediate u32 divisions used about 13 extra instructions versus 2–3 in the replacement; signed sequences also require sign handling. Formal checks and concrete edge/random tests cover the generated formulas.

This opportunity is especially relevant to hand-written PTX or frontends that leave a divisor constant until this stage. NVVM/LLVM often perform equivalent strength reduction before producing PTX, so a general CUDA-C division speedup must not be assumed. Dynamic and 64-bit division are different paths.

## Method and validity checks

Survey 17 divisors across division/remainder and integer widths, inspect the emitted sequences, then prove each supported replacement with bit-vector SMT. Compare zero, boundary and signed edge values. The scalar loop microbenchmark improved from 0.629 to 0.464 ms for division by seven in its historical run; that is a kernel result, not a Tensor Core gain.

## Reproduction and command set

Run from the repository root. Use Python 3.12+ and an installed test environment. Commands that compile require the CUDA 12.8 toolkit; GPU commands additionally require an NVIDIA driver or an authenticated Modal client. Explicit skips mean a dependency/corpus is unavailable, not a passing hardware result.

```sh
.venv/bin/python -m pytest -q tests/test_backend.py
.venv/bin/python -m openassembler compile examples/div_by_const.ptx --verify --out-dir output/divconst
.venv/bin/python -m openassembler run output/divconst --backend modal --budget 120
PYTHONPATH=. .venv/bin/python tools/prove_divconst.py 200
```
The optional proof script needs `z3-solver` and the committed survey JSON. It rewrites `corpus/sm_80/divconst_proofs.json`; run in an experiment checkout if preserving old proof records.

## Code and retained evidence

- [Exact lowering](../../../openassembler/opt/divconst.py)
- [Proof driver](../../../tools/prove_divconst.py)
- [Retained proofs](../../../corpus/sm_80/divconst_proofs.json)
- [ptxas survey](../../../corpus/sm_80/div_const_survey.json)
- [Example PTX](../../../examples/div_by_const.ptx)

## What remains open

Measure sites where constants emerge only after propagation/inlining, and separately profile application impact. Do not extend 32-bit proofs to unproved 64-bit cases.
