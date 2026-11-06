# Four-way counted-loop unrolling

Status: implemented

Status applies to the stated experiment and contract, not to every possible version of the idea.

## Idea in plain language

Execute several loop iterations between counter updates and branches. This saves loop-control work and exposes independent instructions, at the cost of more live registers and a longer generated body.

## Findings and implementation decision

The `opt` scalar path applies four-way unrolling to the supported counted-loop form. It can move an exit test ahead of a body when the body does not use the loop counter. The measured scalar suite benefited on division/remainder and independent-work loops, but dependent arithmetic chains still trailed ptxas because OA paid a predicate wait at the boundary.

Historical `safe` versus `opt` timing includes scheduling and unrolling together; it is not an isolated four-way-unroll effect. Tensor kernels use their separate validated policy and are not passed through this scalar unroller.

## Method and validity checks

Check trip-count boundaries and remainder handling, then compare PTX and poisoned SASS outputs. Measure complete loops with matched iteration count, launches and arithmetic. Record resource growth alongside time.

## Reproduction and command set

Run from the repository root. Use Python 3.12+ and an installed test environment. Commands that compile require the CUDA 12.8 toolkit; GPU commands additionally require an NVIDIA driver or an authenticated Modal client. Explicit skips mean a dependency/corpus is unavailable, not a passing hardware result.

```sh
.venv/bin/python -m pytest -q tests/test_backend_loops.py tests/test_fastmm.py
.venv/bin/python -m openassembler compile tests/data/backend_loops/dot_rows.ptx --opt-level safe --out-dir output/unroll-safe
.venv/bin/python -m openassembler compile tests/data/backend_loops/dot_rows.ptx --opt-level opt --out-dir output/unroll-opt
```
These commands compare generated policies; GPU timing needs a correctly specified workload, not the generic dimensions guessed for a pointer-loop kernel.

## Code and retained evidence

- [Unrolling pass](../../../openassembler/backend/unroll.py)
- [Loop regression fixtures](../../../tests/test_backend_loops.py)
- [Historical scalar timings](../../../corpus/sm_80/gpu_a100_40gb_bench_times.json)

## What remains open

Choose unroll factors using measured register pressure and loop structure. Broader control-flow transformations and ptxas-like loop/chain overlap are not established by the current pass.
