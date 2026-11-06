# Adaptive shared-memory budgeting

Status: untested-or-hardware-unavailable

Status applies to the stated experiment and contract, not to every possible version of the idea.

## Idea in plain language

Choose how much shared memory to devote to each operand, pipeline stage and epilogue for a particular shape. The goal is enough overlap and reuse without consuming the capacity needed for more active CTAs.

## Findings and implementation decision

The production native GEMM has an explicit fixed contract: two 32 KiB shared operand regions, 64 KiB dynamic shared memory and 128 threads. Launch code opts into that allocation. Historical three/four-stage and larger-tile candidates measured the cost of other fixed designs; they were not an adaptive allocator.

No measured runtime shared-memory budgeting policy or general shape-dependent memory-allocation compiler has been implemented. Dynamic CUDA allocation at launch is available, but changing a byte count without regenerating the stage layout would be incorrect. A100's physical shared-memory banks are not arbitrarily reconfigured by this idea.

## Method and validity checks

A proposed policy must jointly choose CTA/warp tile, stage count, operand/epilogue storage, occupancy and allocation size. First prove that all shared objects are disjoint and in bounds through fill/steady/drain phases. Then check full outputs/canaries, registers/local memory, actual launch resources and paired timing. Include any tuning/startup cost in the application comparison.

## Reproduction and command set

Run from the repository root. Use Python 3.12+ and an installed test environment. Commands that compile require the CUDA 12.8 toolkit; GPU commands additionally require an NVIDIA driver or an authenticated Modal client. Explicit skips mean a dependency/corpus is unavailable, not a passing hardware result.

There is no adaptive-budget CLI switch. Establish the fixed control:

```sh
.venv/bin/python -m openassembler gemm --m 1024 --n 4096 --k 1024 --out-dir output/smem-budget-control
.venv/bin/python -m pytest -q tests/test_gemm_codegen.py tests/test_cli_gemm.py
python3 -m json.tool docs/evidence/a100-mma/phase2/summary.json
```
The first command's manifest reports the selected shared-memory requirement. The historical stage tests are evidence about those fixed candidates, not a tested adaptive policy.

## Code and retained evidence

- [Fixed kernel allocation/launch contract](../../../openassembler/kernels/gemm.py)
- [Driver shared-memory opt-in](../../../openassembler/gpu/cuda_driver.py)
- [Historical stage/resource results](../../evidence/a100-mma/phase2/summary.json)
- [Deeper pipeline outcomes](deeper-pipelines-and-tile-shapes.md)

## What remains open

Build and test a bounded shape/resource search with an explicit dispatch policy. More buffering must earn its register/occupancy/control cost; an adaptive policy cannot assume that the deepest pipeline is fastest.
