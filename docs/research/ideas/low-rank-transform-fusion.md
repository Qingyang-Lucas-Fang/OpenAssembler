# Fuse low-rank transforms into operand staging

Status: untested-or-hardware-unavailable

Status applies to the stated experiment and contract, not to every possible version of the idea.

## Idea in plain language

Build transformed A/B tiles while staging operands, reuse common transforms, and consume each product soon enough that many full products need not stay live. Search fusion boundaries and where temporary results live, rather than minimizing multiplication rank alone.

## Findings and implementation decision

The existing runtime-factor and tiny compile-time studies identify this as promising future work. SubCuber's graph/fusion and temporary-placement ideas are relevant prior art, but the repository has not reproduced a competitive native OA implementation of that approach.

The older FP32 fastmm API already has fused research plans, but those did not win: fewer launches could still cost more registers, repeated transforms or slower main kernels. That implementation is not the proposed staging-fused tensor design.

## Method and validity checks

The proposed cost model must include additions, transform rounding, registers, shared storage, launches and recomputation, plus the direct mainloop's throughput. Verify exact factor identities separately from floating outputs and train-step state. Match input/accumulation/output precision and saved activations before timing.

## Reproduction and command set

Run from the repository root. Use Python 3.12+ and an installed test environment. Commands that compile require the CUDA 12.8 toolkit; GPU commands additionally require an NVIDIA driver or an authenticated Modal client. Explicit skips mean a dependency/corpus is unavailable, not a passing hardware result.

There is no native transform-staging compiler command. Existing prerequisites are reproducible:

```sh
.venv/bin/python -m pytest -q tests/test_fastmm.py tests/test_backend_tensor.py
.venv/bin/python -m openassembler gemm --m 4096 --n 4096 --k 4096 --out-dir output/lowrank-direct-control
```
A new compiler/experiment driver must be written under `exp/`; invoking a direct GEMM is only a control, not evidence that transform fusion is implemented.

## Code and retained evidence

- [Current factor/plan infrastructure](../../../openassembler/fastmm/plan.py)
- [Historical low-rank control](../../evidence/a100-mma/phase3/lowrank-result.json)
- [Older fused FP32 plans/results](../../../corpus/sm_80/fastmm_a100_40gb_n8192_sgemm128.json)
- [Research references](../REFERENCES.md)

## What remains open

First make the direct mainloop competitive, then search bounded graph partitions with full numerical and resource gates. A lower rank is not a hardware cost model.
