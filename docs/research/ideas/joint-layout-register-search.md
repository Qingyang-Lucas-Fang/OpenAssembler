# Joint shared-layout and register-fragment search

Status: untested-or-hardware-unavailable

Status applies to the stated experiment and contract, not to every possible version of the idea.

## Idea in plain language

Choose where values live in shared memory and which registers/lanes own them at the same time. A good shared swizzle may still pair badly with a particular load order or long-lived fragment organization.

## Findings and implementation decision

Existing swizzle experiments largely held register organization fixed; fragment-window and tile experiments explored other dimensions separately. No general joint optimizer has yet produced a documented new production schedule. This remains an open research direction, not a measured acceleration.

The current winning layout and grouped-load components provide a correct control. Negative standalone window/stage results do not disprove combinations that have not been tested.

## Method and validity checks

Proposed search variables are swizzle, lane ownership, LDSM order, B-fragment pairing, reuse hints and live ranges. Each candidate must pass semantic/address checks, disassembly, all output elements, canaries and unchanged precision gates before paired timing. Keep known copy, LDSM address-WAR and HMMA edges fixed.

## Reproduction and command set

Run from the repository root. Use Python 3.12+ and an installed test environment. Commands that compile require the CUDA 12.8 toolkit; GPU commands additionally require an NVIDIA driver or an authenticated Modal client. Explicit skips mean a dependency/corpus is unavailable, not a passing hardware result.

There is no implemented joint-search command. Establish the current control with:

```sh
.venv/bin/python -m pytest -q tests/test_gemm_codegen.py tests/test_backend_tensor.py
.venv/bin/python -m openassembler gemm --m 4096 --n 4096 --k 4096 --out-dir output/joint-search-control
```
A new search driver and its actual command set must be added under `exp/` before claiming the proposed method has been reproduced.

## Code and retained evidence

- [Current generator](../../../openassembler/kernels/gemm.py)
- [Current dependency policy](../../../openassembler/backend/tensor.py)
- [Retained search dimensions/results](../../evidence/a100-mma/phase3/search-summary.json)
- [Rejected fragment windows](../../evidence/a100-mma/phase4/fragment-results.json)

## What remains open

Develop a bounded candidate representation and realistic cost model. Shared conflicts, tensor utilization, occupancy, source collection and launch/grid size must be measured together.
