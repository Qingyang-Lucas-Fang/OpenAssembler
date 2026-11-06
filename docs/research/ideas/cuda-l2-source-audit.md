# CUDA-L2 project source audit

Status: untested-or-hardware-unavailable

Status applies to the stated experiment and contract, not to every possible version of the idea.

## Idea in plain language

Study publicly released A100 kernels for useful scheduling patterns: alternating fragment registers, early copies, shape-aware tile selection and grouped CTA traversal. Separate source inspiration from a reproduced OpenAssembler speedup.

## Findings and implementation decision

Five kernels were sampled at commit dbe017722194bb33bafadfbcbb4a65ab6df95dc3. Four CuTe samples accumulate FP32 but store FP16; a small WMMA example accumulates FP16. B packing/transposition occurs outside the published kernel timing. The square 160-row tile pads M from 4096 to 4160, adding 1.5625% work.

Those contracts are not matched to OA's contiguous RCR, FP32-output path. The audit did not reproduce the project's training/reward system, inspect every file or copy a kernel. Transferable patterns remain proposals unless a separate OA experiment validates them.

## Method and validity checks

Pin source URLs and SHA-256, inspect complete input/output precision, padding, preprocessing, tile shape and stage layout, then compare only matching arithmetic contracts. Treat a source description as lower evidence than an executable result.

## Reproduction and command set

Run from the repository root. Use Python 3.12+ and an installed test environment. Commands that compile require the CUDA 12.8 toolkit; GPU commands additionally require an NVIDIA driver or an authenticated Modal client. Explicit skips mean a dependency/corpus is unavailable, not a passing hardware result.

```sh
python3 -m json.tool docs/evidence/a100-mma/phase5/cuda-l2-source-manifest.json
python3 -m json.tool docs/evidence/a100-mma/phase5/cuda-l2-contracts.json
```
These inspect the pinned audit. No upstream source copy or local RL trainer is retained. Re-fetching upstream files must use the pinned commit and verify recorded hashes; this is not a command to reproduce the project's claimed speedups.

## Code and retained evidence

- [Pinned source identities](../../evidence/a100-mma/phase5/cuda-l2-source-manifest.json)
- [Audited numerical/layout contracts](../../evidence/a100-mma/phase5/cuda-l2-contracts.json)
- [Separate L2 API experiment](l2-persistence.md)
- [Joint layout search proposal](joint-layout-register-search.md)

## What remains open

Implement and test promising overlap patterns under matched FP32-output semantics. Public project names and source examples are not evidence that OA already contains their optimizer or training framework.
