# A100 feasibility: TMA, clusters, FP8, DPX and virtual grouping

Status: untested-or-hardware-unavailable

Status applies to the stated experiment and contract, not to every possible version of the idea.

## Idea in plain language

Separate software scheduling ideas from physical mechanisms that this GPU does not contain. A compiler may rearrange A100 work, but cannot enable a later architecture's data-movement engine or arithmetic datapath with a flag.

## Findings and implementation decision

A100 does not provide Hopper TMA, hardware thread-block clusters, DPX, or the H100 FP8 Tensor Core path. None is implemented as an A100 compiler option. Software CTA grouping can improve locality but cannot guarantee adjacent SM placement, cluster synchronization or distributed shared-memory semantics.

Ordinary cp.async, CTA barriers, documented SM80 mbarrier forms, FP16/BF16 Tensor Cores and structured 2:4 sparsity are different, genuinely available mechanisms. The register-file-cache suggestion is not a discovered user-configurable RF cache; register lifetimes and operand reuse are explicit compiler choices.

## Method and validity checks

Audit the target architecture and supported PTX/assembler forms first, then distinguish a ptxas capability probe from independent OA selection and full hardware validation. Simulating later hardware on A100 may change semantics/cost and must be described as a software algorithm, not hardware enablement.

## Reproduction and command set

Run from the repository root. Use Python 3.12+ and an installed test environment. Commands that compile require the CUDA 12.8 toolkit; GPU commands additionally require an NVIDIA driver or an authenticated Modal client. Explicit skips mean a dependency/corpus is unavailable, not a passing hardware result.

```sh
.venv/bin/python -m openassembler status
.venv/bin/python -m openassembler gemm --help
python3 -m json.tool docs/evidence/a100-mma/phase2/literature-discovery.json
```
These inspect current support and the retained capability study. There are no honest A100 commands for enabling TMA, hardware clusters, DPX or H100 FP8. Unsupported architecture/mode requests should remain explicit errors.

## Code and retained evidence

- [Capability/literature discovery record](../../evidence/a100-mma/phase2/literature-discovery.json)
- [Dense SM80 supported contract](../../../openassembler/kernels/gemm.py)
- [Sparse capability dossier](sparse-mma-capability.md)
- [Software producer/consumer result](producer-consumer-warps.md)
- [Morton software locality](morton-cta-order.md)

## What remains open

Target other architectures with their own instructions, encoders and hardware validation, or evaluate a precisely specified A100 software alternative. Do not claim a nonexistent hardware feature was unlocked.
