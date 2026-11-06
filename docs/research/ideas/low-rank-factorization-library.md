# Exact AlphaTensor/Strassen factors and FP32 research plans

Status: implemented

Status applies to the stated experiment and contract, not to every possible version of the idea.

## Idea in plain language

Express a block matrix product as linear combinations of smaller products. AlphaTensor and Strassen reduce the number of products, but introduce operand transforms, reconstruction additions and temporary storage.

## Findings and implementation decision

`openassembler.fastmm` contains exact rational factorization verification, imported AlphaTensor schemes, Strassen/naive constructions, launch plans and scalar FP32 kernel generators. Ninety-seven published/imported factorization schemes were historically verified exactly. This implemented research API is separate from the optimized FP16/BF16 Tensor Core `gemm` CLI.

Historical FP32 n=8192 plans did not beat matched cuBLAS overall. The sgemm128 record contains medians alongside budget-exhaustion errors, so it is not a clean complete paired confirmation. Earlier uninterrupted records still show sampled numerical checks rather than full-output validation. The current runner checks 16 rows × about 128 columns against FP64; its old top-level docstring's 64-row description is stale.

## Method and validity checks

First verify the bilinear identity exactly over rational numbers. Then validate a generated launch plan independently on tiny exact matrices, compile kernels with emulation, and compare numerical output plus complete launch/temporary/conversion costs. Mathematical rank validation does not prove floating precision or speed.

## Reproduction and command set

Run from the repository root. Use Python 3.12+ and an installed test environment. Commands that compile require the CUDA 12.8 toolkit; GPU commands additionally require an NVIDIA driver or an authenticated Modal client. Explicit skips mean a dependency/corpus is unavailable, not a passing hardware result.

```sh
.venv/bin/python -m pytest -q tests/test_fastmm.py
PYTHONPATH=. .venv/bin/python tools/fastmm_build.py --arch sm_80 --variant opt
PYTHONPATH=. modal run tools/fastmm_run.py --n 2048 --schemes strassen_2x2x2,at_2x2x2 --gemm sgemm128 --reps 3 --rounds 2 --budget 400
```
The full build can be expensive because it includes the fused plans' matching fgemm kernels. If using `--only` for a smaller build, do not request plans whose kernels were omitted. Recorded historical sample counts/precision must not be rewritten to today's stronger native-GEMM gates.

## Code and retained evidence

- [Factors and exact verifier](../../../openassembler/fastmm/factors.py)
- [Launch-plan construction](../../../openassembler/fastmm/plan.py)
- [Research kernels](../../../openassembler/fastmm/kernels.py)
- [Current research runner](../../../tools/fastmm_run.py)
- [Earlier uninterrupted FP32 record](../../../corpus/sm_80/fastmm_a100_40gb_n8192.json)
- [Interrupted sgemm128 record](../../../corpus/sm_80/fastmm_a100_40gb_n8192_sgemm128.json)

## What remains open

Use compile-time factors, matched transform/input/output precision, bounded products and a competitive tensor mainloop before claiming an algorithm-only acceleration. Re-run full outputs and complete budgets instead of relying on the interrupted historical table.
