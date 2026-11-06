# Alternative multi-precision accumulation paths

Status: untested-or-hardware-unavailable

Status applies to the stated experiment and contract, not to every possible version of the idea.

## Idea in plain language

Consider whether parts of an accumulation can use a different precision, or whether correction/reconstruction can preserve accuracy while reducing work. Every such path changes the numerical contract and must be evaluated against the application's required outputs and training state.

## Findings and implementation decision

The implemented tensor GEMM already uses mixed precision in the ordinary sense: FP16/BF16 inputs with FP32 accumulation and FP32 output. The emulator has separately validated recorded finite TF32/integer/FP64 tensor cases; that does not mean the generated GEMM supports those modes.

No adaptive accumulation-width policy, compensated multi-path Tensor Core algorithm or lower-precision accumulation optimization has been measured and promoted for OpenAssembler training. Low-rank transform rounding and half-output prototypes are not algorithm-only evidence for a new accumulation path. A100's lack of the H100 FP8 Tensor Core path cannot be bypassed by changing a dtype flag.

## Method and validity checks

A proposed experiment must specify every input cast, multiplication format, accumulator update, correction term and output rounding. Compare against a matched direct control and a higher-precision reference on normal, cancellation, overflow/subnormal and adversarial inputs. Then apply unchanged multi-step parameter/gradient/optimizer gates; small output error alone is insufficient.

## Reproduction and command set

Run from the repository root. Use Python 3.12+ and an installed test environment. Commands that compile require the CUDA 12.8 toolkit; GPU commands additionally require an NVIDIA driver or an authenticated Modal client. Explicit skips mean a dependency/corpus is unavailable, not a passing hardware result.

No alternative-accumulation compiler flag exists. Current controls are:

```sh
.venv/bin/python -m openassembler gemm --m 128 --n 128 --k 128 --dtype f16 --out-dir output/precision-f16
.venv/bin/python -m openassembler gemm --m 128 --n 128 --k 128 --dtype bf16 --out-dir output/precision-bf16
.venv/bin/python -m pytest -q tests/test_gpu_truth.py tests/test_backend_tensor.py
```
These validate existing supported modes and recorded emulator cases, not the proposed multi-path algorithm. A new experiment requires its own explicit precision/error contract.

## Code and retained evidence

- [Current generated precision contract](../../../openassembler/kernels/gemm.py)
- [Fitted tensor arithmetic model](../../../openassembler/emu/tensor.py)
- [Recorded tensor cases](../../../corpus/sm_80/gpu_a100_40gb_tensor.raw.json.gz)
- [Low-rank precision findings](compile-time-low-rank-mma.md)
- [Hardware availability](a100-hardware-limits.md)

## What remains open

Develop a concrete supported-A100 algorithm and measurable error bound, then test complete training behavior. Do not trade accuracy or change optimizer policy silently to admit a faster result.
