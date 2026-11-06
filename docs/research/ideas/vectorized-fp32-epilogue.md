# Vectorized FP32 result stores

Status: implemented

Status applies to the stated experiment and contract, not to every possible version of the idea.

## Idea in plain language

Store adjacent output values as an aligned 64-bit pair and reuse each output-row address, instead of recalculating an address and issuing a scalar store for each float.

## Findings and implementation decision

The native GEMM epilogue groups FP32 stores into pairs and passes an existing aligned register pair directly when possible. C must be eight-byte aligned. It preserves FP32 output instead of narrowing to FP16 to reduce traffic. Full-coverage tests check every element in the final CTA once.

This is distinct from adding bias/residual/ReLU arithmetic. Vectorization is in the default mainloop bundle; its isolated contribution is not the entire measured bundle improvement.

## Method and validity checks

Verify register pairing, store alignment, row stride and complete non-overlapping output coverage. Test the final tile and poisoned destinations. Retain all implicit pair readers in liveness and scheduling.

## Reproduction and command set

Run from the repository root. Use Python 3.12+ and an installed test environment. Commands that compile require the CUDA 12.8 toolkit; GPU commands additionally require an NVIDIA driver or an authenticated Modal client. Explicit skips mean a dependency/corpus is unavailable, not a passing hardware result.

```sh
.venv/bin/python -m pytest -q tests/test_gemm_codegen.py tests/test_backend_tensor.py
.venv/bin/python -m openassembler gemm --m 128 --n 256 --k 128 --out-dir output/vector-epilogue
.venv/bin/python -m openassembler run output/vector-epilogue --backend modal --budget 120
```

## Code and retained evidence

- [Output store generator](../../../openassembler/kernels/gemm.py)
- [Native pair-store selector](../../../openassembler/backend/tensor.py)
- [Store/address evidence](../../evidence/a100-mma/phase3/epilogue-summary.json)
- [Full output coverage tests](../../../tests/test_gemm_codegen.py)

## What remains open

Wider vector stores or shared-memory rearrangement need their own alignment/resource and complete-output tests. A half-output vendor kernel is not a matched FP32-output control.
