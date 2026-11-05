# Reproducing OpenAssembler

This is the current reproduction guide. Historical A100 research outcomes are
consolidated in [docs/research/A100_MMA_RESULTS.md](docs/research/A100_MMA_RESULTS.md),
with compact raw records in [docs/evidence](docs/evidence). The old phase
workspaces and generated build trees were removed after their findings were
indexed; commands below target the current repository.

## Local compiler and tests

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q
```

The tensor-focused regression suite is:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  tests/test_backend_tensor.py tests/test_gemm_codegen.py tests/test_cli_gemm.py
```

Compilation needs CUDA 12.8 `ptxas` and `nvdisasm`, supplied through
`env/cuda.sh` or an installed Linux toolkit. Compilation does not require a
GPU. `openassembler run` needs an NVIDIA driver; Modal execution needs an
authenticated Modal client.

## Standard GEMM

```sh
openassembler gemm --m 4096 --n 16384 --k 4096 --dtype f16 --out-dir output
openassembler compile output/.build/generated/gemm.ptx --out-dir recompiled
openassembler run output --backend modal --budget 300 --job-timeout 120
```

The contract is full-tile `C=A@B.T`, FP16/BF16 inputs and FP32 accumulation/
output. Read [docs/TENSOR_GEMM.md](docs/TENSOR_GEMM.md) for layouts, alignment,
launch geometry and unsupported forms.

The current opt-in paths are:

```sh
# Current Morton mapping: complete 8x8 CTA groups, M/N multiples of 1024.
openassembler gemm --m 16384 --n 4096 --k 4096 --dtype f16 \
  --cta-order morton8 --out-dir output_morton

# FP32 bias + residual, optionally followed by ReLU.
openassembler gemm --m 4096 --n 4096 --k 4096 --dtype bf16 \
  --epilogue bias_residual_relu --out-dir output_fused
```

Morton and fusion are explicit shape-specific choices. They are not automatic
dispatch policies and do not establish whole-training-step acceleration.

## Current A100 validation

The retained validator and matched controls are under
`docs/evidence/a100-mma/current-validation/`. The current production cubins
were built through the normal CLI. To reproduce the latest full-output,
canary and shuffled-graph run, use the archived validator:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m modal run \
  docs/evidence/a100-mma/current-validation/validate_modal.py
```

The November5 result is
`docs/evidence/a100-mma/current-validation/validation_20261105T074142.327676Z.json`.
It checks zero, integer and normal inputs, all outputs, guard regions, ptxas
equality and same-compiler controls. The client also experienced a later Modal
heartbeat failure; that failed submission produced no result and is not used.

## Reverse-engineering evidence

The corrected 42-case scalar latency sweep is in
[docs/evidence/a100-latency](docs/evidence/a100-latency/README.md). Encoding
and async-copy evidence used by the compiler is in
[docs/evidence/a100-mma/encoding](docs/evidence/a100-mma/encoding). The methods
and limits are documented in
[docs/research/REVERSE_ENGINEERING.md](docs/research/REVERSE_ENGINEERING.md).

## Research record

Use [docs/research/README.md](docs/research/README.md) as the entry point.
[A100_MMA_RESULTS.md](docs/research/A100_MMA_RESULTS.md) records every major
success, regression, failure and superseded result, including counters,
producer warps, multistage schedules, shared layouts, cuBLAS observations,
CUDA-L2, mbarrier, low-rank transforms and fusion. The evidence indexes list
the retained machine-readable files and original experiment paths.
