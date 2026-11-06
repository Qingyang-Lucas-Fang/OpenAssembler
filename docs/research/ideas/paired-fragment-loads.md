# Paired B-fragment LDSM loads and register-group use

Status: implemented

Status applies to the stated experiment and contract, not to every possible version of the idea.

## Idea in plain language

Load two related B matrix fragments together and let MMA instructions consume the appropriate halves directly. Avoid loading or copying each fragment independently when an aligned group already exists.

## Findings and implementation decision

The selected native kernel uses paired `LDSM.x4` B loads. The selector can pass the implicit second half of a register group to HMMA without a tuple-copy expansion, and represents both halves as actual readers. This is part of the promoted mainloop bundle; the 16–27% historical bundle gain is not attributed entirely to this one change.

The load grouping and the register organization are inseparable for correctness. A scalar-looking first-register operand can implicitly read additional consecutive registers.

## Method and validity checks

Check fragment-lane ownership against the full integer GEMM oracle, inspect register alignment, and preserve address write-after-read protection until LDSM finishes collecting operands. Hardware tests must check all outputs, not only a corner tile.

## Reproduction and command set

Run from the repository root. Use Python 3.12+ and an installed test environment. Commands that compile require the CUDA 12.8 toolkit; GPU commands additionally require an NVIDIA driver or an authenticated Modal client. Explicit skips mean a dependency/corpus is unavailable, not a passing hardware result.

```sh
.venv/bin/python -m pytest -q tests/test_backend_tensor.py tests/test_gemm_codegen.py
.venv/bin/python -m openassembler gemm --m 128 --n 256 --k 384 --dtype bf16 --out-dir output/paired-loads
.venv/bin/python -m openassembler run output/paired-loads --backend modal --budget 120
```
The original isolated load-order search was retired. These commands reproduce the implemented grouped path and its regressions, not a fresh isolated speed attribution.

## Code and retained evidence

- [Fragment load/emission](../../../openassembler/kernels/gemm.py)
- [Tuple/implicit-reader selection](../../../openassembler/backend/tensor.py)
- [Full output tests](../../../tests/test_gemm_codegen.py)
- [Candidate search record](../../evidence/a100-mma/phase3/search-summary.json)

## What remains open

Search load ownership and register organization together with shared layout. Grouped loads do not remove the need to measure register pressure and operand-collection stalls.
