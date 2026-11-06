# Operand reuse hints and ordering

Status: implemented

Status applies to the stated experiment and contract, not to every possible version of the idea.

## Idea in plain language

Mark an operand that will be used again soon so the instruction machinery may reuse its collected value. Arrange neighboring operations to make those hints useful without changing arithmetic dependencies.

## Findings and implementation decision

The tensor scheduler emits reuse hints for supported neighboring HMMA operands. Earlier output-column order plus reuse work improved a paired sample about 0.3–1.8%; this was small and combined two changes. It is not a separately isolated universal reuse-cache gain.

Reuse hints are encoding controls, not a user-created register-file cache or permission to overwrite a live register. The production path still accounts for all implicit operands and dependency edges.

## Method and validity checks

Inspect exact words/control fields, preserve semantic grouping, compare full outputs and time repeated paired runs. Separate valid encoding from useful hardware behavior; counter evidence has not conclusively isolated a register-bank/operand-collector explanation for every remaining GEMM gap.

## Reproduction and command set

Run from the repository root. Use Python 3.12+ and an installed test environment. Commands that compile require the CUDA 12.8 toolkit; GPU commands additionally require an NVIDIA driver or an authenticated Modal client. Explicit skips mean a dependency/corpus is unavailable, not a passing hardware result.

```sh
.venv/bin/python -m pytest -q tests/test_word.py tests/test_backend_tensor.py
.venv/bin/python -m openassembler gemm --m 1024 --n 1024 --k 1024 --out-dir output/reuse
```
There is no public CLI switch for an arbitrary reuse mask. The original exploratory order/reuse sweep was retired; current code/tests reproduce the selected policy.

## Code and retained evidence

- [Reuse scheduling](../../../openassembler/backend/tensor.py)
- [Control bit representation](../../../openassembler/isa/word.py)
- [Historical pointer/order/reuse evidence](../../evidence/a100-mma/phase2/summary.json)
- [Selected search record](../../evidence/a100-mma/phase3/search-summary.json)

## What remains open

A safe automatic search should vary order, fragment ownership and reuse together and retain all established dependencies. Do not interpret fewer printed bank conflicts as proof of every collector mechanism.
