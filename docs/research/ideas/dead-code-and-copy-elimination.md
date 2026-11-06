# Dead-code removal, copy coalescing and local folding

Status: implemented

Status applies to the stated experiment and contract, not to every possible version of the idea.

## Idea in plain language

Remove computations whose values are never used, combine harmless register copies and fold simple constants into consumers. This reduces assembly work without changing the mathematical operation.

## Findings and implementation decision

These passes are in the regular scalar and tensor compilation pipelines before allocation. They remove unnecessary virtual instructions while preserving side effects, predicates and implicit group operands. They are not a universal license to retarget a value's producer: the later alias-binding bug showed that an existing value can have another live name.

No standalone application speedup is attributed only to these passes; historical compiler comparisons combined them with selection, allocation, scheduling and unrolling.

## Method and validity checks

Use semantic differential tests and poisoned-register execution, retain stores/barriers/control flow as side effects, and check that packed-register consumers remain live. Test aliases and conditional definitions rather than checking only instruction counts.

## Reproduction and command set

Run from the repository root. Use Python 3.12+ and an installed test environment. Commands that compile require the CUDA 12.8 toolkit; GPU commands additionally require an NVIDIA driver or an authenticated Modal client. Explicit skips mean a dependency/corpus is unavailable, not a passing hardware result.

```sh
.venv/bin/python -m pytest -q tests/test_backend.py tests/test_backend_ops.py tests/test_backend_loops.py tests/test_reduction_codegen.py
.venv/bin/python -m openassembler compile examples/vector_add.py --verify --out-dir output/local-passes
```

## Code and retained evidence

- [DCE and coalescing implementations](../../../openassembler/backend/ra.py)
- [Standard compilation pipeline](../../../openassembler/build.py)
- [Alias/dependency fixes](scalar-dependency-correctness.md)

## What remains open

Any more aggressive propagation must account for predicates, memory aliases, asynchronous readers and implicit vector operands. A shorter listing that changes those dependencies is not equivalent code.
