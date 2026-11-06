# Loadable cubins and metadata-safe code replacement

Status: implemented

Status applies to the stated experiment and contract, not to every possible version of the idea.

## Idea in plain language

A kernel needs more than instruction bytes. Its CUDA binary also describes parameters, registers, entry points and resources. Keep a ptxas-built ELF container, replace only selected kernel instructions, and update the metadata required to load and execute them.

## Findings and implementation decision

The production container code parses/serializes CUDA ELF and .nv.info records, replaces kernel text, fixes exit offsets and declares the generated register count. An A100 sweep found that writing register Rd requires a declared count of at least Rd+3, with a minimum allocation used by the helper. Both register metadata locations matter on SM80.

This is not a from-scratch replacement for every CUDA ABI, device-linking or fatbin feature. ptxas still supplies the container and the same-source reference binary.

## Method and validity checks

Check byte-identical ELF serialization, resource records and disassembly after growth/replacement. Execute rewritten cubins against known outputs and separately vary declared register counts. The historical 77-variant count sweep and a 15-variant text-growth sweep distinguish a valid encoding from a loadable binary.

## Reproduction and command set

Run from the repository root. Use Python 3.12+ and an installed test environment. Commands that compile require the CUDA 12.8 toolkit; GPU commands additionally require an NVIDIA driver or an authenticated Modal client. Explicit skips mean a dependency/corpus is unavailable, not a passing hardware result.

```sh
.venv/bin/python -m pytest -q tests/test_backend.py tests/test_gpu_truth.py::test_regcount_rule
.venv/bin/python -m openassembler compile examples/vector_add.py --verify --out-dir output/container-check
.venv/bin/python -m openassembler run output/container-check --backend modal --budget 120 --job-timeout 30
```
Compilation can run without a GPU; the last command is hardware smoke, not a throughput experiment.

## Code and retained evidence

- [ELF reader/writer](../../../openassembler/container/elf.py)
- [CUBIN replacement and register counting](../../../openassembler/container/cubin.py)
- [NVIDIA metadata parser](../../../openassembler/container/nvinfo.py)
- [Register-count hardware record](../../../corpus/sm_80/gpu_a100_40gb_bisect5.json)
- [Hardware replay tests](../../../tests/test_gpu_truth.py)

## What remains open

Arbitrary linked device functions, unfamiliar relocations, fatbin packaging and newer CUDA ABIs are not implied by the tested SM80 splice path. Each needs independent binary and execution checks.
