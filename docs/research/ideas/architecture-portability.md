# Cross-architecture encoding and cubin observations

Status: untested-or-hardware-unavailable

Status applies to the stated experiment and contract, not to every possible version of the idea.

## Idea in plain language

Check which low-level assumptions survive across NVIDIA GPU generations, then use those observations to plan separate backends. A shared field layout does not imply that every opcode, ABI or schedule is portable.

## Findings and implementation decision

Historical CUDA 12.8 probes examined common control fields on SM75/86/90/100/120 and parameter/ELF metadata changes. Parameter banks moved from the observed SM80 base 0x160 to 0x210 on SM90 and 0x380 on SM100+. Newer CUDA ELF ABIs also changed flags, notes and register-count representation.

The supported native product target remains SM80. Similar printed control fields do not establish a complete H100/Blackwell encoder, numerical model, cubin writer or hardware-tested compiler. The observations are limited oracle research, not a cross-architecture performance result.

## Method and validity checks

Compile the same controlled probe for each supported toolkit target, compare raw words/metadata, mutate fields and inspect the tool response. Hardware execution must be repeated per architecture; an A100 check cannot validate a different GPU's timing or memory protocol.

## Reproduction and command set

Run from the repository root. Use Python 3.12+ and an installed test environment. Commands that compile require the CUDA 12.8 toolkit; GPU commands additionally require an NVIDIA driver or an authenticated Modal client. Explicit skips mean a dependency/corpus is unavailable, not a passing hardware result.

```sh
env/cuda.sh ptxas --version
.venv/bin/python -m openassembler status
.venv/bin/python -m pytest -q tests/test_word.py tests/test_backend.py
```
These establish current tools and SM80 regressions. Original cross-architecture temporary probes were retired. Their detailed source-audit note is preserved in the documentation-history snapshot as `docs/research/02-control-bits-and-containers.md`; no complete later-GPU replay command is claimed.

## Code and retained evidence

- [Current target dispatch and build](../../../openassembler/build.py)
- [Container implementation](../../../openassembler/container/elf.py)
- [Current encoding specifications](../../../specs/sm_80/synth_templates.json)
- [Capability record](../../evidence/a100-mma/phase2/literature-discovery.json)

## What remains open

Create separate corpora, instruction maps, ABI handling, synchronization models and hardware gates before advertising another architecture. Keep unverified recollections out of a production specification.
