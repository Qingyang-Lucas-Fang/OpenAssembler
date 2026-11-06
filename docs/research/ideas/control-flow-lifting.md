# Control-flow recovery and a SASS-to-PTX lifter

Status: untested-or-hardware-unavailable

Status applies to the stated experiment and contract, not to every possible version of the idea.

## Idea in plain language

Recover a readable program from assembly: find branches and loops, track register values and predicates, then translate their meaning back to PTX or another intermediate language. This is the reverse direction from the working PTX-to-SASS compiler.

## Findings and implementation decision

Historical CUDA 12.8 probes identified useful building blocks: predicated branches have taken/fall-through edges; BSSY/BSYNC delimit reconvergence regions; indirect branch tables use constant memory; software registers carry return addresses; and floating/division helper windows can be recognized as higher-level operations.

The repository does not currently contain a general CFG/SSA/lifter product or a validated arbitrary-kernel SASS-to-PTX CLI. Source inspection of Mesa NAK, NVLift/CuLifter and related systems was prior-art research, not imported functionality or a successful end-to-end OpenAssembler lift.

## Method and validity checks

The proposed method compares a recovered basic-block edge set with nvdisasm -bbcfg, checks register use/definition sets against -plr, models predicated writes as conditional updates, and validates lifted programs against original hardware outputs. Recompilation alone is insufficient: it can accept a program with the wrong semantics.

## Reproduction and command set

Run from the repository root. Use Python 3.12+ and an installed test environment. Commands that compile require the CUDA 12.8 toolkit; GPU commands additionally require an NVIDIA driver or an authenticated Modal client. Explicit skips mean a dependency/corpus is unavailable, not a passing hardware result.

There is no implemented lift command to reproduce. The current prerequisites can be inspected without claiming a completed lifter:

```sh
.venv/bin/python -m pytest -q tests/test_word.py tests/test_emu.py
.venv/bin/python -m openassembler compile examples/vector_add.py --out-dir output/lifting-input
```
The original detailed probe note is preserved in the documentation-history snapshot under its former path `docs/research/03-lifting-and-ptx-bridge.md`. Its temporary probe generator was retired; that full historical run is not one-command reproducible today.

## Code and retained evidence

- [Current assembly parser](../../../openassembler/emu/program.py)
- [Current semantic execution](../../../openassembler/emu/ops.py)
- [Available PTX parser/interpreter](../../../openassembler/ptx/interp.py)

## What remains open

Implement CFG and conditional-definition IR, reconvergence, indirect targets, calls and memory spaces, then a verified PTX emitter. Preserve scheduling metadata as analysis input rather than treating it as arithmetic semantics.
