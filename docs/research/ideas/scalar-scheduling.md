# Measured scalar instruction scheduling

Status: implemented

Status applies to the stated experiment and contract, not to every possible version of the idea.

## Idea in plain language

Place independent instructions between a producer and its consumer, and wait only when data or operand collection requires it. Measure these requirements on A100 rather than treating the smallest stall printed by ptxas as a latency table.

## Findings and implementation decision

Corrected fixed-pipe probes found four cycles for tested same-pipe ALU/FMA pairs, five across tested classes, and 13 from a predicate definition to tested guarded instructions/branches. Predicate-to-SEL needed four. The production scheduler uses these classes with conservative fallbacks; variable-latency results and asynchronous readers also use scoreboards.

The first clock harness included generic fragment-copy work and was rejected. A held-out match to 89.2% of ptxas stall fields was not evidence of hardware timing equivalence. These scalar measurements are not an HMMA pipeline model.

## Method and validity checks

Create dependent producer/consumer pairs, vary one encoded stall while keeping arithmetic fixed, and require correct GPU outputs at and above a candidate threshold. Separate predicate guards from ordinary source operands. Shuffle matched timing order to avoid GPU ramp-up bias.

## Reproduction and command set

Run from the repository root. Use Python 3.12+ and an installed test environment. Commands that compile require the CUDA 12.8 toolkit; GPU commands additionally require an NVIDIA driver or an authenticated Modal client. Explicit skips mean a dependency/corpus is unavailable, not a passing hardware result.

```sh
.venv/bin/python -m pytest -q tests/test_fastmm.py tests/test_backend.py tests/test_reduction_codegen.py
PYTHONPATH=. .venv/bin/python tools/lat_sweep.py --arch sm_80
```
The second command builds new probe cubins. It does not run a GPU or automatically reproduce the archived corrected 42-case run; its results need the GPU batch tools and separately retained raw outputs. Existing corrected results can be inspected without rerunning hardware.

## Code and retained evidence

- [Scheduler](../../../openassembler/backend/sched.py)
- [Dependency controls](../../../openassembler/backend/compile.py)
- [Corrected latency results](../../evidence/a100-latency/latency-summary.json)
- [Full corrected hardware record](../../evidence/a100-latency/gpu_a100_40gb_latsweep_corrected.json)
- [Latency generator](../../../tools/lat_sweep.py)

## What remains open

Measure more operand forms and instruction pairs; build a distinct clocked tensor model. Later source-WAR and safe-scoreboard reuse fixes are mandatory correctness edges, not optional latency tuning.
