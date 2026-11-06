# Hardware-counter-guided tensor analysis

Status: validated-not-integrated

Status applies to the stated experiment and contract, not to every possible version of the idea.

## Idea in plain language

Measure what the GPU actually waits for: tensor activity, shared bank traffic, short/long scoreboards, barriers, dynamic instructions and cache/DRAM behavior. Use these to choose the next experiment rather than guessing from SASS length.

## Findings and implementation decision

Retained matched profiles show equal tensor work but different overhead and stalls. One early direct/same-PTX comparison reported tensor activity 59.43% versus 78.26%, short-scoreboard stalls 13.79% versus 1.15%, and 90.85M versus 69.62M executed instructions. Later schedule profiles improved tensor activity, and November 5 repeats examined Morton locality.

These are validated measurements and research harnesses, not an integrated automatic compiler cost model. They do not conclusively isolate register-bank or operand-collector causes. Newer direct-pointer and whole-stack counter analyses have their own dossiers.

## Method and validity checks

Validate identical outputs and tensor work first, query available NCU metrics for the installed version, retain raw rows and resource/device identity, and collect each method on matching shapes. NCU replay changes caches and may run many passes; use separate CUDA-graph timings for performance claims.

## Reproduction and command set

Run from the repository root. Use Python 3.12+ and an installed test environment. Commands that compile require the CUDA 12.8 toolkit; GPU commands additionally require an NVIDIA driver or an authenticated Modal client. Explicit skips mean a dependency/corpus is unavailable, not a passing hardware result.

```sh
PYTHONPATH=. modal run exp/a100_counters/run.py --mode preflight
PYTHONPATH=. modal run exp/a100_counters/run.py --mode build-current
PYTHONPATH=. modal run exp/a100_counters/run.py --mode validate-cublas-ncu
```
Read the harness options before scheduling cloud work. NCU collection needs the profiler and permissions in its remote image. A failed collection is not a zero-counter result, and preflight itself can request an A100.

## Code and retained evidence

- [Retained counter runner](../../../exp/a100_counters/run.py)
- [Earlier counter/resource follow-up](../../evidence/a100-mma/phase3/profile-followup.json)
- [November 5 counter records](../../../exp/a100_counters/counter-validation_20261105T171913.378083Z.json)
- [Newer direct-pointer work](direct-pointer-homes.md)

## What remains open

Turn reproducible counter diagnoses into narrowly controlled candidate changes. Counter percentages are correlated observations, not by themselves a causal model or a general performance prediction.
