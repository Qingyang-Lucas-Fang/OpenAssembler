# Persisting-L2 cache allocation policies

Status: tested-not-selected

Status applies to the stated experiment and contract, not to every possible version of the idea.

## Idea in plain language

Reserve part of the GPU's last-level cache and mark selected operand ranges as likely to be reused. The same capacity then becomes less available for other traffic, so a reservation can hurt rather than help.

## Findings and implementation decision

The controlled cache-policy study performed 224 output checks across square/tall FP16/BF16 and policy settings, with graph-node policy readback. Full reservations slowed square cases about 8% and some tall cases by more than 30%. Refined 8 MiB tall settings were effectively neutral/noisy; 16 MiB and larger regressed. No persisting-cache policy is selected by the compiler.

This CUDA API is available on A100, but availability is not a speedup. It is distinct from cp.async prefetch modifiers and from the separate project named CUDA-L2.

## Method and validity checks

Use identical kernels, operands and precision, verify the actual access-policy window on the graph node, check all outputs, and time repeated paired replays. Report reservation size/hit ratio and reset policies between methods. Counter replay perturbs cache state and must not replace event timing.

## Reproduction and command set

Run from the repository root. Use Python 3.12+ and an installed test environment. Commands that compile require the CUDA 12.8 toolkit; GPU commands additionally require an NVIDIA driver or an authenticated Modal client. Explicit skips mean a dependency/corpus is unavailable, not a passing hardware result.

```sh
python3 -m json.tool docs/evidence/a100-mma/phase5/l2-summary.json
python3 -m json.tool docs/evidence/a100-mma/phase5/l2-results.json
python3 -m json.tool docs/evidence/a100-mma/phase5/l2-refine-results.json
```
The original policy-sweep runner was retired. There is no production `--persisting-l2` option. Rebuilding the experiment requires graph-node attribute handling as well as a kernel launch.

## Code and retained evidence

- [Decision summary](../../evidence/a100-mma/phase5/l2-summary.json)
- [Primary policy sweep](../../evidence/a100-mma/phase5/l2-results.json)
- [Refinement](../../evidence/a100-mma/phase5/l2-refine-results.json)
- [Prefetch is a separate idea](prefetch-and-fragment-windows.md)

## What remains open

Other reuse distances, operand sizes and graph partitions may differ. A persistent-cache policy needs shape- and workload-aware evidence, not a globally enabled reservation.
