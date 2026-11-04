# Follow-up hardware evidence: stage specialization and occupancy

The tall-kernel gain occurs at unchanged CTA residency and essentially
unchanged active-warps occupancy. The occupancy experiment separately proves
that shared-address folding crosses a register threshold and nearly doubles
active warps. These are different mechanisms; neither result supports
maximizing occupancy as a universal objective.

All eight OA/ptxas profiler runs and report imports succeeded. The raw
reports live under `profiles/<build>/`; `profile_followup.py` validates their
actual launch contracts and produces the input-hashed
`profile-followup-summary.json`. All new counter profiles are **FP16**, with
FP32 accumulation/output. BF16 has timing/correctness evidence, not matching
follow-up counters.

## Tall stage-pair specialization

M×N×K=16384×4096×4096; grid128×32×1;128 threads/CTA;65,536 dynamic shared
bytes/CTA. Stage specialization combines constant stage addresses with a
two-tile loop. It retains the existing copy waits/barriers and output
arithmetic contract.

| Metric | OA row+B-pair | OA stage pair | ptxas row PTX | ptxas stage PTX |
|---|---:|---:|---:|---:|
| Registers/thread | 232 | 216 | 244 | 234 |
| Register/shared residency limit,CTAs/SM | 2/2 | 2/2 | 2/2 | 2/2 |
| Active-warps occupancy,% | 12.533 | 12.434 | 12.290 | 12.446 |
| Tensor activity,% of sustained active peak | 66.530 | 72.565 | 52.080 | 52.013 |
| Short-scoreboard stall,% per active warp | 9.104 | 9.233 | 2.143 | 1.027 |
| Long-scoreboard stall,% per active warp | 21.306 | 16.586 | 51.810 | 55.481 |
| Wait stall,% per active warp | 41.755 | 45.346 | 24.105 | 22.794 |
| Eligible warps/active cycle | 0.2265 | 0.2366 | 0.1592 | 0.1264 |
| Dynamic instructions,millions | 305.955 | 287.130 | 265.093 | 217.383 |
| Tensor instructions,millions | 134.218 | 134.218 | 134.218 | 134.218 |
| Shared-load wavefronts,millions | 134.493 | 134.491 | 134.585 | 134.480 |
| Shared-load bank-conflict count | 268,751 | 267,466 | 358,019 | 250,634 |
| DRAM traffic,decimal GB | 3.628 | 2.934 | 4.424 | 4.534 |
| L2 sector hit rate,% | 59.094 | 65.026 | 52.827 | 52.085 |
| Profile duration,ms; not benchmark latency | 2.980 | 2.466 | 3.411 | 3.463 |

OA stage-pair execution has about6.15% fewer dynamic instructions and19.14%
less observed DRAM traffic, with tensor work and shared-load wavefronts
unchanged. Tensor activity rises6.035 percentage points and long-scoreboard
stalling falls4.721 points. Short-scoreboard stalling does not improve.
Register use falls, but neither the two-CTA resource limit nor achieved
occupancy increases; configured shared memory remains135,168 bytes.

The evidence is consistent with reduced instruction overhead and improved
global-memory/cache behavior. It does **not** establish why tall shapes gain
more than square/MLP. Address folding, loop specialization, register
placement and resulting instruction timing change together. Lower DRAM
traffic can reflect temporal cache reuse/CTA timing rather than fewer
algorithmic input bytes. A cache-size sweep, controlled register placement
and per-PC dependency evidence would be needed to isolate the cause. The
wait fraction's increase is not an absolute increase in lost cycles.

## Register threshold and actual occupancy

4096³; warp64×32;256 threads/CTA; grid32×32×1. Both folded and unfolded
kernels retain65,536 dynamic shared bytes and1,024 driver shared bytes/CTA.

| Metric | OA unfolded | OA folded | ptxas unfolded PTX | ptxas folded PTX |
|---|---:|---:|---:|---:|
| Registers/thread | 136 | 128 | 128 | 127 |
| Allocated registers/thread | 136 | 128 | 128 | 128 |
| Register/shared residency limit,CTAs/SM | 1/1 | 2/2 | 2/2 | 2/2 |
| Configured shared memory,bytes/SM | 102,400 | 135,168 | 135,168 | 135,168 |
| Active-warps occupancy,% | 12.477 | 23.991 | 23.872 | 23.867 |
| Active warps/SM,from occupancy | 7.985 | 15.355 | 15.278 | 15.275 |
| Tensor activity,% of sustained active peak | 59.088 | 72.984 | 81.013 | 79.875 |
| Eligible warps/active cycle | 0.2305 | 0.3826 | 0.4643 | 0.4700 |
| Short-scoreboard stall,% per active warp | 17.577 | 16.192 | 13.212 | 10.718 |
| Long-scoreboard stall,% per active warp | 8.707 | 8.128 | 3.410 | 5.097 |
| Wait stall,% per active warp | 40.440 | 27.095 | 19.756 | 19.846 |
| Barrier stall,% per active warp | 8.277 | 13.697 | 9.496 | 12.587 |
| Math-pipe-throttle stall,% per active warp | 9.870 | 20.835 | 28.677 | 25.028 |
| Tensor instructions,millions | 33.554 | 33.554 | 33.554 | 33.554 |
| Shared-load wavefronts,millions | 50.332 | 50.373 | 50.412 | 50.398 |
| Shared-load bank-conflict count | 0 | 35,880 | 74,128 | 60,488 |
| Profile duration,µs; not benchmark latency | 798.240 | 820.320 | 668.512 | 751.392 |

Hardware reports65,536 registers and64 maximum warps/SM. At256 threads,
136 allocated registers/thread requires34,816 registers/CTA; two exceed the
register file. At128 registers/thread,32,768 registers/CTA permits two.
The measured residency limit changes1→2, and active warps rise approximately
8→15.35, validating the occupancy increase on hardware.

The configured shared-memory carveout also changes:102,400→135,168 bytes.
At66,560 total shared bytes/CTA, only the latter holds two CTAs. The wrapper
sets dynamic-shared opt-in and does not explicitly request a preferred
carveout. This is a coupled register/allocation/configuration result, not
an isolated measurement of register pressure alone.

Higher occupancy accompanies higher tensor activity and more eligible
warps, with more math-pipe and barrier contention. Bank-conflict count rises
from zero despite the performance gain; minimizing that counter alone
would select the slower kernel.

The independent same-worker graph benchmarks show:

| Warp shape/input | Unfolded,ms | Folded,ms | Folding latency reduction | Folded gap to same-run64×64 row+B-pair |
|---|---:|---:|---:|---:|
| 64×32 FP16 | 0.865536 | 0.689306 | 20.36% | +3.84% |
| 64×32 BF16 | 0.845517 | 0.678758 | 19.72% | +5.58% |
| 32×64 FP16 | 0.846285 | 0.682291 | 19.38% | +2.78% |
| 32×64 BF16 | 0.833997 | 0.667187 | 20.00% | +3.78% |

The smaller warp tiles still lose to64×64. Warp64×32 duplicates A fragments
across four N warps instead of two; its measured shared-load wavefront count
is about1.5× the earlier64×64 profile, at identical tensor-instruction
count. That is a concrete reuse tradeoff, although these profiles do not
isolate what fraction of the total latency gap it causes.

## Benchmark and causality limits

Counter reports use replay passes, cache control and unlocked clocks. Their
duration can disagree with benchmark ordering: the folded occupancy kernel
has a longer profile duration while its independent benchmark latency is
about20% lower. Profile durations are not used to compute speedups. Stall
fractions and tensor activity use different denominators; they cannot be
summed into a simple wall-clock time budget. The shared-load request counter
remains unavailable, so no normalized bank-conflict rate is claimed.

The final same-worker six-case evidence in `evidence-summary.json` confirms
16.13–26.69% lower latency against the prior scalar-epilogue experimental
kernel, with all full-output checks passing. Tall-stage improvement repeats;
the mechanism behind its larger gain remains unresolved. These results
cover full-tile dense GEMMs with matched FP32 outputs, not a complete
transformer-training step or universal cuBLAS parity.

Reproduce the derived analysis without a GPU:

```sh
.venv/bin/python exp/a100_mma/phase3/profile_followup.py
```
