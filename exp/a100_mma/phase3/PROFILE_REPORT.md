# Hardware counters for the paired-B and row-epilogue composition

The FP16 square profile confirms that the new OpenAssembler kernel raises
tensor activity and reduces scoreboard stalls at essentially unchanged
occupancy. It still has materially more short-scoreboard stalling than
ptxas compiling the same transformed PTX. The remaining gap is not explained
by occupancy or a missing tensor instruction reduction.

Raw reports are under `profiles/search_combined_square_f16/{baseline,row_bpair}`.
Each directory retains OA and matched-PTX ptxas `.ncu-rep`, raw CSV and
`profile.json`. `profile_summary.py` creates the input-hashed
`derived-summary.json`. All four profiler executions and imports succeeded.

The contract is FP16 inputs, FP32 accumulation/output,4096³, A[M,K] and
B[N,K] row-major, C=A×Bᵀ, CTA128×128×64 with128 threads and65,536 dynamic
shared bytes. The wrapper reads these values directly from
`search_combined_suite.json`; it does not force FP16 onto a BF16 case or
reconstruct shared memory from defaults. A local preparation check verified
both FP16 and BF16 cases; only FP16 has recorded profiles here.

| Metric | OA baseline | OA row+B-pair | ptxas baseline PTX | ptxas row+B-pair PTX |
|---|---:|---:|---:|---:|
| Tensor activity,% of sustained active peak | 63.511 | 75.512 | 78.055 | 85.440 |
| Active-warps occupancy,% | 11.880 | 11.968 | 12.148 | 12.017 |
| Short-scoreboard stall,% per active warp | 14.045 | 11.115 | 1.143 | 3.995 |
| Long-scoreboard stall,% per active warp | 12.660 | 9.876 | 11.989 | 7.853 |
| Wait stall,% per active warp | 44.345 | 50.251 | 36.977 | 41.012 |
| Barrier stall,% per active warp | 3.240 | 1.776 | 3.367 | 9.011 |
| Math-pipe-throttle stall,% per active warp | 12.143 | 11.470 | 20.363 | 17.620 |
| MIO-throttle stall,% per active warp | 0.393 | 0.255 | 4.806 | 3.780 |
| Eligible warps/active cycle | 0.2303 | 0.2527 | 0.2734 | 0.2736 |
| Dynamic instructions,millions | 81.863 | 76.489 | 69.616 | 66.273 |
| Tensor instructions,millions | 33.554 | 33.554 | 33.554 | 33.554 |
| Shared-load wavefronts,millions | 33.726 | 33.612 | 33.700 | 33.657 |
| Shared-load bank-conflict count | 170,940 | 57,238 | 144,314 | 101,411 |
| Shared-store bank-conflict count | 0 | 0 | 0 | 0 |
| DRAM bytes,decimal MB | 397.387 | 263.869 | 264.877 | 373.840 |
| L2 sector hit rate,% | 78.202 | 84.809 | 85.029 | 78.942 |
| Registers/thread | 228 | 232 | 242 | 244 |
| Register/shared residency limit,CTAs/SM | 2/2 | 2/2 | 2/2 | 2/2 |
| Profile duration,µs; not benchmark latency | 833.312 | 613.408 | 646.400 | 708.512 |

OA tensor activity rises by12.001 percentage points. Short-scoreboard
stalling falls2.930 percentage points, long-scoreboard stalling falls2.784
points and eligible warps rise about9.76%. The register increase does not
change the recorded two-CTA residency limit; achieved occupancy barely
changes. This supports a gain in issue/dependency behavior at the existing
occupancy tier.

The combined kernel issues the same33,554,432 tensor instructions. B
pairing reduces static LDSM instruction count while moving the same matrix
data; the shared-load wavefront count falls only about0.34%. The absolute
bank-conflict count falls about66.5% but remains nonzero. The requested
shared-load request counter is unavailable in this environment, so there
is no measured request denominator for a bank-conflict rate. This evidence
does not establish that shared traffic itself fell by one third or that
bank conflicts were eliminated.

Dynamic total instructions fall about6.56%. The static count falls much
more because most removed epilogue instructions execute once, while the
K-loop dominates a4096-depth multiplication. Tensor work is unchanged.

The remaining short-scoreboard fraction is11.115% for OA versus3.995% for
matched-PTX ptxas. That identifies fragment availability/short-latency
dependency scheduling as an actionable remaining target. These aggregate
counters do not distinguish operand collection from every source of short
scoreboard stalls. Per-PC samples, LDSM dependency distances and isolated
load/compute scheduling sweeps are needed before assigning the whole gap
to one hardware mechanism.

The wait fraction rises from44.345% to50.251% while independent timing
benchmarks improve. Stall percentages use active-warp denominators and
change when other costs shrink. They are not absolute lost cycles, cannot
be added into a simple time budget, and do not imply that the transformed
kernel became slower. Similarly, the DRAM and L2 differences should not be
attributed entirely to either B pairing or epilogue vectorization: this
profile changes both, and cache/replay/CTA timing affect off-chip traffic.

Nsight runs replay multiple passes, uses `--cache-control all`, leaves
clocks unlocked and profiles one regular launch after three matching
launches. In particular, ptxas has higher tensor activity yet a longer
recorded profile duration for the transformed source. Treat these
durations as profiling context, not as a benchmark ordering or a percentage
speedup. The shuffled CUDA-graph benchmark medians establish performance;
counters identify plausible next experiments. BF16 needs its own profiles.

The wrapper uses the existing phase2 remote function and writes phase3
outputs. The default profiles scalar baseline and row+B-pair, each against
ptxas on that exact PTX; `--variants all` supports all six candidates.

```sh
python3 -m modal run exp/a100_mma/phase3/profile_combined.py
python3 -m modal run exp/a100_mma/phase3/profile_combined.py --case search_combined_square_bf16
.venv/bin/python exp/a100_mma/phase3/profile_summary.py
```
