# Bounded joint warp-layout and address-folding experiment

On 4096³ GEMMs, folding shared addresses makes the eight-warp layouts
19.38–20.36% faster than their unfolded controls, but they remain 2.78–5.58%
slower than the Warp64×64 row+paired-B reference. CUDA reports exactly 128
registers per thread for folded kernels and 136 for unfolded kernels. This
crosses the static resource threshold from one to two 256-thread CTAs at
64 KiB shared memory per CTA. Actual residency and the causes of the timing
change have not been established by this experiment.

| CTA 128×128×64, two stages | Threads | Registers/thread | Static CTA bound | Static resident warps |
|---|---:|---:|---:|---:|
| Warp64×64 row + paired-B reference | 128 | 232 | 2 | 8 |
| Warp64×32, unfolded copy addresses | 256 | 136 | 1 | 8 |
| Warp64×32, folded copy addresses | 256 | 128 | 2 | 16 |
| Warp32×64, unfolded copy addresses | 256 | 136 | 1 | 8 |
| Warp32×64, folded copy addresses | 256 | 128 | 2 | 16 |

The model rounds register allocation to eight registers per thread and applies
the 65,536-register SM limit, the two 64-KiB shared allocations and thread
limit. It is an upper bound; compiler resource counts are not a measurement
of issued tensor instructions, actual active warps, or execution time.

This sweep is deliberately bounded to two warp shapes and folded/unfolded
copy addressing, with the existing Warp64×64 combination as the control. All
variants use the same FP16/BF16 input type, FP32 accumulation/output, paired B
LDSM x4 operations, reused-row vector epilogue, HMMA8 schedule, reuse flags,
CTA tile, K64 mainloop and explicit two-stage synchronization. Warp mapping,
fragment dimensions and the copy-chunk layout are derived from each geometry.
At 256 threads, each operand has four copy chunks, with 32-row / 4096-byte
shared offsets that preserve the row&7 XOR swizzle.

`address_occupancy.py` compiled all 18 new kernels: eight alternatives each
for smoke 256×384×256 and large 4096³, plus two smoke references. The large
reference reuses the preserved `search_combined_square_row_bpair_{f16,bf16}`
builds. Every cubin's actual spliced words, scheduled controls and
tensor/memory/store operands matched nvdisasm. Full provenance and the static
resource model are stored in each `experiment.json`; a compact resource list
is `address_occupancy_resources.json`.

Full-CTA semantic checks cover all eight dtype/layout/folding combinations
on 128×128×128 GEMMs with 256 threads and poisoned registers. Every one of
16,384 outputs matches the exact CPU integer oracle, and actual encoded
LDGSTS words match nvdisasm. All eight checks passed in 92.41 seconds;
the result is recorded in `address_occupancy_final_test.log`.

`address_occupancy_smoke_suite.json` and
`address_occupancy_large_suite.json` each contain two same-worker comparisons,
one per dtype, with the control and all four alternatives. The parent ran both
suites with the existing full-output/canary/float64-sample GPU validation and
16 warm-up graph rounds. This subagent has not run GPU jobs.

The large run used an **A100-SXM4-40GB**. Times below are medians in milliseconds
of seven randomized-order measurements, with 20-node CUDA graphs. Folding
reduction is `100*(1-folded/unfolded)`; folded overhead compares against the
same-worker Warp64×64 reference. Table values are rounded to nine decimal
places; the linked raw records retain full precision.

| Input | Warp | Warp64×64 reference ms | Unfolded ms | Folded ms | Folding time reduction | Folded overhead vs reference |
|---|---|---:|---:|---:|---:|---:|
| FP16 | 64×32 | 0.663808012 | 0.865536022 | 0.689305592 | 20.36% | 3.84% |
| FP16 | 32×64 | 0.663808012 | 0.846284771 | 0.682291222 | 19.38% | 2.78% |
| BF16 | 64×32 | 0.642867184 | 0.845516777 | 0.678758383 | 19.72% | 5.58% |
| BF16 | 32×64 | 0.642867184 | 0.833996773 | 0.667187214 | 20.00% | 3.78% |

The smoke run used an **A100-SXM4-80GB** on 256×384×256. All eight-warp
variants beat the Warp64×64 reference here, while folding itself gives little
benefit or a small regression. These launch-scale results do not establish
the same ordering for large GEMMs, and the two sizes were run on different
GPU memory variants.

| Input | Warp | Warp64×64 reference ms | Unfolded ms | Folded ms | Folding time reduction | Folded overhead vs reference |
|---|---|---:|---:|---:|---:|---:|
| FP16 | 64×32 | 0.013260800 | 0.011468800 | 0.011724800 | -2.23% | -11.58% |
| FP16 | 32×64 | 0.013260800 | 0.011571200 | 0.011622400 | -0.44% | -12.36% |
| BF16 | 64×32 | 0.013260800 | 0.011468800 | 0.011724800 | -2.23% | -11.58% |
| BF16 | 32×64 | 0.013260800 | 0.011622400 | 0.011571200 | 0.44% | -12.74% |

All four workers returned zero, each with 56 check records across zero,
integer, normal and scaled patterns with two seeds. Each check covered all
98,304 smoke or 16,777,216 large output elements against matched cuBLAS, plus
the float64 sample oracle, canaries and finite-output checks. Zero and integer
cases passed exact comparisons. Maximum full-output relative L2 was zero /
1.5706e-7 for FP16/BF16 smoke and 3.9812e-6 / zero for FP16/BF16 large, within
the 2e-5 threshold. Runtime resource records agree with 136/128 registers and
report zero local-memory bytes for all four candidate families.

Preserved raw artifacts, including actual samples and cubin/PTX provenance:

- [Large FP16](../runs/20261104T000337.992702Z/address_occupancy_large_f16.json)
- [Large BF16](../runs/20261104T000337.992702Z/address_occupancy_large_bf16.json)
- [Smoke FP16](../runs/20261104T000158.273893Z/address_occupancy_smoke_f16.json)
- [Smoke BF16](../runs/20261104T000158.273893Z/address_occupancy_smoke_bf16.json)

The resource threshold is useful for constructing the candidate, but neither
the register count nor the resident-warp bound selects the fastest kernel:
the large-case reference still wins with a bound of eight resident warps.
Folding changes address instructions, dependency behavior and allocation
together. This result does not isolate occupancy as the cause; no counter
interpretation is assumed here. Keep Warp64×64 for the measured large cases
and retain these variants as validated controls for future scheduling work.

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/a100_mma/phase3/address_occupancy.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  exp/a100_mma/phase3/address_occupancy_test.py
PYTHONDONTWRITEBYTECODE=1 python3 -m modal run exp/a100_mma/modal_run.py \
  --suite phase3/address_occupancy_smoke_suite.json
PYTHONDONTWRITEBYTECODE=1 python3 -m modal run exp/a100_mma/modal_run.py \
  --suite phase3/address_occupancy_large_suite.json
```
