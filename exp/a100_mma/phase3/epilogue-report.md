# Vectorized output epilogue experiment

This candidate changes only output address generation and stores in generated
full-tile RCR GEMMs. No production file or pre-existing experimental module was
edited. The new files are `epilogue.py`, `epilogue_test.py`, generated PTX,
suite JSON and logs under phase3; compiled outputs are parent build directories
named `p3_epilogue_*`.

The 128×128×64 CTA / 64×64 warp baseline uses the phase-two passing options:
FP16/BF16 inputs, FP32 output and accumulation, tensor S8, reserved async group
SB0, explicit address WAR tracking, direct loop-carried pointer increments,
column-major MMA traversal with B reuse hints, two shared stages and the original
XOR layout. All alternatives use matching precision and the same mainloop.

| Variant | Static non-NOP instructions | Output stores | IMAD.WIDE.U32 | Registers |
|---|---:|---|---:|---:|
| Scalar baseline | 1,812 | 128 × STG.E | 144 | 228 |
| Pair stores | 1,428 | 64 × STG.E.64 | 80 | 228 |
| Pair stores, reused output-row base | 1,140 | 64 × STG.E.64 | 24 | 228 |

Each thread owns pairs of adjacent columns in each HMMA accumulator group.
Because N is even, the tile/warp starts are even, and lane column coordinates
are `2*(lane%4)`, every pair address is naturally eight-byte aligned. A row
pointer is formed once per `(mi, low/high row half)`; N-fragment displacements
are static byte offsets in the stores. This saves repeated multiply-wide and
64-bit address additions without extending the mainloop accumulator live range.

An optional selector (`NativePairSelector`) declares both scalar fragment reads
while referencing an aligned half of the existing four-register accumulator.
The normal selector initially inserts pair packing MOVs, but allocation and
coalescing remove them in this exact generated family. The encoded code and
words are equal in `epilogue-selector-equivalence.json`, so a production selector
extension is not justified by this case alone. The useful candidates can be
implemented by PTX output generation using the existing backend.

Six tests passed in `epilogue-final-tests.log` (18.02 seconds). Tests check every
scalar output address exactly once across a 256×384 grid, eight-byte alignment,
full 64×64×128 GEMM output versus an exact integer CPU oracle, poison-register
independence, actual binary encoding, and complete pair-source dependencies.
The tensor emulator is a functional check; the hardware checks below establish
the tested cases on A100.

The preserved A100-SXM4-80GB run at `20261103T232923.792831Z` passed both
4096×4096×4096 cases with FP32 accumulation and output:

| Input | Scalar baseline ms | Pair native ms | Reused-row native ms | Baseline / row | Row time reduction | cuBLAS ms |
|---|---:|---:|---:|---:|---:|---:|
| FP16 | 0.814541 | 0.773478 | 0.771635 | 1.0556× | 5.27% | 0.587571 |
| BF16 | 0.800410 | 0.746803 | 0.749978 | 1.0672× | 6.30% | 0.568422 |

These are medians of seven randomized-order, same-worker measurements per
method, using 20-node CUDA graphs and 16 additional warm-up graph rounds. Row
throughput is 178.11/183.26 TFLOP/s, or 76.15%/75.79% of the matched cuBLAS
throughput in these two cases. Pair-native and row-native timing differences
are small; this run does not establish that row-base reuse improves on pair
stores. The measured reduction against the scalar baseline is the useful
result. A single run does not establish a timing confidence interval or a
transformer-training step improvement.

Each method passed all 16,777,216 output elements for zero/integer/normal/scaled
patterns with seeds 17 and 29, output-canary and finite-output checks, and a
32×32 CPU float64 sample oracle. Zero/integer comparisons were exact. Across
the baseline and row variants, maximum full-output relative L2 versus cuBLAS
was 3.9812e-6 for FP16 and zero for BF16, within the 2e-5 threshold.

Exact preserved raw artifacts are
`../runs/20261103T232923.792831Z/p3_epilogue_large_f16.json` and
`../runs/20261103T232923.792831Z/p3_epilogue_large_bf16.json`; the smoke artifacts
are under `../runs/20261103T232613.706671Z/`. Build directories are
`../build/p3_epilogue_{scalar,pair_packed,pair_native,row_native}_4096_4096_4096_{f16,bf16}`.
The records retain cubin/PTX hashes and compilation provenance. Run
`epilogue_summary.py` to regenerate `epilogue-benchmark-summary.json` from those
immutable raw records, including hashes and unrounded numbers.

Ready-to-run GPU suites are `epilogue-smoke-suite.json` and
`epilogue-large-suite.json`. Both have FP16 and BF16 cases, one scalar baseline
and three alternatives in the same worker, 16 extra warm-up graph rounds, and
the existing GPU harness's full-output/canary/float64-sample checks:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/a100_mma/phase3/epilogue.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/a100_mma/phase3/epilogue.py --large
PYTHONDONTWRITEBYTECODE=1 python3 -m modal run exp/a100_mma/modal_run.py \
  --suite phase3/epilogue-smoke-suite.json
PYTHONDONTWRITEBYTECODE=1 python3 -m modal run exp/a100_mma/modal_run.py \
  --suite phase3/epilogue-large-suite.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/a100_mma/phase3/epilogue_summary.py
```

A separate confirmation suite preserves the original builds and compares only
scalar and row-native alternatives on six phase-two projections: square
4096×4096×4096, MLP 4096×16384×4096, and tall 16384×4096×4096, each in FP16 and
BF16. These twelve new cubins use names
`../build/p3_epilogue_confirm_{square,mlp,tall}_{f16,bf16}_{scalar,row_native}`.
`epilogue-confirm-build.log` records successful compilation. All cases retain
same-worker alternatives, full-output checks and 16 warm-up graph rounds.
The confirmation suite is built; it has not been run by this subagent.

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/a100_mma/phase3/epilogue_confirm.py
PYTHONDONTWRITEBYTECODE=1 python3 -m modal run exp/a100_mma/modal_run.py \
  --suite phase3/epilogue-confirm-suite.json
```

The mainloop PTX is unchanged, so the static instruction reduction is confined
to one output epilogue per CTA and does not scale with the K-loop count.
