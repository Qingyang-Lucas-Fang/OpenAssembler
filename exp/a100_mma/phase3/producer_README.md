# Software producer/consumer warps on SM80

All code changes are isolated in this experiment. `producer_warps.py` generates
the PTX, installs the existing experimental tensor selector in its process, and
locally replaces the one-loop liveness helper with CFG backedge liveness. It
does not edit the OpenAssembler backend, encoder, scheduler, or CLI.

Producer warps use `cp.async`, commit, and `wait_group 0` before every ready
barrier. Consumer warps execute `ldmatrix` and `mma.sync` on the other ring
buffer. Role branches are warp uniform. All threads execute exactly `K/BK`
generations of the same `bar.sync 0`: the first establishes tile zero, and each
subsequent barrier both establishes the next tile and retires prior readers.
The producer exits after the final ready barrier; consumers compute the last
tile and store outputs. No cross-CTA or cross-SM synchronization is assumed.

The functional checks use integer FP16 inputs, poisoned registers, a nonzero
CTA `(1,2)`, the full 128x128 output tile, and sentinels outside the tile. All
five original variants pass this functional SASS emulation. An independent ring
ownership trace checks matching barrier generations and disjoint producer and
consumer slots. Emulation does not test actual asynchronous memory visibility,
scoreboard controls, or resource limits.

Real A100 smoke execution subsequently passed full output checks for four
variants. The 544-thread BK32/32x32 warp-tile case failed with
`CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`; functional emulation cannot detect that
limit. Its source, binaries, and failed result are retained. The large suite
excludes this failed launch. The remaining cases use 288, 160, 192, and 256
threads; their generated native register counts are 106, 200, 182, and 182.
BK64 uses 64 KiB dynamic shared memory; BK32 uses 32 KiB static shared memory.
Additional producer warps cost registers under A100's uniform block allocation.
No runtime register redistribution is available on this architecture.

All four remaining 4096x4096x4096 FP16 variants subsequently passed eight input
patterns (four families, two seeds), complete-output comparison to matched
FP32-output cuBLAS, CPU-float64 sampled oracles, and output canaries.

| BK / consumer warp tile / producer warps | Native ms | Same-worker ptxas ms | Same-worker cuBLAS ms |
|---|---:|---:|---:|
| 32 / 64x32 / 1 | 1.943 | 1.975 | 0.538 |
| 64 / 64x64 / 1 | 1.369 | 1.157 | 0.539 |
| 64 / 64x64 / 2 | 1.019 | 0.743 | 0.577 |
| 64 / 64x64 / 4 | 1.058 | 1.116 | 0.556 |

These are per-case paired worker measurements; clocks vary across workers.
The prior unified-warp direct schedule measured approximately 0.77–0.79 ms in
other workers. This producer/consumer design has not improved that baseline.
Increasing producers from one to two helped within this family, while uniform
register allocation, extra warps, barrier waits, and pointer/address work remain
costs. Hardware counters are needed to assign each cost causally.

From the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/a100_mma/phase3/producer_warps.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/a100_mma/phase3/producer_check.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/a100_mma/phase3/producer_warps.py --large
python3 -m modal run exp/a100_mma/modal_run.py --suite phase3/producer_large_suite.json
```

The default smoke suite preserves the original failing resource experiment; use
the four-case large suite for further timing. Cubins, PTX, disassembly and
compilation metadata are under `producer_build/`. Functional results are in
`producer_emulation.json`; hardware records are in the parent `results/` and
immutable `runs/` directories. None of these results establish a transformer
training speedup or justify merging this schedule into the production compiler.
