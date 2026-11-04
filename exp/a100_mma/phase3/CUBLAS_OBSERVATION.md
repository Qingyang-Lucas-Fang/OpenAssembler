# What the retained cuBLAS evidence establishes

The FP16 invocation launches **grid (256,4,2), or 2048 CTAs**. That is an
observed fact. Two K partitions are a strong interpretation of that grid,
not a directly queried cuBLAS algorithm attribute. The complete invocation
trace rules out a separate reduction kernel for these calls; it does not
identify the reduction mechanism inside the GEMM kernel.

The raw evidence is `cublas-dispatch.json` (all launches) and
`cublas-observation.json` (one selected GEMM's counters and resources).
`cublas_summary.py` parses both into `cublas-derived-summary.json`, retaining
input hashes, full launch order, API flags, resources and derived facts.
No vendor cubin or SASS body is imported into OpenAssembler or retained as
its implementation.

## Exact operation and library

Both observations use 4096×4096×4096, A[M,K] and B[N,K] row-major, computing
C=A×Bᵀ with FP32 output. `cublas_target.py` calls unbatched `cublasGemmEx`
six times, with alpha=1 and beta=0. Its column-major API representation uses
transa=T, transb=N and swaps the row-major operands; the output contract
matches the OpenAssembler tests.

The logger verifies CUDA_R_16F(2) inputs for FP16, CUDA_R_16BF(14) for BF16,
CUDA_R_32F(0) output, CUBLAS_COMPUTE_32F(68), and
CUBLAS_GEMM_DEFAULT_TENSOR_OP(99). `cublasGetVersion_v2` reports 120803 and
the library logger says **cuBLAS12.8.3**. The environment enumerates toolkit
libraries named `libcublas.so.12.8.3.14` and
`libcublasLt.so.12.8.3.14`; enumeration alone does not establish which
absolute shared-object path the dynamic loader selected. Nsight Compute is
2025.1.0.0; cuobjdump/nvdisasm are CUDA12.8 V12.8.55.

The output target checks finiteness and prints its norm. It is a dispatch
observation, not an additional full-output numerical comparison. The
separate full-output benchmark harness supplies that numerical evidence.

## Complete launch sequence

There are fourteen recorded GPU launches per dtype, below the profiler's
25-launch capture limit:

| IDs | Meaning | Count |
|---|---|---:|
| 0–1 | PyTorch random input generation | 2 |
| 2–7 | One selected GEMM kernel for each of six cuBLAS calls | 6 |
| 8–11 | Elementwise operations from the finite-output check | 4 |
| 12 | Boolean finite-output reduction | 1 |
| 13 | Output norm reduction | 1 |

The kernels after ID7 are diagnostics in `cublas_target.py`, not a GEMM
reduction phase. There are no auxiliary GPU launches between the six GEMMs.
The six API records and six GEMM launches agree for both dtypes.

## Launch and resource facts

| Property | FP16 | BF16 |
|---|---:|---:|
| Selected family | CUTLASS-named s16816,128×128,32×5 | Ampere s16816,128×256,64×3 |
| Grid | (256,4,2) | (32,16,1) |
| Total CTAs | 2048 | 512 |
| Threads/CTA | 128 | 256 |
| Registers/thread | 220 | 238 |
| Allocated registers/thread | 224 | 240 |
| Dynamic shared bytes/CTA | 81,920 | 98,304 |
| Static shared bytes/CTA | 0 | 49,152 |
| Operand shared bytes/CTA | 81,920 | 147,456 |
| Driver shared bytes/CTA | 1,024 | 1,024 |
| Register and shared-memory CTA residency limit | 2 | 1 |

The names' tile/stage interpretation has an independent resource check:
2×5×32×(128+128)=81,920 bytes for FP16, and
2×3×64×(128+256)=147,456 for BF16. This is consistent with five stages of
FP16 operands and three larger BF16 stages. It does not expose every
internal allocation or establish the private kernel's source template.

The FP16 name indicates 128×128 output tiles. A square4096 output needs
1024 such spatial tiles. Grid.x×grid.y is exactly1024, while grid.z=2
doubles the launch to2048. The call is unbatched. Public CUTLASS identity
swizzling maps logical K tiles to grid.z, and its width-eight spatial map
would map 32×32 tiles to256×4. This makes split-K=2 a strong explanation.
The private cuBLAS specialization's algorithm attributes and coordinate
mapping were not queried, so the report retains that conclusion as an
inference. A semaphore-based serial reduction, atomic accumulation or
another in-kernel scheme cannot be distinguished by launch count alone.
[Public CUTLASS swizzle implementation](https://github.com/NVIDIA/cutlass/blob/418ba45013423731194f8c487348a5da2511298a/include/cutlass/gemm/threadblock/threadblock_swizzle.h).

For BF16, 4096/128×4096/256=512 spatial tiles matches the observed512 CTAs
with grid.z=1. There is no launch-level evidence for K partitioning there.

## Counters and derived instruction mix

| Metric | FP16 | BF16 |
|---|---:|---:|
| Tensor activity,% of sustained active peak | 86.285 | 90.268 |
| Active-warps occupancy,% | 12.117 | 12.489 |
| Short-scoreboard stall,% per active warp | 1.863 | 0.255 |
| Long-scoreboard stall,% per active warp | 5.974 | 1.492 |
| Wait stall,% per active warp | 35.081 | 38.307 |
| Barrier stall,% per active warp | 7.041 | 6.160 |
| Shared-load bank-conflict count | 380,988 | 0 |
| Profiled kernel duration,µs | 544.256 | 648.768 |

These are Nsight replay profiles with cache control and unlocked clocks,
not replacement benchmark medians. Occupancy near12.5% coexists with high
tensor activity: increasing occupancy alone is not the optimization target.
Likewise, FP16's nonzero bank count does not prevent high tensor activity;
bank-conflict counts need throughput and stall context.

The retained static summaries show x4 LDSM in both vendor families, global
to shared LDGSTS, explicit completion/barrier instructions and vectorized
stores. FP16 also has substantial shared-memory epilogue traffic. Those
counts cover disassembled control paths, not dynamic execution counts.
The source view reports zero `.reuse` annotations; that presentation does
not establish that the vendor never uses reuse control bits.

OpenAssembler can independently explore these mechanisms. Its measured B
pairing improvement already moves toward x4 fragment loading. Larger tiles,
deeper stages and split-K require controlled tests including resource costs,
reduction precision and the complete operation's time. The observations do
not justify assuming cuBLAS uses a naive schedule, copying its kernel, or
claiming training acceleration from an isolated GEMM result.

Reproduce derived analysis without a GPU:

```sh
.venv/bin/python exp/a100_mma/phase3/cublas_summary.py
```
