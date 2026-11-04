# 04 — A100 Compute-Bound Kernel Research Agenda

Date: 2026-11-03. Scope: Ampere A100, transformer training GEMMs and their PTX/SASS/cubin execution path. This memo separates repo evidence, new A100 measurements, and hypotheses. All generated experiment artifacts live in this directory.

## Executive read

Historical snapshot: later work implemented the native SM80 tensor selector,
scheduler and GEMM documented in [TENSOR_GEMM.md](../TENSOR_GEMM.md), with
[phase-four promotion evidence](../../exp/a100_mma/phase4/REPORT.md). Statements
below about the missing MMA backend describe the November 3 baseline. The
prototype and negative results retain their original scope.

OpenAssembler has a real research asset in its A100 reverse-engineering corpus: it can emit and rewrite sm_80 SASS words, patch cubin text and ABI metadata, and run those cubins on an A100. The measured divider strength reduction is concrete. The current evidence does not yet establish a training-kernel speedup, and a tensor-core AlphaTensor prototype we measured is dramatically slower than cuBLAS. The bad result is informative: the current prototype pays large scalar transform, indexing, conversion, and synchronization costs and generates a poor tensor-core pipeline. It does not disprove low-rank algorithms when their transform is performed cheaply in-register/shared memory and their products use an optimized MMA tile.

The immediate strategic target should be a strong hand-authored A100 GEMM kernel with an explicit PTX-to-SASS-to-cubin path, then optimizer experiments whose hypotheses can be checked at the SASS and timing levels. A general SASS decompiler/lifter is valuable infrastructure, but it is not the near-term performance bottleneck. Python is entirely suitable for the host-side compiler, corpus mining, proof harness, and experimentation. It is not currently a reason to rewrite the project in C++; GPU time dominates these kernels and Python emits native GPU instructions. The right split is Python orchestration plus native/GPU implementation only where profiling identifies host overhead or where a code-generation abstraction gives leverage.

## New A100 tensor-core experiment

The research harness at [a100_mma_tensorcore.py](a100_mma_tensorcore.py) compares cuBLAS, a direct Triton GEMM, and one-kernel fused Strassen/AlphaTensor rank-7 schemes. It checks output against cuBLAS, randomizes algorithm order each round, stores generated PTX/SASS, and reports registers, shared memory, instruction counts, and numerical error. The data is in [a100-mma-results.json](a100-mma-results.json), with exact kernels in [a100-mma-artifacts/](a100-mma-artifacts/).

Device: NVIDIA A100-SXM4-40GB, sm_80, 108 SMs, CUDA 12.8, PyTorch 2.7.1, Triton 3.3.1. Timings below are median of three captured-graph rounds, ten replays per round. All methods use f16 or bf16 inputs and f32 output/accumulation; this is a GEMM microbenchmark, not an entire optimizer step.

| M×K×N | dtype | cuBLAS ms | Direct Triton ms | Strassen rank 7 ms | AlphaTensor rank 7 ms |
|---|---|---:|---:|---:|---:|
| 4096×4096×4096 | f16 | 0.511 | 1.325 | 11.611 | 13.041 |
| 4096×4096×4096 | bf16 | 0.511 | 1.317 | 11.772 | 13.163 |
| 4096×4096×16384 | f16 | 2.040 | 6.320 | 44.193 | 49.760 |
| 4096×4096×16384 | bf16 | 1.923 | 6.292 | 44.731 | 50.179 |
| 16384×4096×4096 | f16 | 1.963 | 9.047 | 47.301 | 54.055 |
| 16384×4096×4096 | bf16 | 1.928 | 9.205 | 47.487 | 54.505 |

The direct Triton kernel is 2.6–4.8× slower than cuBLAS, so it is not a fair implementation baseline for judging the algorithm. The current fused rank-7 kernels are 21–42× slower than cuBLAS and 8–15× slower than the direct Triton kernel. Their f16 relative L2 error is about 6.16e-4 for Strassen and 7.14e-4 for AlphaTensor; bf16 error is 4.93e-3 and 5.71e-3. These are consequences of forming block sums in f16/bf16 before the tensor-core products, while direct GEMM multiplies the original low-precision inputs. This precision asymmetry makes the current fast-matmul comparison invalid as an algorithm-only test.

Disassembly explains a major part of the gap. The direct Triton GEMM uses 63 registers, 12 KiB shared memory, `LDGSTS.E.BYPASS.128`, `LDSM`, and `HMMA.16816.F32`. Its body has only 218–221 non-NOP instructions through EXIT in the 4096² variants. The fused rank-7 kernel uses 205 registers for f16 and 209 for bf16, 8 KiB shared memory, 4,560–4,896 non-NOP instructions through EXIT, 56 HMMA operations, 1,120 FFMA operations, and hundreds of half-conversion or packed-data operations (`HADD2` for f16, `PRMT`/`SHF` patterns for bf16). It also performs 168 individual global loads, 128 block barriers, and seven slow-path calls in the FP16 path. It does not use `LDGSTS` or `LDSM`; loads and transforms are repeatedly reconstructed in registers with large live ranges. The high register count alone cuts active warps and makes latency hiding harder. A code-count ratio is not a cycle ratio, but here the instruction mix is direct evidence that our fused design is structurally wrong for an A100.

There is also a harness caveat: Triton 3.3.1 generated and ran the experimental kernels, but its `CompiledKernel` did not expose `n_regs`/spill attributes. The artifact records were corrected from `nvdisasm` and currently include an early instruction count; use the `.sectioninfo SHI_REGISTERS` field from saved SASS as the authoritative register count. The benchmark script now parses that entry metadata and counts instructions only through the entry's EXIT for future runs. The large run predates that accounting fix; the numbers above were recomputed offline from its saved SASS.

## What reverse engineering says about A100 opportunities

### Tensor cores and data movement

The repo already has ptxas-generated examples of `HMMA.16816.F32`, `HMMA.1688.F32.TF32`, bf16 HMMA, IMMA, DMMA, `LDSM`, and `LDGSTS` in `corpus/sm_80/sass/`. The encoder, emulator, cubin splicer, and ELF metadata writer are separate usable pieces. The architectural gap is the PTX-to-SASS backend: [openassembler/backend/ra.py](../../openassembler/backend/ra.py) has scalar arithmetic, memory, control flow, and constant-division selection but no `mma.sync`, warp-fragment, `ldmatrix`, or async pipeline selector. Therefore, tensor-core experiments currently need a ptxas/NVVM/Triton-generated metadata skeleton or an external assembly path; the encoder does not yet close an end-to-end MMA compiler path.

The A100 path should initially use `mma.sync.m16n8k16` f16/bf16 with f32 accumulation. TF32 (`m16n8k8`) is a useful accuracy/performance comparison, with `tf32x3` as a higher-accuracy alternative when available in the frontend. The measured HMMA emulator model is useful for debugging, but its special-value handling is explicitly a hypothesis; compiler validation should first use bounded finite inputs and compare on-device against cuBLAS, then expand NaN/Inf/subnormal testing separately.

The highest-leverage SASS-level target is a software-pipelined CTA mainloop: coalesced wide global loads or `cp.async`/`LDGSTS` to shared, multiple shared stages, asynchronous groups (`LDGDEPBAR`/`DEPBAR.LE`), `LDSM` fragment loads, and HMMA issue while later stages load. The checked-in direct GEMM disassembly already demonstrates the broad pattern, while the old `fastmm` GEMM has a single-buffer scalar FMA loop. However, `tools/fastmm_run.py` says the existing shared-memory GEMM implementation is 4% behind cuBLAS at n=8192, and fused kernels lose; a new low-level kernel must beat that as well as reproduce correctness.

Tile choices should be swept by measured resources, not intuition: CTA tile M/N, K stage, warps per CTA, per-thread accumulator count, shared-memory double/triple buffering, and occupancy. Large tiles improve data reuse and MMA issue density but can reduce resident CTAs. Smaller tiles create more CTAs but repeat loads and address work. Dump `nvdisasm -plr`, ptxas verbose resource usage, shared-memory bytes, register count, and local loads/stores for every candidate. The local assembly wrapper is under `env/cuda.sh`; ptxas has `-v`, `--register-usage-level`, `-maxrregcount`, and `-allow-expensive-optimizations`; `nvdisasm` provides `-hex`, `-json`, `-plr`, `-lrm`, `-bbcfg`, and `-cfg`; `cuobjdump` provides `-res-usage`, `-sass`, `-ptx`, `-elf`, and extraction tools.

### Recovered dependency behavior

The previous committed A100 latency record `corpus/sm_80/gpu_a100_40gb_latsweep.json` only included three predicate-guard sweeps. Its accompanying `hw_latency.json` therefore had `null` for all ALU RAW pairs. `machine_model.json` is mined from ptxas stall fields and should not be described as a hardware latency model. The repo code and status text had conflated this mining data with the A100 run.

I regenerated the existing `lat_sweep.py` probe family into a separate research root, ran 399 cubins × 2 trials on an A100, and saved corrected raw and summary records under [a100-latency/corpus/sm_80/](a100-latency/corpus/sm_80/). All 42 minimum-safe-stall cases were stable. The evidence supports:

| Producer → consumer | Minimum safe SASS stall |
|---|---:|
| ALU → ALU tested pairs | 4 cycles |
| FMA/IMAD → FMA/IMAD tested pairs | 4 cycles |
| ALU ↔ FMA cross-pipe tested pairs | 5 cycles |
| predicate written by ISETP/FSETP → guarded IADD3/IMAD/FADD/BRA path | 13 cycles |
| predicate used as SEL/FSEL input | 4 cycles |
| MOV immediate → IADD3 | 1 cycle |
| IADD3 → STG data dependency | 2 cycles |
| IMAD or FFMA → STG data dependency | 1 cycle |

These are minimum stalls in the tested encodings and operand positions, not universal issue-throughput or latency specifications. The scheduler may use 4/5 for those exact fixed-pipe classes; all untested opcode pairs need a conservative fallback. The experiment also demonstrates why many operations should not be modeled by one generic opcode latency: read positions (`FFMA.a/b/c`, `IMAD.a/b/c`) and store data paths can differ.

### Scalar ALU operations still matter around MMA

The existing proven integer division/remainder-by-constant lowering is a credible compiler win: it replaces ptxas 12.8's reciprocal-based constant division sequence (13–17 operations) with 2–3 multiply-high/shift operations, and the repo includes formal/exhaustive proofs. But for dense GEMM mainloops, the work is dominated by MMA and data movement; division matters mostly in address/shape/index code and reductions, not the multiply-heavy inner loop. More promising ALU work around training GEMMs includes eliminating repeated 64-bit pointer arithmetic, hoisting CTA-invariant offsets, fusing masks into address predicates, reusing packed/converted fragments, and removing control flow from the full-tile path.

The encoder corpus and prior SASS already show ptxas balancing operations across specialized issue pipes, using `.reuse` operand cache fields, scoreboard IDs for async loads, and `DEPBAR.LE` before shared loads. An aggressive compiler can search over semantically equivalent forms and schedules, but each mutation must be validated with four independent checks: decoder text, emulator/PTX semantics, nvdisasm acceptance, and real-A100 result. `tools/bitflip.py`, `tools/reencode_cubin.py`, `tools/encoder_eval.py`, and `tools/compile_check.py` provide a base for this loop.

## Research priority order

1. **Make the current direct tensor-core candidate competitive.** Implement a dedicated A100 GEMM tile with `cp.async`/LDGSTS staging, `ldmatrix`, warp-level HMMA, loop pipelining, and fused epilogue. Compare with cuBLAS at the same precision and output mode. The current direct Triton candidate is only a correctness and disassembly baseline.
2. **Use ptxas as a component-level black-box oracle.** Generate a matrix sweep over PTX shapes, tile sizes, layouts, vectorized loads, pipeline stages, unroll factors, and `-maxrregcount`/register-usage-level settings. Archive PTX, cubin, `cuobjdump -res-usage`, `nvdisasm -hex -plr`, and timings. Diff ptxas scheduling/tiling patterns; do not infer an optimum from instruction count alone.
3. **Repair the scheduler evidence and cost model.** The corrected 42-case latency sweep is in research data, but the checked-in scheduler still uses hard-coded pipe classes, assumes 5 cycles for cross-type FMA cases without direct pair measurements, and does not model HMMA dependencies or tensor-pipe issue spacing. Expand the sweep to HMMA accumulator dependencies, LDSM→HMMA, async-copy completion, register reuse, and independent issue throughput. Separate RAW latency from initiation interval, scoreboard wait, and barrier completion.
4. **Optimize AlphaTensor only after transforms are cheap.** Build a fused low-rank kernel where each tile's transformed operands stay in registers or shared memory, but cap live accumulators and schedule one product at a time if that is faster. Consider hybrid precision: form A/B linear combinations in f32 then round once to f16/bf16, or use TF32 to avoid repeated low-precision conversion. Compare against a direct tensor-core kernel with identical stage count, tile, output accumulation, and numerical policy. The first experiment shows that lowering transformed matrices directly through generic scalar indexing is a dead end.
5. **Exploit workload structure from training.** Profile forward QKV/MLP projections, dX and dW GEMMs, attention QKᵀ and PV, and optimizer/reduction kernels separately. Weight-gradient shapes are transposed/reduction-heavy; fused bias/residual/activation epilogues often matter more than changing the multiplication count. Do not extrapolate a GEMM-only win to tokens/sec until a full forward/backward step is measured.
6. **Keep Python for the research compiler until profiling says otherwise.** Current compilation is local and its expensive steps include Python parsing, data-flow analysis, scheduling, and template search; for repeated JIT workloads those costs may matter, but the repo has no compile-time benchmark showing Python is the bottleneck. First instrument stage time and amortization across kernel-cache hits. If template search or corpus parsing dominates, vectorize hot analysis or move just those passes to Rust/C++; a full C++ rewrite would not speed up the generated GPU kernel.

## Limits and next evidence

- The new AlphaTensor/Strassen kernel is a prototype using Triton as an expedient tensor-core code generator, not an OpenAssembler-produced cubin. It proves the repo's current factorization can be applied around tensor-core dot products, but the SASS shows poor lowering. No end-to-end OpenAssembler MMA compiler has been implemented.
- The small 256³ run was only a functional smoke test. Large timings were graph-captured and shuffled; one warm-up round was high for 4096² and median-of-three was reported. The direct candidate is much slower than cuBLAS and not suitable for claiming compiler performance.
- Large comparisons are f16/bf16-only with f32 output. They do not cover FP32 TF32/tf32x3 or the exact autocast policy of a production transformer.
- The direct Triton output was compared to PyTorch/cuBLAS using identical inputs and matched exactly at the output tensor level in this configuration. This is a local comparison, not bitwise equivalence to every legal FP16/BF16 accumulation contract.
- New A100 latency data uses two trials of 32 lanes per cubin. Thresholds matched cleanly, but repeat on a second A100 and repeat at 4–8 trials before updating shared scheduler assumptions.

## Primary references

- NVIDIA, [PTX ISA 9.4: `mma`](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma), [`ldmatrix`](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-ldmatrix), and [`cp.async`/wait groups](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async).
- NVIDIA, [CUTLASS: Efficient GEMM in CUDA](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html), describing CTA/warp/instruction tiling, reuse, and double buffering.
- NVIDIA, [CUDA C++ Programming Guide: asynchronous data copies](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-data-copies).
- Repository A100 evidence: [corrected latency summary](a100-latency/corpus/sm_80/latency-summary.json), [raw GPU comparison](a100-latency/corpus/sm_80/gpu_a100_40gb_latsweep_corrected.json), and [MMA run](a100-mma-results.json).
