# A100 hardware and MMA research survey

Retrieved 2026-11-03. Discovery used arXiv, Crossref metadata and author/vendor
repositories. The web search service returned 503; direct primary-source retrieval
worked. Downloaded responses and discovery queries are under `sources/` and the
`*-sources.json` files. A title match alone was not treated as technical evidence.

## Papers that change the experiment plan

| Work | Evidence read | Consequence for OpenAssembler |
|---|---|---|
| Sun et al., [Dissecting Tensor Cores via Microbenchmarks: Latency, Throughput and Numeric Behaviors](https://arxiv.org/abs/2206.02874), TPDS 2023 | Author abstract, primary code link; paper metadata | Separate MMA latency, steady-state throughput and numerics. Use `mma`, inspect SASS, and avoid conflating encoded gaps with elapsed cycles. |
| Abdelkhalik et al., [Demystifying the Nvidia Ampere Architecture through Microbenchmarking and Instruction-level Analysis](https://arxiv.org/abs/2208.11174), 2022 preprint | Author abstract | Clocked single-warp instruction probes complement the GEMM counters. WMMA results cannot simply be substituted for our exact HMMA sequence. |
| Valpey et al., [An SMT Formalization of Mixed-Precision Matrix Multiplication: Modeling Three Generations of Tensor Cores](https://arxiv.org/html/2502.15999), 2025 | Full HTML, especially §4–5; [primary model repository](https://github.com/pyxis-roc/tensor_core_semantics) | The fitted repo model is not a universal numerical contract. Alignment truncation differs from global IEEE RTZ; Ampere preserves an additional shifted bit. Added a paper-motivated A100 discriminant. |
| Jangda, [Compiling Strassen-like Matrix Multiplication Algorithms to Fast CUDA Kernels](https://doi.org/10.1145/3808267), PACMPL 2026, SubCuber | Publisher-supplied abstract via Crossref; [Microsoft source](https://github.com/microsoft/subcuber) at `c7b6fd8c72db4fc83982b26ea6e80e47567d674f`: DSL, schedule enumeration, SM80 mainloop and epilogue. Full publisher PDF was inaccessible (403). | Closest prior art. Search dependencies, shared transforms, fusion partitions and hardware schedules together. Its abstract reports up to 12%/22% improvements for one/two recursion levels on A100/H200; these are author claims, not reproduced here. Do not frame the work as an unexplored opposition between smart algorithms and a naive NVIDIA implementation. |
| Huang et al., [Implementing Strassen's Algorithm with CUTLASS on NVIDIA Volta GPUs](https://arxiv.org/html/1808.07984), 2018 | Full HTML and abstract | Fuse input transforms into GEMM staging, consume products into output reconstruction, and account for workspace/atomics. Its V100 results do not establish the same gain at A100 tensor-core throughput. |
| Dao, [FlashAttention-2](https://arxiv.org/abs/2307.08691), 2023 | Author abstract | For A100 training, less scalar work and better warp/block partitioning can be more valuable than changing the bilinear rank. Attention is not an interchangeable GEMM-only benchmark. |
| Shah et al., [FlashAttention-3](https://arxiv.org/abs/2407.08608), 2024 | Author abstract | Asynchronous overlap and precision policies are useful design ideas; TMA, WGMMA and native FP8 performance depend on Hopper hardware. An A100 port must substitute cp.async/SM80 MMA and remeasure. |
| [ThunderKittens](https://arxiv.org/abs/2410.20399), 2024 | Author abstract | Tile abstractions and producer/consumer pipelines offer a useful model for compiler structure. Reported performance across kernels/hardware is not evidence for our A100 compiler. |
| Mishra et al., [Accelerating Sparse Deep Neural Networks](https://arxiv.org/abs/2104.08378), 2021 | Author abstract | Native A100 2:4 sparsity is real. Requires actual structured zeros and metadata; pruning changes the training problem. Sparse throughput is not a free dense speedup. |
| Kwak et al., [INTERPRET: Inter-Warp Register Reuse for GPU Tensor Core](https://doi.org/10.1109/PACT58117.2023.00033), PACT 2023 | Publisher abstract metadata, retrieved through OpenAlex; full paper unavailable | This proposes a modified microarchitecture with inter-warp register renaming/preloading. Its reported average improvement cannot be enabled in existing A100 hardware by SASS. Compiler-controlled operand `.reuse` hints and intra-warp lifetime reduction are implementable substitutes with different scope. |
| [AlphaTensor](https://www.nature.com/articles/s41586-022-05172-4), Nature 2022 | Existing primary factorization/code import, [official explanation](https://deepmind.google/discover/blog/discovering-novel-algorithms-with-alphatensor/) | Exact bilinear rank identities remain useful. Factor choice must include transform cost, quantization, live products and target tile occupancy. The binary-arithmetic rank result is not a real FP16 GEMM identity. |
| [AlphaEvolve](https://arxiv.org/abs/2506.13131), 2025; [official account](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) | Author abstract and official account | Its 48-product 4×4 result is for complex-valued multiplication. It must not be advertised as an immediately deployable rank-48 real FP16 tensor-core kernel. The useful transferable idea is verified schedule/algorithm search. |

The pinned [CUTLASS SM80 multistage implementation](https://github.com/NVIDIA/cutlass/blob/v3.8.0/include/cutlass/gemm/threadblock/mma_multistage.h)
is primary implementation evidence for keeping copy groups, shared stages and
fragment buffering separate. We read it as prior art; OpenAssembler's compiler
remains Python and its generated code remains independently measured.

## Evaluation of proposed hardware directions

| Idea | What existing A100 can do | Decision / measured evidence |
|---|---|---|
| Maximize tensor-core utilization | Change tile/warp geometry, instruction mix, dependencies, reuse and data supply | Highest priority. Nsight measured 59.4% tensor-pipe activity for our original passing kernel versus 78.3% for ptxas on the same PTX. This metric is not model FLOPs utilization. |
| Dynamic shared-memory allocation | Choose per-launch allocation/carveout and stage count; reuse shared storage across phases | Implemented 2/3/4 stages and 64/96/128 KiB opt-in cases. Extra stages passed but lost at 4096³. Hardware cannot dynamically move unused memory between concurrently resident CTAs by compiler fiat. |
| Asymmetric memory banking | Choose different software layouts/swizzles for A and B over the fixed physical banks | Tested A/B masks 7/7, 7/3, 3/7, 0/0. 7/7 won; 0/0 was about 2.8× slower. Physical bank width/count are not changed. |
| Register-file caching | Operand reuse flags, shorter live fragments, bank-aware allocation, shared/register double buffering | Fragment windowing crossed a 132→126 register threshold, and counters confirmed roughly doubled active warps. Blindly deleting read dependencies failed. Cross-warp register aliasing from INTERPRET would require new hardware. |
| Deep warp-level MMA pipelines | cp.async copy groups + LDSM scoreboards + SM80 HMMA with explicit drains | Implemented and clocked. More stages and shorter source stall fields do not automatically improve throughput. Exact probes show an 8-cycle/op slope for both one dependent and eight independent accumulator streams in the tested one-warp sequence. |
| Multiple precision/accumulation paths | FP16/BF16/TF32 inputs with FP32 accumulation, FP16 output modes, split products/error-correction sequences | Feasible, but a different numerical contract. Additional products/casts are measured work; no hidden higher-throughput FP8 path exists. Formal-model paper motivates adversarial tests before claiming accuracy. |
| TMA-like software engine | Coalesced cp.async, precomputed/incremental addresses, producer scheduling and layout conversion | Feasible substitute for selected functions; still consumes ordinary instructions and lacks a Hopper TMA unit. Pointer-update lowering is being measured. |
| Enhanced sparsity | Native 2:4 `mma.sp` and explicit compressed fragments/metadata | Three ptxas-oracle sparse-MMA capability probes passed on A100. No OpenAssembler sparse compiler, arbitrary sparse GEMM speedup, or training accuracy claim yet. |
| FP8 | Store/quantize FP8 and explicitly dequantize to FP16/BF16 or use separately designed integer products | No native A100 FP8 MMA. Emulation cannot create FP8 tensor throughput; quantization/dequantization costs and numerical impact must be included. |
| Async DPX | Software integer min/max/add chains | DPX is a Hopper dynamic-programming feature, not an asynchronous dense-MMA engine. Software emulation does not accelerate dense transformer GEMM by itself. |
| Virtual clusters / grouped SMs | Group tile order for L2 locality, persistent/cooperative launches with supported synchronization | Tested grouped CTA order 2/4/8/16 with full outputs; no gain for this square GEMM. A100 blocks cannot form a hardware thread-block cluster or access another SM's shared memory. Cross-CTA spin barriers can deadlock without guaranteed residency. |

NVIDIA primary sources: [Ampere architecture](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/),
[Hopper architecture](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/),
and the [Ampere tuning guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html).
Software can choose how to use the available datapaths; it cannot add execution
units, remote shared-memory fabric or instruction semantics absent from the chip.

## Concrete ideas arising from the evidence

1. Prefer a register budget that crosses an occupancy boundary over arbitrary
   minimization. A rolling B-fragment window can free enough registers for an
   additional resident block but also increases load-use waits. Use both counters
   and paired GPU timing to choose the window.
2. Optimize address recurrences directly in virtual IR: two carry-linked integer
   operations can update a 64-bit loop-carried pointer in place. Test carry and
   wrap behavior independently before running projection matrices.
3. Replace a broad stall heuristic with instruction-family evidence. A one-field
   synchronization mutation sweep identified DEPBAR≥4 for the tested family;
   BAR alone did not need the same conservative issue gap. Keep completion
   semantics separate from issue gaps and encoded interlocks.
4. Make CTA layout selection conditional on shape and measured L2/DRAM behavior.
   Grouped ordering is worthwhile to test, but the negative result here prevents
   calling software grouping a substitute for Hopper clusters.
5. For low-rank GEMM, build a compile-time transform DAG, common-subexpression
   sharing and a bounded product schedule. `alphatensor-constant-plans.json`
   verifies three existing real-arithmetic schemes and emits no device coefficient
   loads. It is a planning artifact, not a faster kernel. SubCuber is the relevant
   implementation comparison when this phase begins.

The phase-1 README's universal RTZ wording was too strong. A paper-motivated
discriminant, `32 - 2**-40`, produces FP32 `32` on A100 in both our generated cubin
and ptxas's cubin; global IEEE round-toward-zero would produce the predecessor
of 32. This supports the distinction between alignment truncation and global
rounding, while leaving the detailed model's general validity open.
