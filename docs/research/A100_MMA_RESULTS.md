# A100 MMA and Training-Kernel Research Ledger

Last consolidated: 2026-11-05. Target: NVIDIA A100-SXM4-40GB (SM80) unless a
row explicitly says otherwise. The experiments compare a mix of compiler
kernels, ptxas, cuBLAS/cuBLASLt, and non-OpenAssembler prototypes. They do not
all use the same numerical contract. Each claim below states its actual
baseline and scope.

## Reading the evidence

- **Kernel timing:** same-worker CUDA graph timing, shuffled method order,
  warmup before timing, output validation before a candidate enters timing.
- **Correctness:** full outputs are called out explicitly. Small-integer exact
  checks, floating FP32-reference tolerances, canaries and raw-bit equality are
  not interchangeable.
- **Profiler timing:** Nsight counter replay flushes/perturbs caches and is not
  substituted for graph medians.
- **Source audit:** inspection of a public source tree is not a reproduced
  performance result and not evidence that OpenAssembler implements it.
- **Historical run:** a later run may supersede a code path, tool environment or
  measurement. Superseded rows are retained as research history, not current
  claims.

## Executive findings

1. The strongest direct SM80 GEMM result is the fixed stage-pair, paired-B
   fragment, vector-store kernel promoted into the regular compiler. Against the
   earlier direct experimental schedule it reduced time16.13–26.69% over six
   FP16/BF16 square, MLP and tall workloads. It reached79.76–98.91% of matched
   default cuBLAS in the phase-three final run. Later cuBLASLt comparisons
   exposed a remaining gap. No whole-training-step result exists.
2. A four-wide software pipeline and larger CTA/warp tiles were not general
   wins. Three/four shared stages, software producer warps, tested asymmetric
   swizzles and bounded fragment windows did not beat the selected direct
   schedule consistently.
3. The two-slot mbarrier experiment is a positive **one-CTA overlap
   microbenchmark** (1.370× steady versus a matched CTA-barrier control). It is
   not dense GEMM, does not imply an A100 training gain and is not lowered by
   the production compiler.
4. Morton CTA traversal was retested after correcting its rectangular-grid
   decode. The current mapping reduces time7.34% FP16 and6.54% BF16 versus the
   current m_fast compiler kernel at tall 16384×4096×4096. It remains slower
   than ptxas. Earlier Morton numbers used an older mapping and are superseded.
5. Fused FP32 bias+residual(+ReLU) at 4096³ reduces the complete OA GEMM plus
   OA pointwise time4.95–5.34%. It is shape-specific, ptxas fused remains
   faster, and the operation does not include backward dX/dW or optimizer work.
6. Low-rank multiplication count is not enough. Existing AlphaTensor/Strassen
   tests either used runtime factors, mismatched precision/output modes or
   tiles near launch overhead. None establishes a competitive A100 transformer
   kernel.
7. CUDA-L2 cache persistence tests were neutral or slower. The source audit is
   useful for schedule and resource ideas, not as a matched speed comparison.
8. A100 does not provide Hopper TMA or thread-block clusters. A100 has no FP8
   Tensor Core mode comparable to H100's FP8 path and no DPX datapath. Software
   cannot turn CTAs into a hardware cluster or guarantee their co-location.

## Workload and numerical contracts

The production tensor GEMM computes `C[M,N] = A[M,K] @ B[N,K].T`, with A and B
contiguous in K, FP16/BF16 input, FP32 accumulation and FP32 output. Current
production constraints are full tiles and multiples of128 in M/N/K; Morton
requires M/N multiples of1024. The fused epilogue adds FP32 bias `[N]` and
residual `[M,N]`. It applies `(GEMM + bias) + residual`, then optional finite
ReLU. The operation order and precision are part of the contract.

Some earlier Triton experiments used FP16/BF16 output. Some low-rank
experiments formed transformed operands in low precision. Those runs must not
be compared as algorithm-only alternatives to FP32-output direct GEMM. The
phase-two integration hook was padded and used an explicit baseline; its result
is an integration test, not a production-autocast or speedup result.

## Initial Triton low-rank MMA experiment

This was an early, separate one-kernel Strassen/AlphaTensor prototype emitted
through Triton. It must not be merged with the later FP32 multi-launch
AlphaTensor results or the native OpenAssembler GEMM. The device was an
A100-SXM4-40GB (SM80, 108 SMs), CUDA 12.8, PyTorch 2.7.1 and Triton 3.3.1.
Measurements used three captured-graph rounds and ten replays. Inputs were
FP16/BF16, with FP32 accumulation/output in the direct and library paths; the
low-rank transforms were rounded back to the input type before products.

| MxKxN | Type | cuBLAS ms | Direct Triton ms | Strassen rank-7 ms | AlphaTensor rank-7 ms |
|---|---|---:|---:|---:|---:|
| 4096x4096x4096 | FP16 | 0.511 | 1.325 | 11.611 | 13.041 |
| 4096x4096x4096 | BF16 | 0.511 | 1.317 | 11.772 | 13.163 |
| 4096x4096x16384 | FP16 | 2.040 | 6.320 | 44.193 | 49.760 |
| 4096x4096x16384 | BF16 | 1.923 | 6.292 | 44.731 | 50.179 |
| 16384x4096x4096 | FP16 | 1.963 | 9.047 | 47.301 | 54.055 |
| 16384x4096x4096 | BF16 | 1.928 | 9.205 | 47.487 | 54.505 |

The direct Triton schedule was 2.6-4.8x slower than cuBLAS, while fused
rank-7 was 21-42x slower than cuBLAS and 8-15x slower than direct Triton.
Rank-7 FP16 relative L2 errors were about 6.16e-4 (Strassen) and 7.14e-4
(AlphaTensor); BF16 errors were 4.93e-3 and 5.71e-3. The transform precision
differs from direct GEMM, so these are not algorithm-only comparisons.

The saved SASS shows why this lowering is a dead end in its current form. The
direct kernel used 63 registers, 12 KiB shared memory, `LDGSTS`, `LDSM` and
`HMMA.16816.F32`, with 218-221 non-NOP instructions through `EXIT` for the
square cases. Fused rank-7 used 205 FP16 or 209 BF16 registers, 8 KiB shared
memory and 4,560-4,896 non-NOP instructions; it contained 56 HMMA, 1,120 FFMA,
168 individual global loads, 128 CTA barriers and seven slow-path calls in
the FP16 variant, but no `LDGSTS` or `LDSM`. FP16 transforms use `HADD2`;
BF16 transforms use `PRMT`/`SHF` sequences. Large scalar live ranges and
reconstructed loads dominate the design. Instruction count alone is not a
cycle model, but the low-rank lowering is plainly not a competitive A100
mainloop.

Triton's wrapper did not expose trustworthy register/spill attributes for
this run. Registers were recovered from `.sectioninfo SHI_REGISTERS` in the
saved SASS; instruction counts were recomputed through the entry's `EXIT`.
The first accounting pass included unrelated code and is not authoritative.
The complete JSON, metadata and PTX/SASS corpus are retained in
[`initial-triton-lowrank-results.json`](../evidence/a100-mma/initial-triton-lowrank-results.json),
[`initial-triton-lowrank-artifacts-metadata.json`](../evidence/a100-mma/initial-triton-lowrank-artifacts-metadata.json)
and the compressed artifact archive beside them.

## Scalar neural-kernel baseline

The initial NN study is intentionally separated from tensor-core GEMM work.
It ran on an A100-SXM4-80GB using 128x128 elementwise/reduction buffers and
small 32x32x64 matrix cases, with three shuffled rounds and twenty timing
repetitions. These scalar kernels do not exercise Tensor Cores and are not
representative transformer-training throughput. Exact output was required;
timing for a mismatching kernel is diagnostic only.

| Kernel | Bit-exact | OpenAssembler ms | ptxas ms | ptxas/OA |
|---|---:|---:|---:|---:|
| ReLU | yes | 0.0102 | 0.0113 | 1.10x |
| GELU | no | 0.0113 | 0.0102 | 0.91x |
| Softmax | yes | 0.1024 | 0.0809 | 0.79x |
| LayerNorm | no | 0.0922 | 0.0717 | 0.78x |
| Adam update | no | 0.0123 | 0.0102 | 0.83x |
| Cross entropy | no | 0.0584 | 0.0410 | 0.70x |
| Naive matmul | no | 0.0287 | 0.0143 | 0.50x |
| Attention scores | no | 0.0246 | 0.0195 | 0.79x |

The recorded differences were small FP32 bit mismatches, often one to three
ULPs. `FMUL` plus `FADD` instead of ptxas's contracted `FFMA` is a plausible
cause for some examples such as GELU, not a complete diagnosis of all
mismatches. No approximate tolerance was applied after observing the output.
The useful follow-up is an explicit fast-math/contraction policy plus
separately reported exact and tolerance-based workloads; silently changing
rounding is not acceptable. Raw samples and provenance are in
[`nn-study-a100-20261102.json`](../evidence/nn-study-a100-20261102.json).

## Fusion boundaries and storage costs

The relevant compilation unit is a bounded producer/consumer graph with an
explicit arithmetic, rounding, forward-save/backward-use, layout, storage and
launch contract. Eliminating a global-memory intermediate is only useful when
the fused partition retains a competitive MMA pipeline. Multiple kernels can
be faster than an oversized fusion group.

The CUDA-L2 source audit sampled five released A100 kernels at commit
`dbe017722194bb33bafadfbcbb4a65ab6df95dc3`:

| Source MxNxK | CTA BMxBNxBK | Stages | Accumulator | Output | Padding |
|---|---|---:|---|---|---|
| 4096x4096x4096 | 160x128x32 | 3 | FP32 | FP16 | M padded to 4160 |
| 16384x4096x4096 | 128x128x32 | 4 | FP32 | FP16 | none |
| 4096x16384x4096 | 128x128x32 | 4 | FP32 | FP16 | none |
| 64x4096x64 | 64x128x32 | 3 | FP32 | FP16 | none |
| 64x64x64 | 16x16x16 per warp | N/A | FP16 WMMA | FP16 | none |

Four CuTe samples convert FP32 accumulator fragments to FP16 before shared
memory/global storage. The small WMMA example accumulates directly in FP16.
The utility packs/transposes B before the kernel; this preprocessing is
outside the published kernel timing. The 160-row square tile requires the
benchmark's 4160-row padding, adding 1.5625% work; directly using the kernel
on a 4096-row allocation would violate its full-tile memory assumptions.
The sampled cuBLAS baseline also stores FP16 C. These differences disqualify
its timings as a matched comparison with OpenAssembler's FP32-output GEMM.

The large samples use 16-byte `cp.async`, x4 LDSM and swizzled shared layouts.
They load the next K fragment before issuing the current fragment's MMA,
using alternating fragment slots; global copies begin early in the
subfragment sequence. This is a useful next-fragment overlap pattern, not
simply a higher stage count. Their CTA mapping advances N within a strip,
then M, then the strip index; the host selects strip widths by shape. It is
coarse software grouping and gives no physical SM placement guarantee. The
epilogue narrows to FP16 and rearranges values through shared memory before
vector stores, a different tradeoff from OpenAssembler's FP32 register-pair
stores.

The paper describes a training/reward loop for general kernels and HGEMM,
including measured speed, numerical error, code length and profiling context.
The public repository instead provides static shape-specific kernels and
build/correctness/benchmark utilities; the source audit did not reproduce its
training procedure or inspect all 1000 files. Public source samples are not
evidence of OpenAssembler results. Pinned source hashes and URLs remain in
[`phase5/cuda-l2-source-manifest.json`](../evidence/a100-mma/phase5/cuda-l2-source-manifest.json).

For a 4096x4096 FP32 intermediate, a fused producer can ideally avoid one
128 MiB write plus one 128 MiB reread. A second 128x128 FP32 accumulator tile
can cost another 128 registers per thread; shared buffering can require an
additional 64 KiB; retaining an extra 2 MiB row in an MLP-like shape also
affects liveness. These are upper-bound resource estimates, not measured
speedups. They explain why fusion should be searched at each graph boundary
and why MMA resource/scheduling costs belong in the objective.

cuBLASLt already exposes bias and activation epilogues; residual can be
represented as C with beta=1 in applicable modes. Therefore, vendor libraries
do not universally materialize GEMM output before pointwise work. OpenAssembler
can still target graph boundaries outside the chosen GEMM interface, but
comparisons must use the same epilogue semantics.

## Results by phase

### Initial direct MMA implementation

The first A100 tensor prototype established end-to-end PTX emission, HMMA,
LDSM, asynchronous copies and output checks. Its final run was about58–78% of
cuBLAS throughput, with the largest shape-dependent deficit on tall GEMMs.
The early 4096³ FP16 synchronous prototype took2.910ms; later stage-specialized
work made it much faster. This was a baseline, not evidence of cuBLAS parity.

An early Triton low-rank comparison looked very unfavorable: its direct Triton
kernel was2.6–4.8× slower than cuBLAS and fused rank-7 kernels21–42× slower.
The comparison was not algorithm-matched: the candidates used FP16/BF16 output
and runtime coefficients, while the later direct contract accumulates and
outputs FP32. Transform rounding also differed. It documents a poor generic
lowering path, not a disproof of compile-time low-rank algorithms.

### Phase two: counters, synchronization and pipeline variants

**Pointer update and ordering.** Updating loop-carried 64-bit pointers in
allocated homes reduced the selected kernel from1,898 to1,812 static non-NOP
instructions. Six same-worker full-output cases improved0.9–5.4%, depending
on shape/dtype. CTA output-column ordering plus reuse hints then improved the
paired sample0.3–1.8%; this small effect was treated cautiously.

**Counters.** In the original direct kernel, Nsight reported tensor-pipe
activity59.43% versus78.26% for ptxas on identical PTX, active warps11.91%
versus12.12%, short scoreboard13.79% versus1.15%, long scoreboard11.94%
versus12.05%, wait stalls43.89% versus37.17%, and90.85M versus69.62M executed
instructions. Direct register-bank/operand-collector causes were not
established. A later profile found a bounded 64×32 warp tile raised active
warps and tensor activity but was still slower than the selected 64×64 large
GEMM tile. Occupancy alone did not select the winner.

**Clocked tensor probes.** Exact-SASS HMMA dependency probes showed an
approximately8-cycle-per-op elapsed slope for the tested sequence; the
LDSM→LOP3→IADD3 sequence showed about41 cycles per iteration. These include
the generated sequence and are not pure instruction latency, initiation
interval or full-SM throughput. The first clock harness accidentally measured
generic-lowering fragment copies and was rejected.

**Synchronization sweeps.** BAR mutation variants passed; the experiment did
not establish that BAR stall6 is necessary. DEPBAR thresholds1/2/3 failed
full-output checks;4/6 passed. Reusing a scoreboard ID without explicit
retirement passed a limited probe but gave no demonstrated speed gain. Removing
address WAR protection failed earlier checks; the production path preserves
that edge. Synchronization-field validity is not equivalent to a legal async
pipeline or a performance win.

**Stages and shapes.** For the tested 64-wide K tile, two stages beat three/four
stages (about1.003ms versus1.178/1.180ms in that run). Larger 128×256 and
256×128 CTA candidates passed but were slower. Additional staging cost more
shared memory and control work than it hid for this schedule.

**Shared layout.** 7/7 swizzle beat asymmetric7/3 and3/7. The unswizzled case
reported234.9M shared-load conflicts and2.255ms versus0.170M conflicts and
0.807ms for7/7 in the profiled/run comparison. These are software address
layouts over A100 banks; physical bank hardware was not reconfigured.

**Producer/consumer warps.** The tested role-split kernels passed four A100
large-case correctness checks but measured1.019–1.943ms and lost to the unified
warp schedule. A544-thread variant failed launch resources. This rejects those
implementations, not every producer/consumer design.

**Training hook.** The padded/autograd hook passed FP16/BF16 forward/backward
and three-step state checks. End-to-end wall time was about0.98ms versus0.68ms
for the explicit reference. This establishes a working integration hook, not
acceleration or a production-autocast comparison.

**Sparse/hardware feasibility.** Sparse `mma.sp` capability was assembled and
checked with ptxas-oracle code only. There is no OpenAssembler sparse selector
or training speed claim. The same audit concluded that FP8 Tensor Core modes,
TMA, DPX and thread-block clusters were unavailable on A100; these cannot be
enabled by adding a PTX modifier. The complete phase-two literature and
capability classifications are summarized in retained evidence.

### Phase three: direct schedule and vendor inspection

The winning direct schedule combined paired B-fragment `LDSM.x4` loads,
register-pair reuse, a vectorized FP32 output epilogue, compact/folded shared
addresses and two-tile stage specialization with a peeled drain. It preserved
the full MMA and async-copy dependency graph and passed complete output checks.

Against the prior direct schedule, six final measurements improved:

| M×N×K | Type | Prior direct ms | Final schedule ms | Time reduction | cuBLAS ms |
|---|---|---:|---:|---:|---:|
| 4096×4096×4096 | FP16 | 0.811 | 0.642 | 20.84% | 0.595 |
| 4096×16384×4096 | FP16 | 2.968 | 2.461 | 17.09% | 2.434 |
| 16384×4096×4096 | FP16 | 3.657 | 2.727 | 25.43% | 2.351 |
| 4096×4096×4096 | BF16 | 0.783 | 0.628 | 19.79% | 0.556 |
| 4096×16384×4096 | BF16 | 2.909 | 2.439 | 16.13% | 2.136 |
| 16384×4096×4096 | BF16 | 3.635 | 2.665 | 26.69% | 2.125 |

The ptxas-reference code compiled from the same generated PTX and matched
default cuBLAS are separate baselines. In the final run throughput was
79.76–98.91% of matched default cuBLAS. Clock variation across containers
matters; the comparisons use within-worker paired rounds.

The selected profile moved tensor-pipe activity63.51→75.51%, short scoreboard
14.05→11.12%, and shared-load conflicts170,940→57,238, with active warps near
12%. Profiling replays differ from CUDA graph timing and cannot causally assign
the total gain to a counter. A cuBLAS/cuBLASLt source/SASS observation found
vendor activity around86–90% in the inspected cases. It did not extract or
reuse a cuBLAS kernel.

**Rejected/limited follow-ups:** scalar epilogue order, fragment windows,
pre-addresses, operand order and many scheduler variants were tested. A
64×32 warp tile reduced registers and increased active warps but remained
behind the best direct tile. Three/four/five-stage schedules and producer
warps were not winners. Compile-time rank-7 versus rank-8 at96×96×32 used more
registers/instructions for rank7 and ran near the launch floor; it cannot
predict large GEMM. Low-rank transformations still need matched precision,
bounded live products and an equally optimized mainloop.

**cuBLASLt.** The public heuristic sweep returned10 unique FP16 and13 BF16
algorithms in the queried setup. The selected zero-workspace candidates ran
about0.576ms FP16 and0.564ms BF16 versus0.672/0.655ms for the then-current OA
kernel. It was not an exhaustive algorithm enumeration, and these results
must not be mixed with default GemmEx numbers or other clocks.

### Phase four: production CLI promotion and rejected controls

The measured native selector/GEMM was promoted into the ordinary CLI with
strict unsupported-form rejection, provenance checks, full-tile contracts,
manifests and CLI regression tests. Six production cubins were checked against
the phase-three streams and passed A100 output checks. The CLI result is the
same GEMM, not a training graph.

cp.async L2-prefetch modifiers produced small inconsistent effects; the
candidate modes were not promoted. B-fragment windows2/4 and bounded prefetch
were slower in all seven rounds across tested workloads. These variants passed
correctness, but no repeatable speed gain was established.

### Phase five: cache, locality, fusion and barriers

**CUDA-L2 source audit.** The pinned public project provides an RL/tuning
system and shape-specific CUDA/CuTe kernels. Its inspected “FP32 MMA” samples
convert accumulators to FP16 storage; one small WMMA example accumulates in
FP16. Padding/input packing affects some benchmark timings. Therefore its
numbers are not matched to OA FP32-output RCR GEMM. Transferable ideas include
fragment ping-pong, staggered A/B copies, shape-aware tile/padding search,
shared epilogues and coarse CTA order. No kernel was copied.

**L2 set-aside.** Same schedule with graph-node cache policy readback and full
outputs: 224 checks across square/tall FP16/BF16 and policy variants. Full L2
reservations degraded square cases about8% and tall cases substantially (some
over30%). Refined8MiB tall settings were effectively neutral/noisy;16MiB and
larger regressed. The tested cache policies were not promoted. `cp.async`
prefetch hints and L2 set-aside are distinct controls.

**CTA order.** Initial tests compared n-fast, grouped-M, Morton8 and Hilbert8.
Physical HMMA/LDSM/LDGSTS words and controls were unchanged; only logical CTA
assignment differed. Initial and independent runs gave different curve winners,
so choosing the largest from one sweep was explicitly rejected. The first
production Morton implementation flattened rectangular CTA grids into groups
of64; the November5 workspace corrected it to decode each aligned physical
8×8 CTA group separately. See Phase six for current-source validation.

**Fused epilogues.** A bounded batch4 candidate prefetched four independent
residual pairs and reused vector bias loads. The 1491-instruction GEMM
prologue/mainloop prefix was identical; registers stayed216. All full-output,
canary, integer, raw-bit and dReLU checks passed. In the phase-five square
confirmation, fusion improved OA separate by4.42–5.43%; smoke regressed about
20%, and tested MLP was0.3–1.24% slower. ptxas's fused result remained faster.
Phase six rebuilt this feature through production and retested it.

**Mbarrier.** The single-slot SM80 producer/consumer protocol was correct but
slower than CTA barriers in steady mode (0.553–0.557ms versus0.355ms). A
two-slot protocol passed phase-varying full outputs, count/parity accounting
and canaries: 0.332ms versus0.455ms steady (1.370×), and0.989ms versus1.183ms
under artificial stress (1.197×). This is one CTA, one producer warp, one
consumer warp and a16×16×16 product per phase. It is a positive overlap
microbenchmark, not a dense GEMM result; production lowering remains absent.

The barrier work had two repaired failures: first, operand copies overlapped
barrier objects; second, the B-fragment lane map reused A's map. Those failures
were retained in the historical experiment and are not architecture limits.

### Phase six and current-source Morton retest

Phase-six production candidates used normal CLI builds. The initial
2026-11-04 Morton results are superseded because the workspace subsequently
changed Morton decoding. The current 2026-11-05 mapping passed all full-output,
canary, zero, integer and normal checks across the two tall Morton kernels and
four square fused kernels.

| Current candidate | Time ms | Matched OA control ms | Time reduction | ptxas control ms |
|---|---:|---:|---:|---:|
| Tall FP16 Morton8 | 2.528 | m_fast 2.728 | 7.34% | 2.369 |
| Tall BF16 Morton8 | 2.478 | m_fast 2.652 | 6.54% | 2.327 |
| Square FP16 bias+residual | 0.743 | separate OA 0.781 | 4.95% | ptxas fused 0.639; separate 0.742 |
| Square BF16 bias+residual | 0.729 | separate OA 0.770 | 5.26% | ptxas fused 0.627; separate 0.733 |
| Square FP16 bias+residual+ReLU | 0.741 | separate OA 0.783 | 5.34% | ptxas fused 0.630; separate 0.746 |
| Square BF16 bias+residual+ReLU | 0.729 | separate OA 0.768 | 5.16% | ptxas fused 0.624; separate 0.734 |

Results: [current validation JSON](../evidence/a100-mma/current-validation/validation_20261105T074142.327676Z.json).
The superseded November 4 result is retained in the evidence index. The current Morton candidate remains slower than ptxas by
about6.5–7%; fused OA squares are about5% faster than OA separate launches and
roughly even with ptxas's separate operation, while the ptxas fused kernel is
faster. Both choices are opt-in. No complete training step, backward dX/dW,
bias-gradient reduction, optimizer, throughput or tokens/sec measurement was
performed.

## Hardware-idea feasibility on A100

| Idea | A100 status | Evidence and decision |
|---|---|---|
| Tensor Core utilization | Supported and measured | Counter activity rose with the selected direct schedule, but ptxas still had higher utilization in inspected profiles. Continue counter-driven mainloop work. |
| Dynamic shared allocation/layout | Supported within SM80 limits | Shared swizzle7/7 won; tile/stage changes have occupancy and bank-traffic costs. No arbitrary runtime bank remapping was found. |
| Register-file “cache” | Registers are explicit; no user switchable RF cache | Bounded fragment liveness can change occupancy; measured window gains did not beat best direct tile overall. |
| Software producer/consumer warps | Supported by ordinary CTA synchronization | Tested designs lost to unified warp scheduling; one exceeded CTA thread resources. |
| mbarrier/cp.async | SM80 supports documented mbarrier and cp.async forms | Correct protocol tested; single slot loses, two slot wins only in tiny microbenchmark. Do not lower a raw arrival word without full protocol. |
| TMA and thread-block clusters | Not available on A100 | Hopper-era mechanisms; software cannot create hardware cluster semantics or guarantee CTA co-location. |
| FP8, DPX | Not A100 Tensor Core/DPX paths | Do not present as A100 optimization choices. Sparse 2:4 is available, but the test was ptxas oracle only. |
| L2 persistence / CUDA-L2 | API available; benefit workload-dependent | Tested reservation policies were neutral or regressed. Source-repo results were not precision-matched. |
| Hilbert/Morton CTA order | Software scheduling hint only | Current Morton wins in tested tall shape. It does not guarantee SM adjacency and is not universal. |
| Strassen/AlphaTensor | Algorithmically valid, system cost decisive | Runtime-factor prototype invalid as algorithm-only comparison; compile-time microtest near launch floor; no large matched win reproduced. |
| Whole transformer fusion | Possible but resource-/semantics-bound | One GEMM epilogue measured. No full forward/backward training step. |

## Next research priorities

1. **Close the measured direct-kernel gap with counters.** Profile tensor-pipe
   issue, operand collection, shared-bank traffic, short/long scoreboards and
   barrier waits for the same direct OpenAssembler, ptxas and cuBLAS cases.
   Counters should explain the remaining gap; static instruction count cannot.
2. **Search layouts and registers jointly.** Current swizzle tests mostly held
   the register fragment organization fixed. Search shared swizzle, fragment
   ownership/load order, reuse hints and live ranges together. Preserve
   established async-copy, LDSM address-WAR and HMMA dependencies. Test three
   and four stages only with complete outputs, canaries, resource reports and
   matched timing; prior standalone deeper-stage candidates lost.
3. **Improve the tensor scheduler model.** Use exact SASS clocked probes to
   separate HMMA accumulator latency, HMMA issue spacing, LDSM-to-HMMA delay,
   asynchronous-copy visibility and barrier completion. Sweep synchronization
   in isolation before recombining. Scalar minimum-stall tables are not a
   tensor pipeline model.
4. **Cover actual training shapes and measure the whole step.** Add transpose,
   tails and boundaries for dX/dW, then measure backward GEMMs, bias-gradient
   reduction, fused activations, attention/normalization boundaries and
   optimizer work. Include packing, layout conversion, launches and required
   saved activations. No kernel-level result can stand in for a forward/backward
   training-step or tokens-per-second result.
5. **Revisit low-rank products only after the direct schedule is competitive.**
   Use compile-time factors, bounded live products and common transformed-tile
   staging. Match input, accumulation and output precision with the direct
   control. SubCuber's graph/fusion and temporary-placement cost model is
   relevant prior art; rank alone is not a cost model.
6. **Automate equivalent SASS sequence search.** Vary address formation,
   fragment load order, `.reuse`, scheduling controls and legal instruction
   forms under semantic checks, valid disassembly, full hardware correctness
   and paired timing. Never relax a proven dependency edge for a faster-looking
   instruction listing.

### Compiler implementation language

Python is not on the GPU instruction path: it emits/loads cubins and schedules
host work; the GPU executes native SASS. A C++ rewrite will not make HMMA
execute faster. Compile-time latency can matter for frequent JIT or autotuning,
but no current end-to-end compile profile demonstrates that Python dominates
user-visible runtime. First measure parsing, graph construction, allocation,
scheduling, encoding, ptxas invocation, cache hits and total application
startup separately. If a specific pass or search loop is material, optimize
that pass or move only the profiled hot loop to a compiled implementation.
Keep code-generation results and performance comparisons identical before and
after such a change.

## Retired workspaces

The original scripts, per-variant build outputs, logs and phase-specific reports
were cleaned after extracting results. Durable experiment outputs and selected
hashes live in [the evidence index](../evidence/a100-mma/README.md); primary
papers and hardware documentation are in [REFERENCES.md](REFERENCES.md).
