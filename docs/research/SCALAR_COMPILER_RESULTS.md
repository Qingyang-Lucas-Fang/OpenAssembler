# Scalar Compiler and Reverse-Engineering Results

This ledger preserves the measured scalar-backend and cubin findings that are
separate from the A100 tensor-core work in
[A100_MMA_RESULTS.md](A100_MMA_RESULTS.md). Results are historical snapshots;
run the current tests before making claims about a new checkout or toolkit.

## A100 scalar microbenchmarks

These results used 4096 CTAs of 32 threads, 4096 loop iterations, and the best
median of three shuffled rounds on an A100. All 49 compared cubins were
bit-exact. `ptxas/opt` greater than one means the optimized OpenAssembler build
was faster for this microbenchmark.

| Kernel | ptxas ms | safe ms | mined ms | sched ms | opt ms | ptxas/opt |
|---|---:|---:|---:|---:|---:|---:|
| `chain.ffma` | 0.089 | 0.242 | 0.208 | 0.212 | 0.097 | 0.92x |
| `chain.imad_lop3` | 0.129 | 0.286 | 0.246 | 0.224 | 0.135 | 0.95x |
| `ilp4.ffma` | 0.257 | 0.386 | 0.368 | 0.328 | 0.254 | 1.01x |
| `ilp4.imad` | 0.257 | 0.357 | 0.334 | 0.315 | 0.250 | 1.03x |
| `loop.div7` | 0.629 | 0.687 | 0.646 | 0.566 | 0.464 | 1.36x |
| `loop.rem1000` | 0.373 | 0.457 | 0.421 | 0.388 | 0.312 | 1.19x |
| `loop.smem` | 0.748 | 0.828 | 0.736 | 0.725 | 0.713 | 1.05x |

`safe` to `opt` changes scheduling and unrolling, not instructions per loop
iteration. Constant division/remainder and loops with enough independent work
benefit. ptxas remains 5-8% faster on the dependent arithmetic chains because
it overlaps loop-counter and branch work with the chain; the OpenAssembler
unrolled path pays a predicate wait at the loop boundary. These are selected
kernel results, not a compiler-wide speed claim.

## Hardware findings that affect code generation

- Writing SASS destination register `Rd` faults unless the cubin declares at
  least `Rd + 3` registers. The count appears in both `EIATTR_REGCOUNT` and
  `.text.<kernel>` ELF `sh_info` bits 24-31. A 77-variant hardware sweep
  established the rule; a separate 15-variant sweep verified text growth,
  updated metadata and rewritten words.
- For tested fixed-latency pairs, minimum safe RAW stalls were four cycles
  within the measured FMA or ALU pipe classes and five across those classes.
  Tested predicate definition to guarded operation/branch needed 13 cycles;
  predicate to `SEL` needed four. These are operand/encoding-specific
  thresholds, not universal instruction latency. The full corrected sweep is
  indexed in [A100 latency evidence](../evidence/a100-latency/README.md).
- A hidden `ISETP` field survived disassembly: bits 32-39 retained an immediate
  fragment even though `nvdisasm` did not print it. Reusing the stale value
  changed loop behavior. Encoder minimization now preserves hidden bits using
  the majority value from real ptxas words sharing the opcode.
- PTX `@p` writes are partial definitions: the prior destination value must
  survive when the guard is false. CFG liveness and fixed homes for guarded
  writes prevent accidental fresh-register definitions.
- Uniform datapath `UIADD3`/related instructions execute once per warp, not
  once per lane. The emulator originally repeated uniform updates per thread.
- `BAR.SYNC` must match participating threads; one `bar.sync 1,64` experiment
  deadlocked. Hardware probes also found/fixed 128-bit register-read
  truncation, `RSQ64H` underflow, misaligned vector accesses, shared-memory
  races/uninitialized reads and an aliased unwritten register pair.
- Timing in fixed job order was biased by GPU ramp-up (identical cubins
  recorded 0.125 versus 0.098 ms). Current timing utilities shuffle order and
  keep paired samples/rounds.

The fitted tensor arithmetic emulator reproduces the recorded finite A100
HMMA/IMMA/DMMA/LDSM/LDGSTS cases, including 512/512 lanes per modeled family.
That dataset does not establish universal NaN, infinity, subnormal or rounding
behavior. Its measured finite-input rule aligns products at nominal exponents,
uses a bounded accumulator window and rounds at the end; broader adversarial
hardware validation remains required. See [reverse-engineering methods](REVERSE_ENGINEERING.md).

Other scalar observations retained from the historical probe suite include:
F64 NaN payload preservation in selected casts/arithmetic, `RSQ(-0) = -inf`,
raw-bit `FSEL` behavior, `BRX` table-entry targeting through constant memory,
and legal scheduling control requirements for stalls 12-15. These findings
are implemented or guarded in the emulator/compiler and are covered by the
focused tests available in the current checkout.

## Compiler pipeline and cost model

The scalar path selects PTX to virtual 32-bit registers, removes dead code,
coalesces and folds constants, allocates consecutive aligned register groups,
then lowers, schedules, encodes and splices the selected text into a
ptxas-generated ELF skeleton. Branching and guarded code use CFG dataflow;
registers with repeated definitions get stable homes. Optional optimization
passes use a basic-block list scheduler and four-way counted-loop unrolling.
Memory source-read dependencies are retained to prevent WAR hazards when
registers are reused. The measured-latency scheduler uses 4/5-cycle tested
classes and a 13-cycle predicate guard delay, with conservative fallback for
other cases.

The schedule uses a variable-latency write scoreboard and memory-read
scoreboard. Unrolling can move the exit test before the body when the body does
not read the loop counter. This reduces repeated counter work but lengthens
some live ranges and does not reproduce ptxas's chain/branch overlap. The
latency measurements do not model HMMA issue spacing or the tensor-pipeline
completion behavior; see the next priorities in the A100 ledger.

## AlphaTensor and Strassen: FP32 multi-launch study

The separate fast-matrix-multiplication study imported 97 factorization
schemes, all verified exactly over the rationals. Its A100-SXM4-40GB FP32
output study at n=8192 used three repetitions in two shuffled rounds. Time in
milliseconds:

| Plan | Launches | OA compiled | ptxas compiled | cuBLAS ratio | Relative error |
|---|---:|---:|---:|---:|---:|
| cuBLAS SGEMM | 1 | 57.6 | - | 1.000x | 1.6e-6 |
| Direct `sgemm128` | 1 | 60.0 | 62.1 | 0.960x | 1.6e-6 |
| Strassen 2x2x2 rank 7 | 21 | 58.4 | 60.6* | 0.986x | 2.3e-6 |
| AlphaTensor 2x2x2 rank 7 | 19 | 58.3 | 60.6 | 0.988x | 2.7e-6 |
| Strassen squared, rank 49 | 163 | 69.5 | 71.8 | 0.829x | 3.2e-6 |
| AlphaTensor V100 4x4x4, rank 49 | 174 | 71.6 | 74.0 | 0.804x | 5.7e-6 |
| Fused Strassen 2x2x2 | 8 | 72.2 | - | 0.798x | 2.3e-6 |
| Fused AlphaTensor 2x2x2 | 8 | 76.5 | 68.0 | 0.753x | 2.7e-6 |

*The Strassen ptxas value is from the prior run because the later run exhausted
its budget. One rank-7 level saves 12.5% of block multiplications but improves
only 2.7% over the OpenAssembler direct path and remains 1.2-1.4% behind
cuBLAS. At n=2048, cuBLAS took 0.98 ms while Strassen took 1.9 ms. The 4x4x4
products are 2048x2048 blocks with 256 CTAs (about 2.4 waves on 108 SMs), plus
114-125 addition launches. The original fused kernels also lose, and ptxas
beats OpenAssembler on the high-register fused PTX. This is evidence that
launch count, tile occupancy, transformations and resource footprint dominate
rank reduction alone, not evidence against low-rank algorithms in a matched
native tensor pipeline.

## Historical broad compiler checks

Earlier probe snapshots reported 110/110 OpenAssembler cubins bit-exact on an
A100; 532/541 emulator comparisons were bit-exact or within the PTX
approximation contract. The remaining nine were nondeterministic identifiers
or high-magnitude sine/cosine values for which the PTX specification did not
give a useful error bound. Tensor-core and async instruction emulator tests
were separate from that scalar count. A held-out encoding-control comparison
matched 89.2% of mined stall fields exactly, which was not sufficient to treat
the mined table as hardware latency. Later random rewrite checks disassembled
1239/1239 rewritten words as intended. Historical test counts (372, 486,
493) describe their specific repository/toolkit snapshots and should not be
quoted as the current suite result.

The current full suite and current compiler regressions are reproduced using
the commands in [REPRODUCING.md](../../REPRODUCING.md). The tensor-core results,
including the precision-mismatched Triton prototype and scalar neural baseline,
are recorded separately in [A100_MMA_RESULTS.md](A100_MMA_RESULTS.md).
