# A100 MMA compiler experiment: measured progress, remaining gap

Latest repeated kernel improvements and cuBLAS/SASS research are in
[phase3/REPORT.md](phase3/REPORT.md). Follow-on hardware-counter experiments are in
[phase2/REPORT.md](phase2/REPORT.md); primary papers and hardware feasibility are
in [phase2/LITERATURE.md](phase2/LITERATURE.md). This file retains the first-round
baseline and its historical validation scope.

2026-11-03. All work is isolated under `exp/a100_mma/`. No production
source, specifications, existing dirty file, branch, or commit was changed.

## Result

An opt-in Python selector now compiles complete, specialized PTX GEMMs through
the existing OpenAssembler CLI/build/container machinery. It generates HMMA,
LDSM, async-copy instructions and scheduling controls with the repo encoder;
ptxas supplies the reference and ELF metadata container. The experimental
function bodies replace ptxas code. This is an OpenAssembler MMA backend
experiment, with explicit tensor layouts and dimensions, rather than a
replacement for the PyTorch/Inductor application compiler.

The final validation series reaches about 58–78% of cuBLAS throughput.
**cuBLAS-level performance has not been achieved.** No model training
tokens/sec or forward/backward-step speedup is claimed.

## Matched comparisons

Device: A100-SXM4-40GB, 108 SMs, driver 13030; PyTorch 2.7.1+cu128.
Both inputs are FP16 or BF16; all methods accumulate and output FP32.
`A` is M×K contiguous, `B` is N×K contiguous, and `C = A @ B.T`.
The cuBLAS baseline calls `cublasGemmEx` directly with FP32 output;
there is no timed output conversion or PyTorch dispatch.

| M × N × K | Input | OpenAssembler ms | ptxas, same PTX ms | cuBLAS ms | cuBLAS throughput fraction |
|---|---|---:|---:|---:|---:|
| 4096 × 4096 × 4096 | f16 | 0.950 | 0.720 | 0.633 | 66.6% |
| 4096 × 4096 × 4096 | bf16 | 0.815 | 0.627 | 0.545 | 66.9% |
| 4096 × 16384 × 4096 | f16 | 3.138 | 2.450 | 2.440 | 77.8% |
| 4096 × 16384 × 4096 | bf16 | 3.102 | 2.410 | 2.139 | 69.0% |
| 16384 × 4096 × 4096 | f16 | 3.694 | 3.324 | 2.312 | 62.6% |
| 16384 × 4096 × 4096 | bf16 | 3.634 | 3.293 | 2.092 | 57.6% |

Each entry uses a median of seven randomized method-order rounds, with 20
kernel nodes per CUDA graph replay. Warm-up and compilation are excluded.
Raw samples remain in each JSON. Clocks were not locked; container-to-container
timing variation is visible in the cuBLAS column. Compare methods within a run;
the fastest absolute time from another container is not a paired speedup.

The earliest full-output-checked 4096³ FP16 synchronous prototype took 2.910 ms
in `results/v2_4096_sync_base.json`. The final 0.950 ms FP16 result is roughly
3× faster than that prototype, with the clock caveat above. It does not imply
a 3× improvement over the production scalar compiler at identical precision.

A separate comparison-only Triton sweep reached 0.597 ms versus cuBLAS
0.579 ms in the first FP16 square run. The final BF16 sweep reached 0.597 ms
versus cuBLAS 0.545 ms. These establish a useful tiling/pipeline target; their
cubins are not OpenAssembler output.

## Changes established by the experiment

* `backend.py` adds strict m16n8k16 FP16/BF16 selection with four-register FP32
  accumulator groups and loop-carried liveness. Unsupported shapes, mixed
  types, layouts and predicated collective instructions raise errors.
* LDSM x4/x2 and transpose-x2 selection uses consecutive fragments and
  completion scoreboards. x2 encoding was discovered through the existing
  nvdisasm oracle, cached only here. Aligned address offsets were independently
  round-tripped before using them to reduce live address registers.
* The LDGSTS encoder extension handles both independently varying memory
  base registers. Operand bits were recovered with bit probes and checked
  against nvdisasm; see `ldgsts-encoding-evidence.json`.
* Async groups require `LDGDEPBAR` with W0 in the ptxas pattern. The first
  implementation omitted W0. The current scheduler reserves SB0 for copy
  groups, uses 1–5 for ordinary memory, retires IDs before reuse, and tracks
  multiple outstanding readers of an address.
* Generic scalar schedule shortening changed deferred BAR and DEPBAR issue
  gaps to S1. Large full-output tests failed. Preserving BAR≥6 and DEPBAR≥4,
  alongside corrected group tracking, repaired those variants. These are
  supported settings for this kernel family, not independently established
  universal minimum barrier latencies.
* Omitting LDSM read barriers without preserving address WAR dependencies
  caused large failures. The current alternative protects the address with
  the load’s write-completion scoreboard: a completed result implies the
  address was consumed. A regression check covers several loads sharing one
  address before it is overwritten.
* `pipeline.py` emits a prologue, two shared stages, overlapped next-stage
  copies, a single backedge, and a peeled final stage. No speculative tail
  reads are needed. The final tile is CTA 128×128×64 with four 64×64 warps,
  228 registers and 64 KiB opt-in dynamic shared memory.
* Compact shared addresses made a previously out-of-register 128×128×64
  candidate compile. Larger 128×256/256×128 CTA trials passed but were slower.
  Register-reuse hints did not establish a repeatable normalized speedup.

## Correctness and verification

Every final workload ran zero, small-integer, random-normal and scaled-normal
patterns with two seeds. Entire output tensors were checked for finite values
and compared with cuBLAS. Zero/integer comparisons were exact. Random/scaled
comparisons used relative L2 ≤ 2e-5. An independent CPU float64 oracle evaluates
32 sampled rows × 32 sampled columns of the actual quantized inputs; small
smoke matrices use the full oracle. Maximum final sampled relative L2 error
was 6.74e-06.
Output canaries checked writes beyond the output. This does not establish
general NaN/Inf/subnormal semantics, every finite dynamic range, all shapes,
or the absence of every possible input overread.

The actual `oa.py ... cnr ... --backend modal` path also passed through the
existing manifest/spec/driver runner, byte-exact against ptxas. See
`cli-final.log` and `build/cli-final/run.json`. The CLI check uses packed
FP16 ones; the separate GPU suite supplies broader independent arithmetic
checks. Seven experimental tests and 64 selected existing CLI/backend/runner
tests passed. The latter used a Git discovery ceiling for temporary fixtures
under `exp/`; otherwise a fake `.git` fixture discovered the enclosing repo.

The 128-operation HMMA chain sweep passed encoded gaps down to 2, but direct
result-to-STG.128 checks failed at gaps 2–14 and passed at 15+. Independent
accumulator probes also passed short encoded gaps. These are correctness
observations with interlocks, not measurements of execution latency or peak
issue throughput. The final kernel uses HMMA S8 and an explicit drain when
leaving a contiguous tensor sequence.

Failed experiments are retained in `results/` and `evidence-summary.json`;
they are excluded from performance claims. Many early small tests checked
only sampled output locations, so they do not have the same strength as the
final full-output suite. Do not promote those older schedules.

## Python and compile time

Python is the host compiler; the GPU executes the emitted cubin. Rewriting
the host in C++ would not itself accelerate an already-generated kernel.
However, compile-time profiling now identifies a real optimization target.
In the final series, warm generation spent about 0.62–0.69 s in register
allocation; complete CLI compilation was about 2.3–2.4 s. The profiled run
shows `ra.allocate` calling `_alloc` four times, each recomputing `_intervals`.
Caching invariant group/liveness analysis across allocation hint attempts
is a concrete Python optimization to test before considering another language.
Profiler overhead makes its absolute timings unsuitable for comparison with
the unprofiled stage records. See `compiler-profile.txt` and experiment.json
under the final build directories. No C++ rewrite was performed.

## Next research priorities

1. Profile the passing direct kernel’s tensor issue, operand collection,
   shared-bank traffic and scoreboard/barrier stalls with hardware counters.
   The remaining gap is measured; its causes need counters, not instruction
   count alone. Reduce fragment/address live ranges and test three/four stages
   while retaining the full-output checks and explicit synchronization model.
2. Strengthen the scheduler model with clocked HMMA/LDSM experiments and
   isolated synchronization sweeps. Preserve proven dependency edges when
   reordering; the old scalar model is insufficient for tensor pipelines.
3. Add transpose/layout and boundary handling for actual dX/dW shapes, fused
   epilogues, and a forward/backward integration hook. Measure a whole step
   before making any transformer-training acceleration claim.
4. Revisit AlphaTensor with compile-time factors and bounded live products
   after the direct schedule is competitive. Earlier runtime-coefficient
   Triton lowering is not an algorithm-only comparison. Keep low-rank
   transformation precision and output mode matched.

## Reproduction

From the repository root, with the existing CUDA wrapper and Modal account:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/a100_mma/final_workloads.py
PYTHONDONTWRITEBYTECODE=1 python3 -m modal run exp/a100_mma/modal_run.py --suite validation-suite.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/a100_mma/summarize.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/a100_mma/audit.py --check
```

`oa.py` is the isolated CLI adapter; it rejects source/output paths outside
this experiment. Use `--ssa-locals --interlocked-tensor --tensor-issue 8
--mined-scalar --ldsm-completion-wars` for the current schedule. `--list-schedule`
is an optional separately measured variant. Experimental schedules are
recorded in experiment.json even when the reused CLI opt-level says safe.

Final results include cubin/reference/PTX hashes and compiler flags. Source
snapshots for final builds are under `snapshots/`. Future GPU invocations also
archive binaries by hash and immutable run records under `binaries/` and
`runs/`; older records predate those additions. Generated build/ cubins are
ignored by the repository’s existing build/ rule and remain local artifacts.

NVIDIA primary reference: [Ampere tuning guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html)
for async copies, tensor-core modes and resource limits; a local retrieved
copy is `ampere-guide.html`. Numerical/scheduling conclusions above come from
the saved repository experiments, not from assuming undocumented latencies.
