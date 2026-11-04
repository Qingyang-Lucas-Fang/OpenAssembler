# Isolated A100 MMA compiler experiment

Started 2026-11-03. Phases one through three kept compiler experiments here.
Phase four, explicitly authorized by the user, promotes the validated SM80
selector and GEMM plan into the standard CLI. Python remains the implementation
language. See [phase4/REPORT.md](phase4/REPORT.md) and
[the public API](../../docs/TENSOR_GEMM.md).

**Measured research results:** [phase3/REPORT.md](phase3/REPORT.md). A fixed
paired-B-fragment/vector-epilogue/stage-specialized schedule reduces kernel time
by16.1–26.7% across six FP16/BF16 projections, reaching80–99% of matched default
cuBLAS throughput in the final run, with separate component confirmations.
All timed outputs are checked in full. The [first-round report](REPORT.md) retains
its historical baseline. No core changes have been merged. Use final result
records rather than earlier debug runs. Native CLI promotion and further tested
candidates are in [phase4/REPORT.md](phase4/REPORT.md).

**Follow-on results:** [phase2/REPORT.md](phase2/REPORT.md) records real Nsight
counters, clocked HMMA/LDSM probes, isolated synchronization mutations,
two/three/four-stage pipelines, shared-memory layouts, paired pointer-update
improvements, and a padded fused-bias autograd hook. The associated
[literature survey](phase2/LITERATURE.md) distinguishes A100 features from hardware
changes proposed in papers or introduced on Hopper. Root README/REPRODUCING edits
were explicitly authorized for this phase; production compiler code remains intact.

## Order and promotion rule

1. Trace the existing CLI/build/run ownership and record the repository state.
2. Close the missing PTX MMA/fragment selection boundary experimentally; compile
   complete cubins through the existing CLI machinery, then validate on an A100.
3. Build a stronger direct tensor-core performance baseline with matched precision;
   use it to identify tiling and pipeline targets for the experimental backend.
4. Measure HMMA dependencies/issue behavior and async-copy completion before
   relaxing schedules. Only then revisit cheap, compile-time low-rank transforms.

Functional compiler support alone is not a performance win. Any proposed core
promotion needs correctness evidence, retained unsupported-input errors, regression
checks, and workload timings against both ptxas and cuBLAS. No automatic merge.

## CLI ownership map

`openassembler/__main__.py` calls `cli.main`. No arguments enters the UI REPL;
one-shot commands share the same parser and dispatch. The shell banner is a UI
concern, separate from compilation.

* Standalone `compile`: `project.classify/walk` selects sources;
  `frontend.to_ptx_module` invokes pyfront or nvcc as required; `build.generate`
  selects instructions, eliminates dead code, coalesces copies, allocates groups,
  optionally list-schedules, applies scoreboard/stall controls, and encodes words.
  `ptxas` builds the reference/container. `splice_kernels` replaces our function
  bodies and fixes per-symbol register metadata. `nvdisasm` checks the final cubin.
  Atomic publication writes PTX, cubin, SASS and manifest under `--out-dir`.
* `run`: validates manifest paths and artifacts, selects sources/kernels, creates
  buffer/launch jobs from the auto-harness or a spec, and invokes local/SSH/Modal
  driver execution. The standalone comparison is byte-exact; tensor experiments
  also need a numerical oracle and explicit layouts rather than random-f32 auto
  buffers. `cnr` runs only successfully built kernels, preserving failure status.
* Configured PyTorch application: `openassembler.toml` changes dispatch to
  `torchfront`. It instruments model forwards and captures/replays
  Dynamo/AOTAutograd/Inductor/Triton output. This path does not yet lower tensor
  kernels through OpenAssembler's selector. cuBLAS extern calls are distinct from
  captured Triton cubins. Treat its provenance separately.
* `check`, `bench`, `fastmm`, `gpu run`, and `corpus` invoke research tools with
  their own output roots. Do not use these default destinations in this experiment.

The backend already models contiguous 2/4-register groups and CFG liveness.
Missing MMA selection is not solved by adding a single opcode: all accumulator
registers must survive backedges, operand groups must retain alignment, LDSM needs
scoreboards, and HMMA requires independently validated timing rules. Encoder
template discovery can write `specs/sm_80/synth_fieldmaps.json`; this experiment
redirects discovery caches and oracle scratch space here.

## Corrections to the first research memo

The earlier large Triton comparison used **f16/bf16 output**, with f32
accumulation, not f32 output. Its rank-7 coefficient arrays were runtime data,
which caused expensive generic lowering; it is not evidence against an optimized
AlphaTensor implementation. Generic SASS CALLs should not be labeled numerical
slow paths without following their CFG. Stalls across an intervening instruction
are not isolated producer-to-store latency measurements. These distinctions apply
to all conclusions in this experiment.
