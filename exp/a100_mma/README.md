# Isolated A100 MMA compiler experiment

Started 2026-11-03. Everything new in this experiment stays here. No production
package, specifications, CLI, or existing dirty files are edited. Python remains
the compiler implementation language.

**Current results and reproduction:** [REPORT.md](REPORT.md). The final six
FP16/BF16 projection workloads passed full-output checks on A100; the compiler
reaches approximately 58–78% of cuBLAS throughput in that series. No core changes
have been merged. Use final result records rather than the earlier debug runs.

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
