# Checkpoint: 2026-11-03 — OpenAssembler 0.2.0 implementation

This worktree contains the uncommitted 0.2.0 implementation. The original mid-session
checkpoint below the first heading was superseded by the changes listed here.

## Completed in this pass

- Linux/macOS support only; Windows is documented as unsupported.
- `compile SRC -o NAME --out-dir DIR` writes `DIR/cubin/NAME.cubin`,
  `DIR/sass/NAME.sass`, `DIR/manifest.json`, PTX, and a ptxas reference cubin.
- Directory compilation discovers PTX, CUDA C/C++, and statically recognized Python
  `@kernel` files; scripts and host-only files are skipped.
- `run` consumes a manifest; `cnr` compiles and runs the kernels built in that call.
  `--spec`, `--dry-run`, Modal/SSH/local backends, timestamped run reports, validation,
  and cleanup guidance are documented.
- Backend arithmetic coverage includes float division, sqrt, common casts, register
  integer division/remainder, approximate math fragments, and pointer-increment loops.
  The fragments are generated from recorded CUDA 12.8 ptxas listings and reallocated
  through the OpenAssembler IR.
- Cross-warp `bar.sync 0` is modeled by both the SASS emulator and PTX interpreter.
- CLI/banner/version/docs identify OpenAssembler 0.2.0 as a Python/CUDA C/C++/PTX to
  SASS + cubin compiler.
- 370 local pytest tests pass in the CUDA-toolkit environment.
- Modal checks on November 2, 2026: vector add; 62 arithmetic fragment pipeline with
  zeroed and adversarial input passes; dot rows/grid-stride SAXPY/naive matmul/nested
  loop pipeline; and a 64-thread shared-memory barrier all matched ptxas.
- `docs/NN_OPTIMIZATION_STUDY.md` records the pre-optimization NN/transformer baseline.
  Aggressive fusion, software pipelining, and tiling remain a proposed next phase;
  they were not silently implemented after the study.

## Known explicit limits

- Encoder target is sm_80. Some non-default float rounding/encoding forms and
  advanced PTX families remain unsupported.
- The generic `--verify` harness cannot infer arbitrary application buffer shapes.
- NN study bitwise mismatches on GELU, LayerNorm, Adam, cross-entropy, matmul, and
  attention identify numerical/optimization work; the report keeps them visible.
