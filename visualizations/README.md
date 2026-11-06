# Architecture illustrations

The authoritative current architecture and implementation status are in
[STATUS.md](../docs/STATUS.md). The [research catalogue](../docs/research/README.md)
provides the evidence behind each optimization.

- `openassembler_pipeline.mmd` mirrors the Mermaid architecture in STATUS. It
  keeps native code generation separate from Inductor/Triton application capture.
- `openassembler_speedups.mmd` is the current optimization map, including explicit
  tuning limits and the separate low-rank research path.
- `nvidia_cuda_pipeline.mmd` is a background CUDA-toolchain illustration, not an
  OpenAssembler implementation or benchmark result.

The existing Manim `.py` scenes are historical explanatory assets. They are not
an authoritative current feature inventory, and their conceptual arrows do not
prove that a path is integrated. Rendering them requires optional Manim; it is
not needed for the compiler or documentation checks. The original diagrams and
README are preserved in the compressed documentation-history snapshot.
