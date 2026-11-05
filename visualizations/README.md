# OpenAssembler visualizations

This directory contains the requested compiler visualizations and keeps all new
files scoped to `visualizations/`.

## Mermaid diagrams

- `openassembler_pipeline.mmd` is the source-to-execution OpenAssembler path:
  Python/CUDA C++/PTX/PyTorch input, PTX convergence, scalar and tensor backends,
  SM80 encoding, ptxas ELF container splicing, verification, and local/SSH/Modal
  execution on an A100.
- `nvidia_cuda_pipeline.mmd` is the NVIDIA reference path, including `nvcc`,
  NVVM/LLVM, PTX, `ptxas`, SASS/cubin/fatbin, driver JIT, the CUDA runtime, the
  SM pipeline, and the separate cuBLAS/cuBLASLt library dispatch path.
- `openassembler_speedups.mmd` expands the optimization mechanisms and uses
  Mermaid links back to the relevant OpenAssembler pipeline stages.

## Manim scenes

- `openassembler_architecture.py`
  - `OpenAssemblerArchitecture`: top-level architecture for new engineers.
  - `OpenAssemblerPipelineDetail`: detailed scalar/tensor backend and packaging.
- `openassembler_speedups.py`
  - `SpeedupOverview`
  - `ConstantDivision`
  - `LoopUnrollAndSchedule`
  - `RegisterAllocation`
  - `TensorStagePair`
  - `AlphaTensorFastMM`

When Manim is available, render a scene with:

```sh
manim -pqh visualizations/openassembler_architecture.py OpenAssemblerArchitecture
manim -pqh visualizations/openassembler_speedups.py TensorStagePair
```

The diagrams preserve the repository's explicit evidence boundaries: constant
division is SMT-proven; tensor stage-pair improvements are component-kernel
results; the NN study still reports numerical mismatches and slower kernels;
and unsupported PTX is rejected or marked not verified instead of being hidden.
