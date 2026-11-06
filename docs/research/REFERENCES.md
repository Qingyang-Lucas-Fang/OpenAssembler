# Research References

## GPU instruction sets, tools and APIs

- NVIDIA, [Ampere GPU Architecture Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html).
- NVIDIA, [CUDA C++ Programming Guide: L2 Cache Control](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#l2-cache-set-aside-for-persisting-accesses).
- NVIDIA, [Parallel Thread Execution ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html), including `cp.async`, `mbarrier`, `mma.sync`, and memory consistency.
- NVIDIA, [cuBLAS documentation](https://docs.nvidia.com/cuda/cublas/index.html), including cuBLASLt matmul epilogues and heuristic selection.
- NVIDIA, [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).
- NVIDIA, CUDA 12.8 `ptxas`, `nvdisasm`, `cuobjdump`, CUDA Driver API and installed headers. Versions used in each raw record are retained in its provenance.

## Reverse engineering and compiler systems

- CuAssembler, [source repository](https://github.com/cloudcores/CuAssembler), for public Volta-era SASS assembly/control-field work.
- NVIDIA, [CUTLASS efficient GEMM documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html), for producer/consumer pipelines, shared layouts and epilogue structure.
- NVIDIA, [CUTLASS source](https://github.com/NVIDIA/cutlass).
- Mesa NAK, [source repository](https://gitlab.freedesktop.org/mesa/mesa), used as comparative public ISA implementation material, not as an NVIDIA specification.
- NVBit, [source repository](https://github.com/NVlabs/NVBit), for binary instrumentation background.
- NVLift, paper/source details cited in [lifting research](ideas/control-flow-lifting.md).

## Matrix multiplication algorithms and graph compilers

- PyTorch, [pinned v2.7.1 Inductor configuration](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/_inductor/config.py), for the documented `emulate_precision_casts` behavior. The A100 cast-preservation experiment in the remediation report failed the unchanged BF16 parameter gate; the option was not promoted.

- Fawzi et al., [Discovering faster matrix multiplication algorithms with reinforcement learning](https://www.nature.com/articles/s41586-022-05172-4), Nature (2022).
- DeepMind, [AlphaTensor code and factorization data](https://github.com/google-deepmind/alphatensor).
- Microsoft, [SubCuber source](https://github.com/microsoft/subcuber), inspected at commit `c7b6fd8c72db4fc83982b26ea6e80e47567d674f`.
- Microsoft, [Mirage source](https://github.com/microsoft/mirage), for graph-level kernel synthesis and persistent-kernel context.
- NVIDIA, [FlashAttention-2 paper](https://arxiv.org/abs/2307.08691), for whole attention partition and data-movement context.

## Important comparison boundary

Public papers, source snapshots and hardware manuals describe possible
techniques. They are not OpenAssembler performance results. The A100 ledger
labels each source-inspection conclusion separately from a correctness-checked
and timed OpenAssembler experiment. Hopper-only TMA and thread-block clusters,
Blackwell-era features, FP8 acceleration on later devices, and DPX are not
available as A100 optimization paths.
