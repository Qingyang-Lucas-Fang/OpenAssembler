# Current A100 Counter and Schedule Experiments

This directory is an isolated research harness. It does not change the
production compiler. The first experiment compares the current full-output
validated tall FP16/BF16 candidates on an A100:

- OpenAssembler `m_fast`
- corrected OpenAssembler `morton8`
- ptxas compiled from the same generated PTX
- cuBLAS, when the remote image exposes a usable library/API

The worker validates every output and canary before timing or counter replay.
Nsight Compute collection is attempted only when `ncu` is present in the remote
image. The metric list is queried dynamically because names vary by NCU build.
Counter replay perturbs caches, so CUDA-graph timings and counter data are
stored separately.

The harness also records the next controlled search dimensions: fragment
ping-pong versus shared-stage depth, independent accumulator chains versus
register/occupancy cost, shared layout/load order, and `cp.async` cache mode.
Those variants are hypotheses until generated PTX, valid SASS, complete output
checks and paired A100 timing exist.

## Run

```sh
python3 -m modal run exp/a100_counters/run.py --mode preflight
python3 -m modal run exp/a100_counters/run.py --mode build-current
python3 -m modal run exp/a100_counters/run.py --mode validate
python3 -m modal run exp/a100_counters/run.py --mode validate-cublas
python3 -m modal run exp/a100_counters/run.py --mode validate-cublas-ncu
```

`--preflight` checks local tools and requests an A100 to query remote NCU and
cuBLAS availability. `--build-current`
regenerates the two tall candidates under a caller-selected output root. The
`validate` mode profiles the matched OA and ptxas cubins with NCU. The separate
`validate-cublas` mode skips NCU replay, checks the library output against the
FP32 reference, and includes cuBLAS in graph timing. Both Modal modes write
JSON under this directory and never overwrite `docs/evidence`. The
`validate-cublas-ncu` mode adds the same NCU replay to the three-way comparison;
it is slower and perturbs caches, so use it for counters rather than timing.

## Interpretation

Historical matched profiles show equal Tensor Core instruction counts but more
OpenAssembler total instructions, short-scoreboard stalls and wait stalls.
That is a hypothesis about scalar/address/control overhead, not proof of an
operand-collector bottleneck. A current profile must compare the same launch
geometry and full-output contract before any schedule change is promoted.

Primary references: [NVIDIA Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html),
[CUTLASS efficient GEMM](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html),
[PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html),
[Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html),
and [Ampere microbenchmarking](https://arxiv.org/html/2208.11174).
