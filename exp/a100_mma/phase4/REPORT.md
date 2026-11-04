# A100 phase four: further research and native CLI integration

2026-11-04, branch `codex/a100-tensor-cli`. The user authorized moving effective
research improvements into the CLI. Existing working-tree changes were preserved;
`start-state.json` records the starting revision, dirty files and production
source hashes. No commit or merge is made by this work.

## Additional experiments

| Experiment | Validation | Result |
|---|---|---|
| cp.async L2 prefetch64/128/256B | Matching ptxas probes,80 exact randomized disassembly roundtrips, unchanged nonpolicy bits/controls, full-output A100 checks for10 variants | Small, inconsistent effects. Not promoted. |
| B-fragment windows2/4 and bounded prefetch | Per-output K order, exact disassembly, poisoned numerical checks,48 full-output GPU check rows | All six variants slower in all seven rounds. Not promoted. |

The L2 policy exists on SM80, but a vendor-like modifier alone does not reproduce
a vendor kernel's performance. Fragment windows reduce registers216→204/208
while retaining the same HMMA/LDSM/copy counts. Their additional safe tensor-epoch
drains accompany8–28% worse time at4096³. More precise dependency research may
revisit the drains; this work preserves proven edges. Sources, encoding evidence,
timing samples and result hashes are in [`README.md`](README.md) and
[`fragment_README.md`](fragment_README.md).

## Production integration

The standard `compile` command now detects supported SM80 tensor PTX and invokes
`openassembler.backend.tensor`. The validated generator is available through:

```sh
openassembler gemm --m 4096 --n 16384 --k 4096 --dtype f16 --out-dir output
openassembler run output --backend modal --budget 300 --job-timeout 120
```

The promoted implementation includes paired B `LDSM.x4` loads with explicit
implicit-register dependencies; aligned vector stores and reused row pointers;
shared swizzling; reduced local homes; carry-linked64-bit pointer updates; proven
shared LDGSTS displacements; a stage-specialized two-tile loop; tensor reuse;
HMMA8/epoch-drain scheduling; copy scoreboard0 reservation; LDSM address WAR
protection; and safe scoreboard recycling. Tensor code preserves source order
instead of using the scalar list scheduler. Scalar-only compilation keeps its
existing pipeline.

Configuration is immutable per call. Encoding templates and measured fields are
bundled with provenance. Encoding performs no experiment imports, runtime tool
probes, corpus bootstrap or generated-cache writes. Unsupported tensor modes,
architectures and control flow raise diagnostics. With the normal SASS listing
enabled, compilation checks exact encoded words/controls and tensor/memory
operands against nvdisasm before publication.

Generated source contains a launch contract, retained only if the plan validates
and the complete body matches the generator. `run` uses the actual half-precision
sizes,2-D grid,128 threads and64-KiB shared opt-in. Uncontracted tensor kernels
require `--spec`; they never enter the scalar launch guesser. Multi-step jobs
retain the largest shared opt-in requested per function.

The runner uses deterministic small integers in FP16/BF16 for bitwise comparison
with ptxas. This is an explicit smoke check. Floating-point validation and graph
benchmarking remain separate artifacts.

## Validation

* **452 repository tests pass** in148.64 seconds, including scalar/compiler,
  CLI, runner, tensor encoding/dependencies, exact words, generator control flow,
  poisoned full-output numerics and toolkit checks. Log: `regression-final.log`.
* All12 stagepair/combined PTX sources match the retained sources byte for byte.
  Six production CLI `.text.gemm` sections match their phase-three kernels
  exactly, including controls/padding. Large kernels use216 registers.
  `native-build-evidence.json` retains commands and hashes.
* All six native CLI cubins pass eight A100 input cases, checking every output
  against FP32-output cuBLAS plus sampled CPU FP64, guards and finite writes.
  Native/ptxas/retained-kernel comparison yields192 complete-output check rows.
  Normal/scaled L2 tolerance remains2e-5; zero/integer checks are exact.
  `native-gpu-summary.json` hashes all raw results.
* Public `gemm`→`run --backend modal` passes FP16128×128×128 and
  BF16128×256×384 exact-integer checks, covering minimum K and repeated-loop/tail
  contracts. See `cli-*-build.log`, `cli-*-gpu.log` and each output's `run.json`.
* A wheel builds and, extracted outside the repo, the normal build dispatcher
  produces the exact216-register/1600-word BF16 square stream without `exp/`
  imports or an encoding oracle. Logs: `wheel-build.log`, `wheel-isolation.log`.

| Native CLI M×N×K | Input | CLI ms | Identical phase-three code ms | Matched cuBLAS ms |
|---|---|---:|---:|---:|
| 4096×4096×4096 | FP16 | 0.668058 | 0.668058 | 0.595405 |
| 4096×16384×4096 | FP16 | 2.518630 | 2.536806 | 2.507418 |
| 16384×4096×4096 | FP16 | 2.846822 | 2.829926 | 2.425651 |
| 4096×4096×4096 | BF16 | 0.651930 | 0.652186 | 0.575846 |
| 4096×16384×4096 | BF16 | 2.512128 | 2.515507 | 2.228019 |
| 16384×4096×4096 | BF16 | 2.755635 | 2.775450 | 2.279680 |

These timings check promotion; identical streams are not a new algorithmic
speedup. Seven shuffled rounds,20 nodes/graph,24 warmup rounds and unlocked clocks
match the research protocol. The retained phase-three comparison established
16–27% less kernel time than the earlier direct schedule. Default cuBLAS and
separately tuned cuBLASLt remain different targets; universal parity is unproven.

The supported operation is full-tile RCR `C=A@B.T`, FP16/BF16 inputs and FP32
accumulation/output. Native tails, arbitrary strides/layouts, fused epilogues,
broader tensor control flow and PyTorch training integration remain work to do.
No whole-transformer speedup is claimed. The public contract is in
[`docs/TENSOR_GEMM.md`](../../../docs/TENSOR_GEMM.md).

## Reproduce

```sh
# Production CLI; no experiment imports or generated research artifacts.
.venv/bin/python -m openassembler gemm --m 128 --n 128 --k 128 --out-dir output
.venv/bin/python -m openassembler run output --backend modal

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests

# Exact-code promotion audit requires the retained phase-three reference builds.
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/a100_mma/phase4/native_build.py
PYTHONDONTWRITEBYTECODE=1 python3 -m modal run exp/a100_mma/modal_run.py \
  --suite phase4/native_suite.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/a100_mma/phase4/native_summary.py
```

Reference-build order is in [`REPRODUCING.md`](../../../REPRODUCING.md).
Production CLI compilation/execution does not need those research artifacts.
