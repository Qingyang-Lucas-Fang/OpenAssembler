# System integration review — November 4, 2026

This review covers the production scalar compiler, native SM80 tensor path,
PyTorch application frontend, output manifests, GPU runner, interactive shell
and installed wheel. It preserves the current tensor instruction streams and
the numerical policies used by the retained experiments.

## Research and implementation boundaries

| Path | Optimizations actually used | Validation and boundaries |
|---|---|---|
| Standalone Python subset / CUDA C/C++ / PTX | Constant division/remainder lowering, arithmetic fragments, DCE/coalescing, register allocation, optional scalar unrolling/list scheduling | PTX reference plus poisoned-register emulator checks; GPU comparisons against the same PTX's ptxas build. Unsupported constructs are explicit errors. |
| Native SM80 tensor PTX and `gemm` | Paired B-fragment LDSM, vector epilogue, shared swizzle, pointer carry chains, stage-pair specialization, operand reuse, HMMA8/epoch-drain15, protected scoreboards and address WAR edges | Six retained complete instruction hashes, pipeline/address tests, poisoned integer emulation, disassembly, and fresh A100 execution. Full-tile FP16/BF16 RCR remains the supported generated contract. |
| Configured PyTorch repository | Dynamo/AOTAutograd capture and TorchInductor/Triton generation | Eager/compiled observations and saved runtime-cache replay. This route does not automatically substitute native OpenAssembler GEMMs. Host Python, ATen and CUDA libraries remain runtime dependencies. |
| Research-only candidates | Cache-prefetch hints, fragment windows, deeper pipelines, producer warps, low-rank tensor transforms | Correctness/performance results remain in `exp/a100_mma/`; no new candidate was promoted by this integration review. |
| SASS lifting / decompilation proposals | Proposed semantic IR, idiom recovery and debug-line bridges | Research infrastructure/design notes, not an arbitrary-repository compilation feature. |

The encoder/control/container findings in research notes inform the existing
implementation. Historical notes keep their original date and evidence tags;
later native-MMA promotion is linked explicitly from the earlier agenda.
The fitted tensor arithmetic model is useful for the recorded finite cases;
special-value behavior and general tensor accuracy remain separate questions.

The retained 16–27% tensor-kernel time reductions compare the promoted schedule
with its earlier direct experimental schedule. They are not whole-transformer
speedups. The scalar NN study's floating mismatches and slower kernels also
remain visible; this review does not reinterpret them as successful optimization.

## Integration fixes

- Explicit source files always use their source frontend, including files inside
  an application directory. Missing files no longer trigger training capture.
- Application-only and kernel-only options are checked before work starts.
  Application `--no-sass`, watchdog overrides, fractional budgets and saved
  replay reports now work through both one-shot commands and the shell.
- Nested project forwards are inlined into the owning compiled model and observed
  consistently in eager mode. Multiple model instances keep stable identities.
- Output writers use a lock per output root. Separate roots remain independent;
  competing writers fail clearly. Mixed manifests preserve existing entries,
  refuse cross-path ownership collisions, and clean only recorded stale files.
- Corrupt manifests, incomplete runtime metadata, missing requirements, invalid
  output names, output/source overlap and modified bundle checksums fail early.
  Archive extraction also rejects normalized duplicate paths.
- `run --spec` supports FP16/BF16 buffers, correct element sizes, numeric lists,
  BF16 ties-to-even conversion and worker-generated exact integer inputs.
- The REPL uses the same dispatcher and interrupt cleanup as the one-shot CLI;
  malformed quoting leaves the shell usable. Help/banner text includes native
  tensor and application workflows.

## Validation

The machine-readable record is
[integration-20261104.json](evidence/integration-20261104.json). It identifies
inputs, source hashes, test outcomes, actual devices, and raw local log hashes.
All GPU checks below are fresh Modal runs from this review.

| Check | Result |
|---|---|
| Repository tests with CUDA 12.8 toolkit | 486 passed in 148.85 seconds; no failures or skips |
| Scalar probe-corpus check | 262 accepted and semantically matching probes; 279 unsupported probes skipped. Current instruction totals: 3,248 versus ptxas 3,075 (+5.6%). Text aliases match literally on 245/262; no semantic mismatch. |
| Combined native GPU batch | Vector-add, 64-thread cross-warp barrier, FP16 128×128×128 GEMM and BF16 128×256×384 GEMM: 4/4 bit-exact comparisons |
| Explicit BF16 tensor spec | Two-byte buffers, actual 2-D launch and 64-KiB shared opt-in: bit-exact |
| Scalar `cnr --verify --backend modal --spec …` | Build/emulator/GPU comparison pass |
| Original Transformer repository `cnr --verify` | 57 emitted cubins; 92 tensors checked; two optimizer steps; cache-only replay with zero new GPU code generation |
| Nested/multiple-model application | Five emitted cubins, no SASS with `--no-sass`; 44 tensors checked; two instances, four forward calls, two optimizer steps; replay with zero GPU code generation |
| Installed wheel outside checkout | No `exp` imports; BF16 native stream matches retained 1,600-word/216-register hash; scalar build/verify, tensor build, dry-run and status pass |

Small GPU timings in these checks are correctness smoke measurements, not new
performance benchmarks. The wheel includes the encoding/runtime data and has no
generated Python bytecode or experiment package dependency. Audit apps completed
and the final Modal inventory showed no running containers.

## Reproduce

```sh
uv run pytest -q
openassembler check
openassembler cnr examples/vector_add.py --verify --backend modal \
  --spec examples/run_spec.json --out-dir output/audit-scalar --budget 60 --job-timeout 20
openassembler gemm --m 128 --n 256 --k 384 --dtype bf16 --out-dir output/audit-tensor
openassembler run output/audit-tensor --backend modal --budget 60 --job-timeout 20
openassembler cnr testcases/test_whole_repo/ --verify --out-dir output/audit-transformer \
  --build-budget 300 --budget 120 --job-timeout 100
openassembler compile tests/data/torchfront_nested/ -o nested_application --verify --no-sass \
  --out-dir output/audit-nested
openassembler run output/audit-nested --application nested_application --verify-application \
  --budget 120.5 --job-timeout 100
```

Scale within the explicit contracts: source/cache bundles up to 256 MiB and
20,000 files, verification snapshots up to 128 MiB, sm_80 GPU code, and captured
application shapes/arguments. Per-output locking protects writers; it is not a
distributed build service or a multi-file transaction under power loss. Avoid
running from an output root while replacing its build. Real SSH execution,
other GPU architectures, arbitrary host-language compilation and distributed
training remain outside this review's hardware validation.
