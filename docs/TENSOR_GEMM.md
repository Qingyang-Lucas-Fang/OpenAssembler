# A100 tensor GEMM in the CLI

The standard compiler now selects the SM80 tensor backend when PTX contains
`mma`, `ldmatrix`, or asynchronous-copy instructions. It generates its own SASS
and uses ptxas for the ELF container and a reference compilation of the same PTX.
There are no imports from `exp/`, experimental switches, runtime encoding probes,
or fallback to vendor code unless the user explicitly requests `--fallback ptxas`.

Generate and compile the measured full-tile GEMM:

```sh
openassembler gemm --m 4096 --n 16384 --k 4096 --dtype f16 --out-dir output
openassembler run output --backend modal --budget 300 --job-timeout 120

# Optional A100 Morton traversal; M and N must be multiples of 1024.
openassembler gemm --m 16384 --n 4096 --k 4096 --dtype f16 \
  --cta-order morton8 --out-dir output_morton

# Optional fused FP32 bias + residual + ReLU epilogue.
openassembler gemm --m 4096 --n 4096 --k 4096 --dtype bf16 \
  --epilogue bias_residual_relu --out-dir output_fused
```

Use `--dtype bf16` for BF16 inputs. Compilation requires Python 3.12+ and the
CUDA toolkit; it does not require a GPU. Local GPU execution needs the CUDA
driver. Modal execution uses the existing optional authenticated Modal client.

The operation is `C = A @ B.T`: A is contiguous `[M,K]`, B is contiguous `[N,K]`,
and C is contiguous `[M,N]`. Inputs are FP16/BF16; accumulation and output are
FP32. M and N must be positive multiples of128; K must be a positive multiple
of128. A/B require16-byte alignment and C eight-byte alignment. Arbitrary tails,
strides, transposed storage and alpha/beta are not handled by this command.
Unsupported dimensions or modes fail before compilation. This is A100/sm_80;
other architectures require their own validated backend.

The default is `--cta-order m_fast --epilogue none`, preserving the validated
GEMM contract. `--cta-order morton8` changes only logical output-tile assignment;
it requires complete 8×8 CTA groups (M and N multiples of1024). It does not
change the MMA, shared-memory or synchronization schedule. The repeated A100
result is shape-specific to the tested tall 16384×4096×4096 workload. MLP
Morton candidates regressed, and square results were within timing noise, so
the compiler never selects Morton automatically.

`--epilogue bias_residual` computes `(A @ B.T + Bias) + Residual`; adding
`_relu` applies finite-value ReLU after the residual. Bias is contiguous FP32
`[N]`; Residual is contiguous FP32 `[M,N]`. The generated kernel signature is
`(A, B, C, Bias, Residual)`. The feature currently requires the measured
`stagepair` schedule and full-tile dimensions. Callers must provide both extra
buffers. Generic `openassembler run` uses zero-filled auxiliary buffers as an
exact-integer smoke check; use the A100 validation for nonzero auxiliaries and
full floating-point checks.

Both options remain explicit. Current evidence supports Morton only on the
tested tall grid and fused epilogues on square 4096³ cases; it does not justify
a shape-independent policy or a whole training-step speed claim. Fusion
regressed on smoke and was slightly slower on the tested MLP shape.

The default `stagepair` plan uses128×128×64 CTA tiles, four64×64 warp tiles,
128 threads and65,536 bytes of dynamic shared memory. It combines:

* Paired B-fragment `LDSM.x4` loads, with aligned register halves passed directly
  to HMMA and both implicit register reads represented in the dependency IR.
* XOR-swizzled shared operands, direct carry-linked64-bit pointer increments,
  reduced local register homes and proven shared-copy address displacements.
* Alternating stage offsets specialized into a two-tile loop, with a peeled
  final pair to prevent speculative out-of-bounds copies.
* Adjacent FP32 stores grouped into64-bit stores, reusing each output-row pointer.
* Tensor operand reuse hints and source-order HMMA scheduling, with epoch drains,
  protected scoreboard recycling and LDSM address WAR dependencies. Scoreboard0
  is reserved for asynchronous-copy completion; ordinary operations use1–5.

`--schedule combined` retains the earlier paired-load/vector-epilogue plan for
comparison. The scalar list scheduler is not applied to tensor kernels: all
public optimization levels select the validated tensor policy, recorded in the
manifest. Scalar-only kernels retain their existing optimization pipeline.

The command writes the normal cubin, SASS, reference cubin and manifest. It also
keeps annotated generated PTX in `output/.build/generated/gemm.ptx`. Recompile
that file through the ordinary compiler:

```sh
openassembler compile output/.build/generated/gemm.ptx --out-dir another_output
```

The launch contract is retained only if its dimensions validate and its body
exactly matches the built-in generator. Edited or unannotated PTX still uses
native tensor instruction selection, but `run` requires an explicit `--spec`
instead of guessing scalar buffers and launch geometry. `--emit-only` writes
the annotated source without invoking the toolkit. The Python API is
`openassembler.kernels.gemm.GemmPlan` and `generate(plan)`.

Unannotated tensor kernels can use `run --spec` with `f16` and `bf16` buffers
of two-byte elements, numeric initialization lists or deterministic small
integer random inputs. Supply the real grid, block, shared memory and vector
alignment. `-o NAME` changes filenames, not the PTX entry name: select
`--kernel gemm --source NAME.ptx` when several GEMMs share one manifest.

`run` uses the retained contract and deterministic small integers in the input
precision, compares every FP32 output bitwise with ptxas, and opts into the
required shared memory. This is an exact-integer smoke comparison. It is distinct
from the six-candidate, full-output A100 validation and CUDA graph timings
retained in the evidence archive. Generic `compile --verify` can report
tensor PTX as unmodeled; it must not be treated as full tensor validation.

The raw PTX tensor subset currently covers dense
`mma.sync.aligned.m16n8k16.row.col.f32.{f16,bf16}.{f16,bf16}.f32`, supported
shared `ldmatrix` forms,16-byte `cp.async.cg` copies and explicit copy-group
commit/wait. Control flow is restricted to straight-line code or one guarded
loop latch with an unguarded body. Unsupported arithmetic, modifiers and control
flow raise a compiler diagnostic. This is not a general CUDA tensor compiler.

Run the compiler, plan-contract and full integer emulator tests with:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  tests/test_backend_tensor.py tests/test_gemm_codegen.py tests/test_cli_gemm.py
```

The production CLI emits exactly the instruction streams previously validated
for all six projection workloads. Native CLI cubins separately pass full-output
A100 checks and same-worker comparison with those retained cubins. The promoted
schedule previously reduced time16–27% versus the prior direct experimental
schedule; this is a component-kernel result, not whole-transformer speedup.
The default cuBLAS and tuned cuBLASLt comparisons remain separate.

Further research tested L2-prefetch policies and bounded B-fragment windows.
Both passed correctness; neither established a repeatable gain, so neither is
enabled in the production plan. Producer warps, larger/deeper pipelines and
low-rank transforms also remain experiments until their performance and numerical
contracts justify promotion.

See [the current compiler contract](research/A100_MMA_COMPILER.md),
[the complete results ledger](research/A100_MMA_RESULTS.md), and
[reproduction instructions](../REPRODUCING.md).
