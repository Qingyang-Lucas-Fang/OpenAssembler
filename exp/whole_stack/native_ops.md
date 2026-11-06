# Native scalar neural-operator comparison

The current result is
`runs/native-ops-20261106T110300.124164Z.json`.
Reproduce from the repository root with:

```sh
python3 -m modal run exp/whole_stack/native_ops.py --rebuild
```

The experiment compiles the unchanged `examples/nn_ops.py` through the ordinary
OpenAssembler CLI, keeps the same-PTX ptxas cubin, and disassembles both. It runs
five operator families at two sizes on A100-SXM4-40GB, PyTorch 2.7.1+cu128,
CUDA 12.8, SM80. All build artifacts are isolated under `native_ops_build/`.
No production compiler or kernel source is changed.

## Contract and validation

All inputs/outputs are FP32. ReLU uses max(x, 0); GELU uses the tanh
approximation with the repository's coefficients; softmax is row-wise;
LayerNorm has no affine weight/bias and uses population variance plus `1e-5`;
the Adam-like step computes the repository's moment updates and parameter
formula. **It is not a full bias-corrected Adam/AdamW optimizer comparison.**

Every output element is compared against an FP64 formula applied to the exact
FP32 input values, for normal and zero input patterns. The script declares
tolerances before remote execution: ReLU exact; GELU `rtol=2e-5, atol=5e-6`;
softmax `5e-5, 1e-6`; LayerNorm `1e-4, 2e-5`; Adam-like step `2e-5, 2e-6`.
These gates are tolerance-based checks, separate from OA/ptxas bitwise equality.
Native OA and ptxas output buffers have 256-element guards on both sides.
Every case passed both patterns, all outputs, and all applicable guards.

OA and ptxas are bitwise equal on normal ReLU and softmax inputs. Normal GELU,
LayerNorm and Adam-like outputs differ in bits, while satisfying the declared
numerical tolerances. All zero-pattern cases were bitwise equal. Normal maximum
absolute OA errors were approximately `5.56e-6` for GELU, `5.82e-8` for softmax,
`2.48e-6` for LayerNorm, and `2.38e-7` for updated Adam-like weights. No
bitwise claim is made for those tolerance-only cases.

## Timing

Five warmup calls precede twenty shuffled single-invocation CUDA-event samples.
There are no CUDA graphs. The timed inputs are identical to the nonzero inputs
that passed full validation. Event intervals include host enqueue gaps; they
are application operator latency rather than an isolated instruction latency.
The PyTorch methods use public CUDA operators: clamped ReLU, tanh GELU, softmax,
LayerNorm, and an eager multi-operation implementation of the Adam-like formula.
LayerNorm and Adam-like return fresh outputs and include those allocations;
no artificial output copies are added. Native output arrays are preallocated.

| Operation / shape | OA ms | Same-PTX ptxas ms | PyTorch CUDA ms |
|---|---:|---:|---:|
| ReLU / 1,048,576 elements | 0.02304 | 0.02355 | 0.02150 |
| ReLU / 4,194,304 elements | 0.04608 | 0.04608 | 0.03226 |
| GELU / 1,048,576 elements | 0.02253 | 0.02355 | 0.02560 |
| GELU / 4,194,304 elements | 0.04659 | 0.04710 | 0.03738 |
| Adam-like / 1,048,576 elements | 0.03994 | 0.03584 | 0.12288 |
| Adam-like / 4,194,304 elements | 0.11110 | 0.10138 | 0.36864 |
| Softmax / 128 x 128 | 0.11264 | 0.08448 | 0.02048 |
| Softmax / 1024 x 512 | 0.40704 | 0.30618 | 0.02150 |
| LayerNorm / 128 x 128 | 0.09830 | 0.08499 | 0.03482 |
| LayerNorm / 1024 x 512 | 0.35533 | 0.29901 | 0.03686 |

## Interpretation

The strongest observed native weakness is reduction algorithm mapping. The
unchanged OpenAssembler examples assign one entire row to each scalar thread,
loop serially over its columns, and access memory with `width` elements between
adjacent lanes. The tested row counts launch only one or eight CTAs. That gives
little device parallelism and a long per-thread dependency chain. The same
source is slow even when compiled with ptxas, so the much larger gap to
PyTorch cannot be attributed solely to OpenAssembler instruction scheduling.
Parallel row reductions and coalesced vector access should precede further
minor scheduling work on these example kernels.

OA is also slower than ptxas for the same softmax/LayerNorm source. This isolates
an additional compiler/scheduling gap from the algorithm gap. Lower static
instruction count did not predict the winner: the OA reduction listings have
fewer static instructions than their ptxas counterparts, yet are slower.
Actual instruction issue/dependency counters would be needed to establish the
cycle-level cause. Register counts alone are also inadequate: OA LayerNorm
uses 20 registers against ptxas's 24, while ptxas remains faster.

The Adam-like OA kernel is faster than the eager PyTorch formula because OA
executes one fused kernel while the baseline materializes multiple arithmetic
intermediates and launches multiple operators. ptxas compiles the same fused
kernel faster still, so this is evidence for fusion rather than a unique OA
backend advantage. It must not be reported as beating tuned/fused AdamW.

The 1-Mi-element GELU case has a small OA timing advantage; the 4-Mi-element
case loses to PyTorch. These event timings include launch overhead and do not
establish a general GELU advantage. They also use a larger numerical error than
PyTorch's tanh GELU in the recorded inputs; precision and latency are both
reported rather than silently equated.

## Earlier attempts retained

`native-ops-20261106T110046.582765Z.json` included extra output copies in the
PyTorch LayerNorm/Adam-like baseline and is superseded for those comparisons.
`native-ops-20261106T110205.429734Z.json` removed those copies; its timed
nonzero distribution differed from the checked distribution. The current run
restores the identical checked inputs before timing and adds the harness hash.
All three raw files remain to make protocol changes explicit.
