# Native GEMM hybrid MLP: A100 whole-step results

This is an isolated experiment, not a production compiler integration. Two
equivalent residual MLPs use the **native OpenAssembler CLI-generated cubins**
for every forward, input-gradient and weight-gradient GEMM. Controls replace
only those GEMMs with ptxas compilation of the exact same PTX or cuBLAS GemmEx.
All methods share PyTorch CUDA LayerNorm, ReLU, residual addition, MSE loss,
FP32 AdamW, FP16 operand casts and transposed packing. These shared costs are
included. This does not establish whole-native Transformer training support.

Code: [native_mlp.py](native_mlp.py). Complete current raw record:
[native-mlp-20261106T105944.492233Z.json](runs/native-mlp-20261106T105944.492233Z.json).
Earlier repeat records are retained alongside it. Failed setup and numerical
gates are documented in [native_attempts.md](native_attempts.md).

## Contract and verification

The network is `x + W2(ReLU(W1(LayerNorm(x))))`, with biases on both linears.
Master parameters and surrounding operations are FP32. Each GEMM rounds its
operands to FP16 and accumulates/outputs FP32. The custom autograd contract
casts gradients to FP16 and explicitly packs transposed operands for dX/dW.
This is the same prescribed derivative for all methods, not a claim of
bit-equivalence to unmodified PyTorch autocast or to an all-FP32 network.
AdamW uses learning rate 3e-4, epsilon 1e-4 and shared foreach implementation.

Before timing, all three paths run three optimizer steps from identical
parameters, inputs and targets. Every GEMM output is checked in full against
an FP32 product of its actual quantized operands; output guards are checked.
Output, loss, input gradients, all parameter gradients/values and AdamW state
are compared. OA and ptxas require full raw-bit equality. The fixed cuBLAS
comparison gates are relative L2 <5e-3 for gradients/optimizer state and <5e-4
for outputs/parameters, with a 2e-6 maximum-absolute fallback near zero.

Both MLPs pass OA-vs-ptxas full-state bitwise comparison. All 108 verification
GEMMs (18 per method, three methods, two cases) pass full-output and canary
checks. The smaller MLP passes the cuBLAS full-state gate too. The larger
MLP's first-layer weight gradient has relative L2 0.006364095 versus cuBLAS
after three optimizer steps (max absolute 9.83872e-6), above its preset 0.005
gate. The gate was not loosened; the larger cuBLAS whole-step timing is
omitted. Its individual stateless GEMMs remain independently checked.

Individual OA/ptxas GEMM errors versus FP32 products remain below 3.32e-6
relative L2 for the small training case and 6.62e-6 for the larger one;
cuBLAS's own product errors are below 1.71e-6 and 2.32e-6. Correct individual
products combined with diverging three-step low-precision gradients are
consistent with rounding propagation through conversions and ReLU boundaries;
this is not a complete causal isolation of the numerical divergence.

## Whole training and inference

Device: A100-SXM4-40GB, PyTorch 2.7.1+cu128, CUDA 12.8. A whole timing is the
median of 15 shuffled paired single-step CUDA-event spans after five warmup
steps. Synchronized host-wall samples are also retained. These spans include
host dispatch/idle gaps and all stated casts/packing, not only device kernels.
Inference reloads identical original parameters and performs a full-output
comparison before timing. No whole-step CUDA graph is used.

| MLP (batch, width, hidden) | Method | Training step ms | Inference ms |
|---|---|---:|---:|
| 512,512,2048 | OA | 1.6896 | 0.3635 |
| 512,512,2048 | ptxas | 1.6916 | 0.3451 |
| 512,512,2048 | cuBLAS | 1.6056 | 0.3318 |
| 1024,1024,4096 | OA | 2.1340 | 0.4567 |
| 1024,1024,4096 | ptxas | 2.0746 | 0.4065 |
| 1024,1024,4096 | cuBLAS | not admitted | not admitted |

Worker-to-worker absolute time varies. In two earlier complete runs, small
training OA/ptxas were 1.4858/1.4807 ms and 1.6906/1.7203 ms; their ordering
reverses, so a tiny apparent OA training win is not established. Inference
and isolated kernel measurements consistently expose an OA deficit. Shared
runtime/launch/optimizer costs substantially dilute the GEMM gap at whole-step
level, which is precisely why a GEMM number cannot replace whole-step timing.

## Isolated matched GEMM kernels

The complete current record also captures preallocated ten-GEMM CUDA graphs
for all three methods, times nine shuffled paired replays, and divides each
replay duration by ten. This removes packing, allocations and most host launch
overhead from the kernel measurement. cuBLAS is explicitly rebound to the
capture stream. Outputs are poisoned **after capture and before replay**, then
checked in full with guards, so an empty graph cannot pass.

| M,N,K (C=A@B.T) | OA ms | ptxas ms | default GemmEx ms |
|---|---:|---:|---:|
| 512,512,2048 | 0.05960 | 0.04157 | 0.01546 |
| 512,2048,512 | 0.01997 | 0.01393 | 0.02427 |
| 1024,1024,4096 | 0.11387 | 0.07956 | 0.07834 |
| 1024,4096,1024 | 0.08448 | 0.05990 | 0.05837 |
| 2048,512,512 | 0.02109 | 0.01454 | 0.02611 |
| 4096,1024,1024 | 0.08786 | 0.06236 | 0.06963 |

OA beats this **default** cuBLAS GemmEx dispatch on two small expanded/tall
products, but ptxas beats OA for every identical-PTX product. The wins against
default GemmEx therefore support shape/algorithm-selection specialization, not
an inference that OA's backend schedules SASS better than ptxas. No exhaustive
cuBLASLt search was performed. These products also do not establish an admitted
large cuBLAS whole-training comparison.

## Stage diagnosis and static evidence

Separate diagnostic event scopes cover LayerNorm forward/backward, ReLU
forward/backward, each linear's forward packing/cast, forward GEMM, dX/dW
packing/transposes, dX/dW GEMMs, bias/gradient reduction, residual, loss and
AdamW. The full raw record contains all five samples per stage. Added event
instrumentation substantially inflates short stages; these nested diagnostic
medians must **not** be summed or substituted for whole-step medians.

For the larger OA MLP, diagnostic backward packing/transposes cost about
0.171 and 0.179 ms per linear, while each of its six GEMM scopes is about
0.09-0.12 ms. AdamW is about 0.72 ms in that instrumented pass. Shared
transposes/packing and optimizer work are large next integration targets, but
their measured event spans include dispatch effects.

[native_static.json](native_static.json) records SASS/opcode/control counts
through the first EXIT for all six OA/ptxas pairs. OA uses 216 registers versus
ptxas's 231/234, no local-memory bytes, and 64KiB dynamic shared memory.
Despite fewer registers, OA is slower on every isolated same-PTX comparison.
OA statically contains 282 MOVs and sets stall8 on all 512 static HMMA words.
ptxas often removes accumulator initialization/moves and interleaves issue
controls: on K2048/4096 cases it has only one MOV, 512 HMMA words, and many
HMMA stall1 controls mixed with stall8. On K512/1024 ptxas unrolls substantially
more, so its larger static HMMA/LDSM count must not be mistaken for more
dynamic mathematical work. Static counts and stall fields are clues, not a
cycle model or a proof that it is safe to shorten OA's stalls.

## Bounded hardware-counter replay

Nsight Compute 2025.1 collected all six method/shape profiles in
[native-counters-20261106T110405.921716Z.json](runs/native-counters-20261106T110405.921716Z.json).
The replay uses a fresh driver context, zero-filled inputs, default replay
cache behavior and `--clock-control none`. The earlier full-output-checked
cubin hashes are retained in the record. One replay per case is diagnostic,
not a confidence interval or a causal decomposition. Its durations are never
substituted for the graph medians above.

| M,N,K | Method | Grid | Tensor active | Short SB | Long SB | Wait | Total instructions | Tensor instructions | Shared LD conflicts |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 512,512,2048 | OA | 4,4,1 | 48.16% | 10.00% | 11.38% | 58.50% | 577,856 | 262,144 | 0 |
| 512,512,2048 | ptxas | 4,4,1 | 69.67% | 2.75% | 7.05% | 60.86% | 433,408 | 262,144 | 0 |
| 512,512,2048 | cuBLAS | 8,8,1 | 52.88% | 5.21% | 4.79% | 48.61% | 889,088 | 262,144 | 0 |
| 512,2048,512 | OA | 4,16,1 | 37.61% | 8.34% | 19.04% | 49.24% | 680,192 | 262,144 | 0 |
| 512,2048,512 | ptxas | 4,16,1 | 55.19% | 1.74% | 9.42% | 47.01% | 432,640 | 262,144 | 0 |
| 512,2048,512 | cuBLAS | 16,8,2 | 26.21% | 4.02% | 13.78% | 26.44% | 1,521,101 | 262,144 | 10,040 |

The OA/ptxas comparison has identical CTA grid and tensor instruction count,
but OA issues 33% more total instructions on the long-K shape and 57% more
on the expanded-N shape, with much higher short-scoreboard stalls and lower
tensor activity. Zero shared-load conflicts make shared-bank swizzling a
less compelling first explanation for these two OA deficits. The static
move/control observations above support testing scalar/address/control work
and instruction interleaving; counters do not prove a specific operand
collector or register-bank mechanism.

cuBLAS selects `ampere_s16816gemm_fp16_64x64_sliced1x2_ldg8_stages_64x5_tn`
for the long-K case, launching 64 CTAs rather than OA's 16. It wins despite
more total instructions, consistent with better parallel coverage from its
smaller tile. Tensor activity is normalized to active cycles and must not be
mistaken for GPU-wide SM coverage.

On expanded N, cuBLAS selects
`ampere_s16816gemm_fp16_128x64_ldg8_stages_32x6_tn` and launches a larger grid
with an additional z dimension. It executes more than twice OA's total
instructions and has lower tensor activity plus nonzero shared-load conflicts.
These support dispatch/resource/staging overhead as a candidate explanation
for the default-GemmEx loss here; grid-z alone does not establish the precise
internal algorithm. A tuned cuBLASLt control and repeated counters are still
needed before generalizing either vendor win or loss.

## Reproduce

```sh
python3 -m modal run exp/whole_stack/native_mlp.py --mode build
python3 -m modal run exp/whole_stack/native_mlp.py
python3 -m modal run exp/whole_stack/native_counters.py
```

The first command rebuilds six cubin pairs through the ordinary native OA CLI
and validates disassembly. Builds stay under `native_builds/`; raw results stay
under `runs/native-*.json`. The local build assumes the repository's `.venv`
Python and working CUDA toolkit wrapper. Modal uses one bounded A100 container.
