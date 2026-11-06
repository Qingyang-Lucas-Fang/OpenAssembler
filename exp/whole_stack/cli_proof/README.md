# Production CLI Capture and Replay Proof

This is a small integration proof of the real `openassembler compile` and
`openassembler run` application path. It is separate from the larger
whole-stack worker benchmark and is not a performance result for native
OpenAssembler SASS generation.

The copied model uses FP32 parameters and activations with TF32 disabled,
batch 2, sequence 32, width 64, four heads, one transformer block, and a
256-token synthetic vocabulary. It performs two warmup and three measured
optimizer steps, retaining its final gradients for verification.

Reproduce from the repository root:

```sh
python3.13 exp/whole_stack/cli_proof/prove.py
```

`prove.py` copies the current experiment model into `application/`, writes an
explicit application configuration, invokes the production CLI with
`compile --verify --no-sass`, and then invokes saved-application replay with
`--verify-application`. It writes command logs and `summary.json`. The source
and runtime-cache hashes, cubins, graph records, capture report, manifest, and
replay report are retained under `build/`.

## Recorded result

Both commands returned zero on NVIDIA A100-SXM4-40GB, Python 3.11.13,
PyTorch 2.7.1+cu128, and CUDA 12.8. Capture emitted two graphs and 35 cubins.
Every cubin is explicitly marked `compiled_by: torch-inductor-triton`; these
whole-model graphs do not use the native OpenAssembler PTX-to-SASS backend.

Capture and replay each verified 77 tensors: the first two observed outputs,
final parameters, 15 final gradients, and optimizer state. Verification uses
the declared output/gradient tolerance `rtol=0.001, atol=0.0001` and
parameter/optimizer tolerance `rtol=0.02, atol=0.002`. The reported values are
tolerance-based checks, not bitwise identity or long-run convergence evidence.

Saved replay confirmed five optimizer steps and **zero new GPU code
generation**. Its resident-data median step was 3.405 ms eager and 2.550 ms
compiled, from only three samples; these are integration timings, not the
primary suite's performance estimates. Local command wall times were about
54.9 seconds for capture and 30.0 seconds for replay, including remote startup,
transport, verification, and runtime overhead. They do not isolate compilation.

The saved source archive is 3,912 bytes, runtime cache archive 1,065,073 bytes,
and emitted cubins total 422,968 bytes. Exact values and the copied model hash
are in `summary.json`.
