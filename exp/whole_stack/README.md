# A100 whole-stack and native-stage study

The canonical conclusions and numerical failures are in
[WHOLE_STACK_RESULTS.md](../../docs/research/WHOLE_STACK_RESULTS.md). This
workspace contains reproducible experiments; it does not modify the production
compiler or automatically replace library kernels in user applications.

## Comparisons

1. `run.py` and `model.py`: ten transformer configurations, each with two
   process-order rounds, eager PyTorch versus the production OpenAssembler
   application frontend **worker** (AOTAutograd + TorchInductor/Triton).
   This is not an all-native OpenAssembler transformer.
2. `cli_proof/`: an actual CLI capture, verification, published bundle and
   cache-only replay of the small training model. This proves the application
   workflow; its three-sample timings are not the primary performance study.
3. `native_mlp.py`: two equivalent residual MLPs with native OA / same-PTX
   ptxas / cuBLAS GEMMs in forward, dX and dW, including conversion and
   transposed packing costs. Surrounding operations use shared PyTorch CUDA.
4. `native_ops.py`: native scalar ReLU, GELU, softmax, LayerNorm and an
   Adam-like update, at two sizes, versus same-PTX ptxas and public PyTorch
   CUDA operators. The Adam-like formula is not complete AdamW.
5. `native_counters.py`, `native_static.json`: six NCU profiles on two
   contrasted GEMM shapes and static SASS counts. Replay timings, static
   counts and application timings are different measurements.

## Run

Use an authenticated Modal client, the pinned PyTorch 2.7.1/CUDA 12.8 image and
an A100-40GB. Native builds also need the local CUDA toolkit wrapper and a
Python environment that imports the current checkout.

```sh
# Whole-transformer process study: saves every case immediately.
python3 -m modal run exp/whole_stack/run.py --rounds 2

# Analyze raw data without modifying it. GPU annotations are excluded from
# derived kernel totals and copies are separated from kernel launches.
python3 exp/whole_stack/analyze.py exp/whole_stack/runs/whole-stack-20261106T105757.119747Z.json
python3 exp/whole_stack/analyze.py --selfcheck

# Genuine native OA GEMM whole-MLP and kernel controls.
python3 -m modal run exp/whole_stack/native_mlp.py --mode build
python3 -m modal run exp/whole_stack/native_mlp.py
python3 -m modal run exp/whole_stack/native_counters.py

# Native scalar stage controls.
python3 -m modal run exp/whole_stack/native_ops.py --rebuild

# Separate BF16 numerical-policy sensitivity test, not a speedup benchmark.
python3 -m modal run exp/whole_stack/bf16_diagnostic.py

# Complete CLI application capture/replay proof.
python3.13 exp/whole_stack/cli_proof/prove.py
```

The whole-model timing is resident-data application latency, including host
enqueue gaps. It excludes cold startup, model construction, graph compilation,
data loading from storage and final verification copies. Worker wall times
retain those orchestration costs but do not isolate compiler passes. Inference
decode is batch-one fixed-window recomputation, not a KV-cache serving engine.

Numerical failures are retained and excluded from valid speedup claims. BF16
uses separately declared tolerances, not bitwise equivalence. Operator profiles
are a separate diagnostic pass; fused kernels cannot be split into invented
independent normalization or activation costs. Read the canonical report before
comparing ratios across the experiment families.

`native_REPORT.md`, `native_ops.md` and `native_attempts.md` are experiment-local
implementation notes. The canonical report links the primary raw JSONs and
distinguishes superseded harness attempts from usable measurements. Build trees
and runtime caches are regenerable; source hashes, manifests, comparisons and
raw samples provide provenance.

The primary run contains twenty completed process-order comparisons: eighteen
pass the numerical gates and both `train_wide_bf16` comparisons fail. A
`completed` execution status is not an all-correctness-passed status. Use the
derived analyzer's eligibility fields before making a speedup claim.
