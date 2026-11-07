# Small Transformer Training Example

This directory contains a self-contained character-level Transformer language-model training example. It trains on a bundled text corpus, so no dataset download is required.

## Dependencies

- Python 3.10 or newer
- PyTorch 2.0 or newer (`torch>=2.0,<3.0`)

From the repository root, install the dependency with:

```bash
python3 -m pip install -r testcases/test_whole_repo/requirements.txt
```

## Train

```bash
python3 testcases/test_whole_repo/train_transformer.py
```

For a quick CPU smoke run:

```bash
python3 testcases/test_whole_repo/train_transformer.py --device cpu --steps 10 --batch-size 4 --block-size 16 --d-model 32 --nhead 4 --num-layers 1 --new-tokens 32
```

## Test

The tests use Python's built-in `unittest` module; PyTorch from the dependencies above is required:

```bash
python3 -m unittest -v testcases/test_whole_repo/test_train_transformer.py
```

## Compile and replay with OpenAssembler

`openassembler.toml` defines a small, deterministic CUDA capture workload for
this script. It records CUDA-event medians for batch construction, forward/loss,
backward, optimizer and end-to-end steps. From the repository root, compile it
through the TorchInductor/Triton application frontend and compare eager and
compiled training state:

```bash
openassembler compile testcases/test_whole_repo --backend modal --verify
openassembler run testcases/test_whole_repo --backend modal --verify-application
# Or compile and replay with the same transport in one command:
openassembler cnr testcases/test_whole_repo --backend modal --verify
# With a configured Linux A100 SSH host:
openassembler cnr testcases/test_whole_repo --backend user@a100 --verify
```

The default build root is `testcases/test_whole_repo/output/`; the GPU cubins
are in `cubin/applications/test_whole_repo/` and readable listings in
`sass/applications/test_whole_repo/` below that root. A custom build root uses
`--out-dir DIR` on both `compile` and `run`. The repository path selects the
saved application automatically. `run` replays the captured source/arguments;
edit the configuration and compile again to change the workload.

Python training control flow and the matching PyTorch runtime remain part of
execution. Capture uses PyTorch 2.7.1 on an A100; replay must match its exact
recorded runtime. See [repository transport setup](../../docs/REPOSITORY_WORKFLOW.md)
and [application contracts](../../docs/FRONTENDS.md#pytorch-training-applications).

The application report stores `eager_timing` and `timing` records when profiling
is enabled. These timings compare PyTorch eager execution with the captured
TorchInductor/Triton cache on the same GPU. They do not claim that the full
training graph was lowered by OpenAssembler's native PTX/SASS backend; host
Python, PyTorch and CUDA libraries remain runtime dependencies.
