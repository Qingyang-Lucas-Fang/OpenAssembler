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
this script. From the repository root, compile it through the TorchInductor/
Triton application frontend and compare eager and compiled training state:

```bash
openassembler compile testcases/test_whole_repo/ --out-dir output --verify
openassembler run output --application test_whole_repo --backend modal
```

The build produces GPU cubins in `output/cubin/applications/test_whole_repo/`
and their readable SASS in `output/sass/applications/test_whole_repo/`. Python
training control flow and the matching PyTorch runtime remain part of execution;
this is not a standalone host binary. See the root
`docs/FRONTENDS.md#pytorch-training-applications` for configuration and coverage
details.
