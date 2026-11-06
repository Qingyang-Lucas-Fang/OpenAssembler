<p align="center"><img src="figs/openassembler_logo.png" width="680" alt="OpenAssembler"></p>

# OpenAssembler

OpenAssembler compiles supported Python GPU kernels, CUDA C/C++ kernels and PTX
into native NVIDIA SASS and loadable CUDA binaries. Its own instruction selector,
register allocator, scheduler and encoder target **A100 / SM80**.

It uses NVIDIA tools as black-box encoding oracles, for CUDA C/C++ to PTX,
for the cubin container and for reference disassembly. A separate configured
PyTorch application workflow uses Dynamo/AOTAutograd and Inductor/Triton to
capture and replay model graphs. That workflow does not automatically replace
its kernels with the native OpenAssembler backend.

[Architecture and status](docs/STATUS.md) · [All commands](docs/COMMANDS.md) ·
[Research catalogue](docs/research/README.md) · [Development setup](docs/DEVELOPMENT.md)

## What you can do

- Compile standalone kernels or discover supported kernels in a repository.
- Generate full-tile FP16/BF16 GEMMs with FP32 output, with explicit Morton,
  fused-epilogue and pointer-home options.
- Generate cooperative FP32 softmax and population LayerNorm with row/column tails.
- Inspect cubins and SASS, emulate supported instructions, and run matched GPU
  checks locally, over SSH or on Modal.
- Capture a configured PyTorch application's bounded workload and replay its
  saved compiler/runtime cache with state verification.
- Study factorized matrix multiplication, compiler encodings and scheduling
  through the retained research tools and evidence.

Support and performance are workload-specific. The [status page](docs/STATUS.md)
separates native compiler results, library controls and application-frontend
results. The [research catalogue](docs/research/README.md) includes successful,
rejected, mixed and untested ideas, each with its own methods and reproduction.

## Quick start

Use Python **3.12 or newer**:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Compilation needs CUDA toolkit tools, not a local GPU. On Linux an installed
CUDA toolkit can be used. On macOS the toolkit runs in a Colima Linux VM:

```sh
# macOS only: install Colima if necessary, then start its VM.
colima start
# Download the pinned tools and check their version.
env/fetch_cuda.sh
env/cuda.sh ptxas --version
```

See [development setup](docs/DEVELOPMENT.md#toolkit-and-host-requirements) for
platform details, optional dependencies and an existing-toolkit installation.

Compile a kernel and inspect the intended launch without sending a GPU job:

```sh
openassembler compile examples/vector_add.py --out-dir output/vector
openassembler run output/vector --dry-run
```

Generate a native tensor kernel or cooperative reduction:

```sh
openassembler gemm --m 1024 --n 1024 --k 1024 --dtype f16 --out-dir output/gemm
openassembler rowwise --op softmax --rows 1024 --width 512 --out-dir output/softmax
```

A configured application selects its own entry point and workload:

```sh
openassembler compile testcases/test_whole_repo --verify --dry-run
```

To execute native outputs on an authenticated Modal GPU:

```sh
openassembler run output/gemm --backend modal --budget 180 --job-timeout 60
```

Run `openassembler` without arguments for the interactive shell, or use
`openassembler --help` and `openassembler COMMAND --help` for one-shot commands.
The [complete command reference](docs/COMMANDS.md) explains every command and
flag. Actual research reproduction belongs to the
[individual idea dossiers](docs/research/README.md), not this introduction.

## Repository map

- `openassembler/`: frontends, native backends, binary support and GPU/application runtime.
- `specs/`, `corpus/`, `probes/`, `tools/`: encoding data, test inputs and research tooling.
- `tests/`, `examples/`, `testcases/`: regression tests and executable examples.
- `docs/`: architecture, command reference, per-idea research and retained evidence.
- `exp/`: isolated experiment drivers and raw records; no production import dependency.

The package is version **0.2.0**. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
