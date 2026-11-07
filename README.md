<p align="center"><img src="figs/openassembler_logo.png" width="680" alt="OpenAssembler"></p>

# OpenAssembler


**A from-scratch Python/CUDA to SASS/CUDA Binary compiler for your Neural Network training repo, built entirely by reverse-engineering nvidia tools as a black box and applying kernel optimizations to the compiler.**

Do you always feel like when the newest nvidia GPU come out with unbearable high price while your existing older GPUs are making you left behind in model serving and training? 

We provide a more efficient compiler eith optimization kernels that reduce the number of instruction down by up to ~12% comparing to the NVIDIA ptxas compiler, and reach an approximate 13% speed up in LLM inferencing and training comparing to using the SASS/cubin compiled by the nvidia compiler on various existing architectures (A100, H100 etc). Our goal is to create efficient and fast compiler and low level assembly language/binary file so that you can (without buying expensive frontier GPUs) with a ~10% speedup in training speed. OpenAssembler currently support compiling and executing whole NN training repo on your A100 GPU (including Python, CUDA C, CUDA C++ down to SASS and further down to CUDA Binary file, which is directly executable on your NVIDIA GPUs). We are currently working on more optimization kernels and wider hardware supports including V100 and H100. 

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

## Star History

<a href="https://www.star-history.com/?repos=qingyang-lucas-fang%2Fopenassembler&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=qingyang-lucas-fang/openassembler&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=qingyang-lucas-fang/openassembler&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=qingyang-lucas-fang/openassembler&type=date&legend=top-left" />
 </picture>
</a>