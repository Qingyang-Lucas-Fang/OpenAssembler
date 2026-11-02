# OpenAssembler 0.2.0: Python / CUDA C/C++ / PTX to SASS

OpenAssembler accepts GPU kernel source files and source directories. It emits
encoded sm_80 instructions in a loadable CUDA binary (`.cubin`) and a readable
SASS disassembly (`.sass`) of that same binary. A cubin is loaded and launched
through the CUDA driver; it is not a standalone host executable.

Python uses OpenAssembler's typed numeric subset. CUDA C/C++ uses `nvcc --ptx` for
parsing and NVVM optimization. PTX goes directly to our backend. Compilation still
needs `ptxas` to supply ELF metadata and a reference binary; `nvdisasm` decodes the
finished binary. Floating-point and integer-division arithmetic sequences are
recovered from CUDA 12.8 probe listings and then allocated, scheduled, and encoded
by OpenAssembler. See [REPRODUCING.md](../REPRODUCING.md) for toolkit setup.

## Compile a file or repository

```sh
openassembler compile examples/div_by_const.ptx -o out --verify
openassembler compile examples/vector_add.py -o vector_add --verify
openassembler compile examples/vector_add.cu -o vector_add --verify
openassembler compile my_repo/ --out-dir output
```

The default layout is:

```text
output/
  cubin/<relative-source-stem>.cubin
  sass/<relative-source-stem>.sass
  manifest.json
  .build/ptx/<relative-source-stem>.ptx
  .build/reference/<relative-source-stem>.cubin
  run.json                         # latest GPU report, after run/cnr
  runs/<UTC-timestamp>.json         # preserved GPU reports
```

`-o NAME` names one file's outputs, without a directory or extension. For backward
compatibility, `.cubin` or `.sass` suffixes are stripped with a notice. Use
`--out-dir DIR` to relocate all artifacts. A directory compile mirrors its source
paths and does not accept `-o`. Same-stem sources retain their source extensions;
remaining case-insensitive collisions receive a deterministic suffix.

Repository discovery uses Git's tracked and non-ignored file list when available,
otherwise a recursive walk. It recognizes `.ptx` with `.entry`, CUDA
`.cu/.cuh/.cpp/.cc/.cxx/.c` with `__global__`, and `.py` with an OpenAssembler
`@kernel` decorator. Hidden directories, environments, generated output, and
symlinks are excluded. Other files are skipped and counted. Python modules are
parsed without executing imports or top-level code. CUDA headers and dependencies
must still be visible to nvcc; `-I DIR` and `--nvcc-flag=-DNAME=value` are repeatable.

This compiles discovered GPU kernels. It does not translate arbitrary Python
applications, PyTorch graphs, shell scripts, host C++, or a repository's build
system into one GPU program. Device linkage and unsupported constructs fail with
a diagnostic. `--kernel NAME` selects an entry; without it, all entries in each
supported source are included in that source's cubin. C++ entry names may be
mangled: use `extern "C"` or the exact name listed in the diagnostic.

| Option | Meaning |
|---|---|
| `--arch sm_80` | Only the sm_80 encoder is supported |
| `--opt-level safe|mined|sched|opt` | Conservative scheduling; measured scheduling; list scheduling; existing counted-loop unrolling respectively |
| `--verify --trials 6` | Emulator versus PTX interpreter; no GPU is used |
| `--no-sass` | Explicitly omit the readable listing |
| `--dump-ptx` | Print the frontend's PTX |
| `--fallback ptxas` | Keep ptxas code only for unsupported backend/encoder operations; each fallback is labelled in the manifest and report |

Fallback never hides a verification mismatch or an internal compiler error.
Successful artifacts are published with atomic file replacements; the manifest
is updated while Ctrl-C is deferred. Earlier successful builds may remain after
a failed recompilation, but are marked failed and refused by `run`. Deleted or
no-longer-discovered sources have their manifest-owned outputs removed during a
subsequent nonempty repository build. Unlisted files are preserved. Power loss
or a filesystem error during publication is not a multi-file transaction; rebuild
before running if publication was interrupted.

## Python kernel subset

```python
from openassembler.pyfront import Array, f32, i32, kernel, tid_x, ctaid_x, ntid_x

@kernel
def saxpy(a: f32, x: Array(f32), y: Array(f32), n: i32):
    i = tid_x() + ctaid_x() * ntid_x()
    if i < n:
        y[i] = a * x[i] + y[i]
```

Supported types are `i32`, `u32`, `f32`, and arrays of those types. The subset
includes arithmetic, comparisons, `if/elif/else`, conditions with `and/or/not`,
`for range(...)` with a constant nonzero step, `while`, `break`, `continue`, early
return, array loads/stores, `float()`/`int()`, `min`/`max`/`abs`, and thread/block
coordinates along x/y/z. For loops retain Python's visible loop-variable behavior.

`/` produces f32, including when its inputs are integers. `//` and `%` support
integers and correct truncating GPU division toward Python's floor/sign rules.
Floating `//` and `%` are rejected. Arithmetic uses fixed-width GPU integers and
f32 rounding, not arbitrary-precision Python integers or Python float64.

GPU exceptions differ from Python: float division by zero produces IEEE infinity
or NaN; `int(f32)` truncates and saturates to i32 bounds, with NaN converted to zero.
Integer divide-by-zero and signed overflow follow the emitted PTX/device behavior,
not Python exceptions. Avoid those cases in portable mathematical code. Mixed
integer/float arithmetic requires explicit casts; an f32 array needs `1.0`, not `1`.

`sqrt` is correctly rounded; `exp`, `log`, `tanh`, and `rsqrt` use approximate GPU
operations. They are available from `openassembler.pyfront`; supported `math`
aliases are recognized statically. Arbitrary calls, classes, dynamic allocation,
recursion, general imports as callable helpers, atomics, and f16/f64 arithmetic
are not supported.

`shared_array(TYPE, CONSTANT_SIZE)` allocates static shared memory and
`syncthreads()` synchronizes a block. A barrier under a thread-dependent condition
is rejected conservatively. Both interpreters now cooperate across warps for
`bar.sync 0`; valid 64-thread shared-memory kernels can pass `--verify`. Divergent
barriers/deadlocks and unmodelled barrier forms still fail verification clearly.

## CUDA C/C++ and PTX

The backend supports the pointer-increment loops nvcc emits for dot products,
grid-stride SAXPY, nested loops, and naive matmul. This includes loop-carried
64-bit values, add/subtract immediates, and common pointer casts/bit operations.
Float division, `sqrtf`, f32/i32/u32 conversions, register integer division and
remainder, and predicate logic are supported. The precise supported arithmetic
forms are recorded in `openassembler/backend/ptxas_fragments.json`.

Unsupported areas include warp shuffle/vote lowering, f16/f64 arithmetic,
atomics, register spilling, and some vector/memory-ordering forms. Unsupported
instructions raise a diagnostic rather than silently approximating them. The
SASS emulator covers more instructions than the compiler can generate.

On macOS, nvcc runs in Colima. Keep projects and include directories under your
home directory, which the VM mounts. Linux uses the toolkit directly. Windows
is not supported.

## What verification proves

`compile --verify` compares generated SASS with the PTX interpreter on deterministic
random inputs and repeats with poisoned registers to expose uninitialized reads.
The generic harness treats 64-bit parameters as pointers to f32 buffers, uses
small integer dimensions, and defaults to 64 elements in one block. It cannot
infer arbitrary matrix shapes, indexing contracts, or application semantics.
The fixed `(out, in, k, kf)` probe ABI uses its dedicated harness.

Use an explicit GPU `--spec` and application references for real workloads.
A successful finite test is evidence for those inputs, not a proof for all inputs
or all CUDA synchronization behavior. Approximate operations need an appropriate
numerical tolerance in application-level tests.

Exit status: `0` means compilation and requested verification succeeded; `1`
means a build error or mismatch; `2` means compilation succeeded but the checker
could not model the kernel. Code failing verification is not published.

## Run compiled outputs or compile and run

```sh
openassembler run output --backend local --spec examples/run_spec.json
openassembler run output --backend user@gpu-host --spec examples/run_spec.json
openassembler run output --backend modal --spec examples/run_spec.json --dry-run
openassembler cnr examples/vector_add.py --backend modal --spec examples/run_spec.json
```

`run` requires `manifest.json`, `cubin/`, and `.build/reference/` from a previous
compile. It does not recompile. `cnr` runs only entries built in that invocation.
Each command launches our cubin and ptxas's reference on identical inputs,
compares requested buffers byte for byte, and reports timing. A fallback entry
is explicitly identified as comparing ptxas code with itself.

Without `--spec`, the convenience harness allocates f32 buffers, guesses scalar
dimensions, and uses a one-dimensional grid. Inspect `--dry-run`; use it only for
kernels matching those assumptions. It does not infer the meaning of an arbitrary
kernel signature. `--kernel NAME` and `--source FILE` restrict existing manifests;
`--source` resolves duplicate names from different files.

A complete spec is [examples/run_spec.json](../examples/run_spec.json). Its schema:

```json
{
  "buffers": {
    "x": {"dtype": "f32", "count": 128, "init": "random"},
    "out": {"dtype": "f32", "count": 128, "init": "zeros"}
  },
  "steps": [{
    "kernel": "copy", "source": "kernels/copy.py",
    "grid": [1, 1, 1], "block": [128, 1, 1], "smem": 0,
    "args": [{"buf": "x", "offset": 0}, {"buf": "out"}, {"i32": 128}]
  }],
  "compare": ["out"],
  "reps": 3
}
```

Buffer dtypes: `f32`, `i32`, `u32`. Initialization: `random`, `zeros`, or an explicit
list (remaining elements stay zero). Arguments: `buf` plus optional byte `offset`,
`i32`, `u32`, `u64`, or `f32`. Argument order is the kernel's declared order.
Dimensions are a positive integer or a list of up to three positive integers.
`smem` is dynamic shared-memory bytes. Steps execute sequentially and share named
buffers. `compare` defaults to all buffers; a specified list must be nonempty.
Timing repetitions execute the sequence repeatedly without resetting buffers;
use idempotent workloads when interpreting timing. Correctness is captured after
the first sequence. Missing outputs and faulted jobs are failures, never passes.

`--budget SECONDS`, `--job-timeout SECONDS`, `--gpu A100-40GB`, `--reps N`, and
`--dry-run` control execution. `-n`/`--n` controls only the convenience harness.
The runner returns nonzero for mismatches, errors, or skipped kernels. Reports
are written to `run.json` and a timestamped `runs/` file. `--save-jobs` and
`--from-results` use Python pickle for local trusted diagnostics; do not load
untrusted pickle files.
