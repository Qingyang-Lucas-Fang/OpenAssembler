"""CUDA C/C++ -> PTX, via nvcc's own frontend and NVVM optimizer (--ptx).

Unlike the SASS side of this project, we don't reverse-engineer C++ parsing
or NVVM's optimizer -- nvcc is already vendored as a black-box *oracle*
(env/cuda.sh, used everywhere else in this repo to check our own work), and
turning C++ into PTX is squarely what it's for. What we replace is
everything downstream of PTX: our own backend (openassembler.backend.ra) takes
nvcc's PTX and generates the SASS itself, same as for hand-written PTX or
openassembler.pyfront's output. See README.md "How it works" -- this is the same
"reuse the container, replace the code generator" split, just one level
earlier in the pipeline.

Kernels should be `extern "C" __global__ void name(...)` so nvcc doesn't
C++-name-mangle them -- otherwise --entry has to match the mangled symbol,
which this module does not attempt to demangle for you.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

from openassembler import oracle

ROOT = oracle.ROOT
_ENTRY = re.compile(r"\.visible\s+\.entry\s+(\w+)\s*\(")


class CppFrontendError(Exception):
    pass


def _kernel_names(ptx: str) -> list[str]:
    return _ENTRY.findall(ptx)


def compile_to_ptx(path: Path, *, entry: str | None, arch: str = "sm_80") -> tuple[str, str]:
    """Run `nvcc --ptx` on a .cu/.cpp file and return (ptx_text, kernel_name).
    On macOS nvcc runs in the colima VM, which only mounts $HOME; a source outside
    it is compiled from a scratch copy (so keep local #includes under $HOME).
    Raises CppFrontendError with nvcc's own message on a compile error, same as a
    bad .ptx file raises from the PTX parser. The intermediate .ptx goes to a
    scratch file and is removed -- nothing is written next to your source."""
    import os
    import uuid

    scratch = oracle.SCRATCH
    scratch.mkdir(parents=True, exist_ok=True)
    tag = f"cppfront_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    out_path = scratch / f"{tag}.ptx"
    staged = None
    try:
        src_arg = oracle.tool_path(path)
    except oracle.OracleError:
        # macOS, source outside $HOME: nvcc (in the VM) can't see it, so compile a
        # copy from scratch. A local `#include "foo.h"` next to the original won't
        # resolve from there -- keep such sources under $HOME.
        staged = scratch / f"{tag}{path.suffix}"
        staged.write_bytes(path.read_bytes())
        src_arg = oracle.tool_path(staged)
    try:
        r = oracle.run_tool("nvcc", "--ptx", f"-arch={arch}", "-o", oracle.tool_path(out_path),
                            src_arg, check=False)
        if r.returncode != 0:
            raise CppFrontendError(f"nvcc failed on {path}:\n{r.stderr or r.stdout}")
        ptx = out_path.read_text()
    finally:
        out_path.unlink(missing_ok=True)
        if staged is not None:
            staged.unlink(missing_ok=True)
    names = _kernel_names(ptx)
    if not names:
        raise CppFrontendError(f"nvcc produced no __global__ kernels from {path} -- "
                                "did you mark it extern \"C\" __global__?")
    if entry:
        if entry not in names:
            raise CppFrontendError(f"no kernel named {entry!r} in {path} (nvcc may have "
                                    f"C++-mangled it; use extern \"C\"); have {names}")
        return ptx, entry
    if len(names) != 1:
        raise CppFrontendError(f"{path} has {len(names)} kernels, pass --kernel NAME; have {names}")
    return ptx, names[0]
