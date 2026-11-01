"""openassembler.pyfront: a from-scratch, minimal Python-subset compiler to PTX.

Not a general Python compiler -- a small, explicitly scoped numeric subset
(Numba @cuda.jit-lite), enough to write elementwise ops, reductions, and
tiled matmuls by hand with Python syntax instead of raw PTX text:

    @kernel
    def add(a: Array(f32), b: Array(f32), out: Array(f32), n: i32):
        i = tid_x() + ctaid_x() * ntid_x()
        if i < n:
            out[i] = a[i] + b[i]

`compile_kernel(fn)` returns PTX source text, which goes through the exact
same openassembler.backend.ra / openassembler.container.cubin pipeline as any
other PTX in this repo -- this module only replaces writing PTX by hand, not
anything downstream of it.

Supported: typed params (Array(f32|i32|u32), i32/u32/f32 scalars), local
int/float variables, for i in range(...) with a constant or thread-index
expression bound, if/elif/else, +-*/ and comparisons on scalars, indexed
array reads/writes, and the intrinsics tid_x/ctaid_x/ntid_x (and _y/_z).
Anything outside that raises NotSupported with the offending node, the same
"never guess" policy as openassembler.backend.ra.
"""

from __future__ import annotations

from openassembler.pyfront.types import Array, f32, i32, u32
from openassembler.pyfront.compile import NotSupported, compile_kernel, kernel

__all__ = ["Array", "f32", "i32", "u32", "NotSupported", "compile_kernel", "kernel"]
