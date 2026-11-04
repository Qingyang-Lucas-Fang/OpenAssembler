"""Minimal openassembler.pyfront example: elementwise vector add, compiled from a
small Python subset straight to SASS by our own backend -- no ptxas code
generation involved (see README.md "How it works").

    openassembler compile examples/vector_add.py -o out.cubin --dump-ptx

This isn't general Python: openassembler.pyfront accepts a small, explicitly
scoped numeric subset (typed params, for-range loops, if/else, arithmetic,
array indexing, and a few GPU intrinsics) -- see openassembler/pyfront/__init__.py
for exactly what's supported.
"""

from openassembler.pyfront import Array, f32, i32, kernel


@kernel
def vector_add(a: Array(f32), b: Array(f32), out: Array(f32), n: i32):
    i = tid_x() + ctaid_x() * ntid_x()
    if i < n:
        out[i] = a[i] + b[i]
