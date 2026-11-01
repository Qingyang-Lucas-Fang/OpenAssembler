"""openassembler.verify: the generic correctness checker (any kernel signature,
not just the probe corpus's fixed out/in/k/kf ABI). Exercised here on a
pyfront kernel, since that's exactly the case the original probe-only
checker couldn't handle (the CLI's own --verify picks this automatically
for non-probe-shaped kernels; see openassembler/cli.py:_verify_compiled)."""

from __future__ import annotations

from openassembler.backend import ra
from openassembler.ptx.parse import parse_module
from openassembler.pyfront import Array, compile_kernel, f32, i32, kernel
from openassembler.verify import run_and_diff


@kernel
def vec_add(a: Array(f32), b: Array(f32), out: Array(f32), n: i32):
    i = tid_x() + ctaid_x() * ntid_x()
    if i < n:
        out[i] = a[i] + b[i]


def test_generic_verify_bit_exact_on_pyfront_kernel():
    mod = parse_module(compile_kernel(vec_add))
    code, stats = ra.compile_kernel(mod.kernels["vec_add"])
    mismatches = run_and_diff(mod, "vec_add", code, stats["labels"], n_elems=32, seed=0)
    assert mismatches == []


def test_generic_verify_catches_a_real_mismatch():
    """Sanity check on the checker itself: corrupt one instruction's opcode
    (ld -> different op) and confirm run_and_diff actually reports it,
    rather than silently agreeing with itself."""
    mod = parse_module(compile_kernel(vec_add))
    code, stats = ra.compile_kernel(mod.kernels["vec_add"])
    # Flip an FADD to FMUL in the compiled code -- same operands, different op,
    # so results differ whenever the two addends aren't equal.
    broken = []
    flipped = False
    for text, ctl in code:
        if not flipped and text.strip().startswith("FADD"):
            text = text.replace("FADD", "FMUL", 1)
            flipped = True
        broken.append((text, ctl))
    assert flipped, "expected an FADD in vec_add's compiled code"
    mismatches = run_and_diff(mod, "vec_add", broken, stats["labels"], n_elems=32, seed=0)
    assert mismatches, "corrupting FADD->FMUL should produce at least one mismatch"
