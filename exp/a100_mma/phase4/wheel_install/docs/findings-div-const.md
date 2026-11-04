# ptxas 12.8: integer division by a constant

Source: `tools/div_const_survey.py` → `corpus/sm_80/div_const_survey.json`
(17 divisors × {div, rem} × {u32, s32, u64, s64}, plus loop variants), with spot
checks on sm_90.

## Result

ptxas **never** strength-reduces division by an immediate. Every divisor, including
2 and 16, gets the same run-time reciprocal sequence it uses for a variable divisor:

| Type | Lowering of `div x, C` | Instructions past the baseline probe |
|---|---|---|
| u32 | I2F.U32.RP C → MUFU.RCP → F2I → Newton step → IMAD.HI → two fix-ups | 13 (3 on the XU pipe) |
| s32 | same plus IABS / sign fix-up | 17 |
| u64 / s64 | CALL `__internal_0_$__cuda_sm20_div_u64` (≈60 insns, also reciprocal based) | 61 / 70 |

- `div.u32 x, 2` on sm_80 compiles to I2F + MUFU.RCP + F2I + 2×IMAD.HI + …; sm_90 is the same.
- In loops, ptxas hoists the 6-instruction reciprocal setup, so each iteration costs
  IMAD.HI + IMAD + 2×ISETP + 3 predicated IADD3. Magic-number division would be
  IMAD.HI + SHF (+ IMAD for rem).
- `rem.u32` by 2 or 16 inside a loop *is* reduced (3 XU ops drop to 0). That's
  probably an induction-variable rewrite, not divisor analysis.

## How much this matters (honestly)

NVVM (nvcc) and LLVM's NVPTX backend (Triton, clang CUDA) already turn `x / C` into
multiply-high **before** emitting PTX. So the gap only shows up when:
1. PTX is hand-written or comes from a frontend that doesn't do this (JITs, DSLs,
   inline asm), or
2. the divisor only becomes a constant inside ptxas (after inlining or constant
   propagation). That's untested so far, and it's the next experiment.

The win per site in our own compiler is about 13 → 2–3 instructions, and 3 → 0 on the
XU pipe (MUFU throughput is 1/4 of the ALU on sm_80). Each magic-number replacement
gets proved correct by SMT (z3, still to be installed) before we rely on it.
