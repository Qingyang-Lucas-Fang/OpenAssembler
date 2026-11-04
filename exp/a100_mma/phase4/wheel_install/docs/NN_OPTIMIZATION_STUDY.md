# NN and transformer kernel study

This is a baseline study for OpenAssembler 0.2.0. It deliberately measures the
current compiler before applying aggressive optimizations. The study does not
change scheduling, fusion, tiling, or unrolling passes.

Run it with:

```sh
python tools/nn_study.py --dry-run
python tools/nn_study.py --backend modal --budget 180 --rounds 3 --reps 20
```

The script compiles `examples/nn_ops.py`, builds explicit buffer/launch specs,
shuffles our and ptxas jobs across three rounds, compares requested output buffers,
and writes a timestamped report under `output/nn-study/runs/`. The report stores
the source hash, manifest, exact specs, device string, timings, and mismatches.
The comparison is byte exact by default; an approximate neural kernel that differs
by one or more f32 ulps is reported as a mismatch rather than silently accepted.

The first recorded run used an A100-SXM4-80GB through Modal on **November 2, 2026**,
with 128-row/128-column elementwise and reduction cases, 32×32×64 matmul, three
shuffled rounds, and 20 timing repetitions. The full specs, source hash, per-round mismatch details,
and SASS instruction/register counts are preserved in [the raw study record](evidence/nn-study-a100-20261102.json). Results are the best median per kernel:

| Kernel | Exact output | Ours ms | ptxas ms | ptxas / ours |
|---|---:|---:|---:|---:|
| ReLU | yes | 0.0102 | 0.0113 | 1.10× |
| GELU | no | 0.0113 | 0.0102 | 0.91× |
| Softmax | yes | 0.1024 | 0.0809 | 0.79× |
| LayerNorm | no | 0.0922 | 0.0717 | 0.78× |
| Adam update | no | 0.0123 | 0.0102 | 0.83× |
| Cross entropy | no | 0.0584 | 0.0410 | 0.70× |
| Naive matmul | no | 0.0287 | 0.0143 | 0.50× |
| Attention scores | no | 0.0246 | 0.0195 | 0.79× |

The mismatches were small f32 bit differences for the current inputs (1–3 integer
steps in the stored word in these examples), but bit differences still matter for
a compiler regression. Disassembly shows the Python source emitted separate `FMUL`
and `FADD` operations where the ptxas build often contracts operations to `FFMA`;
that is a plausible contributor to some differences, but it does not explain every
case without a per-kernel trace. Do not treat timings for mismatching kernels as a
like-for-like performance result. They identify two separate
workstreams:

- The frontend/backend arithmetic path needs a documented numerical comparison
  policy for approximate math and a tighter validation matrix for f32 reduction
  order. Bitwise comparison remains the default for compiler correctness.
- Performance needs kernel-aware transformations. The initial candidates are FFMA
  formation (`mul + add`), software pipelining/unrolling for reductions, invariant
  index hoisting, tiled shared-memory matmul, and fused attention/normalization
  epilogues. These changes can alter rounding, register pressure, shared-memory
  use, and occupancy.

The current evidence supports measuring and reviewing those changes before they are
implemented. The results do not support a blanket claim about LLM or training speed.

The eight generated kernels contain these SASS opcode families and counts (combined
OpenAssembler listing through the first `EXIT` in each kernel; aliases are grouped
by base opcode, and post-`EXIT` container padding is excluded):

| Opcode | Count | Opcode | Count | Opcode | Count |
|---|---:|---|---:|---|---:|
| MOV | 233 | IADD3 | 158 | IMAD | 137 |
| BRA | 116 | LOP3 | 91 | ISETP | 79 |
| FFMA | 73 | FMUL | 37 | FSETP | 36 |
| FADD | 31 | SHF | 28 | MUFU | 25 |
| LDG | 22 | PLOP3 | 20 | S2R | 16 |
| IABS | 12 | SEL | 11 | STG | 10 |
| ULDC | 8 | EXIT | 8 | I2F | 7 |
| F2I | 4 | FCHK | 4 | LEA | 4 |

This baseline exercises addressing/control flow, integer index arithmetic, floating
point reductions, approximate transcendental operations, and exact/approximate
arithmetic helpers. The kernels are deliberately scalar and do not exercise tensor
cores, warp collectives, or atomics. The ptxas-produced GELU sequence contracts a
multiply-plus-add into `FFMA`; the current frontend/backend sequence preserves the
separate operations, which is consistent with the observed few-ULP GELU difference.
That finding motivates an explicit fast-math experiment; contraction should not be
silently enabled under Python's current per-operation rounding semantics.

## Optimization candidates and risks

The data supports these experiments, in this order:

| Candidate | Expected mechanism | Main risk |
|---|---|---|
| Optional fast-math FFMA formation | Remove an instruction and shorten multiply/add dependency chains; can reduce rounding steps | Changes Python's per-operation f32 rounding and can change NaN/signed-zero behavior; must stay opt-in and be tested against tolerances |
| Counted-loop software pipelining and unrolling | Overlap independent loads and arithmetic, reduce branch/index work per element | More live registers, spills, altered reduction order, and poor behavior for short tails |
| Loop-invariant index/address hoisting | Reduce repeated IMAD/LEA work inside matmul/reduction loops | Code growth and register pressure; requires proof that dimensions and bases are invariant |
| Tiled shared-memory matmul | Reuse A/B tiles across outputs and reduce global memory traffic; the measured naive matmul is 2× slower than ptxas | Barrier correctness, bank conflicts, occupancy loss, tile-tail handling, and substantial numerical-order changes |
| Fused softmax/attention epilogues | Avoid intermediate global-memory writes/reads and extra launches | Shared-memory pressure, register usage, different approximations, and more complex launch constraints |

No expected speedup percentage is assigned yet: these kernels are not bit-exact
with ptxas, and the matrix shapes are small correctness baselines rather than a
representative training workload. The next implementation step should first add
an explicit fast-math mode plus numerical tolerances, then test FMA formation and
loop pipelining on separately recorded exact and approximate workloads. Tiling and
fusion should follow only after those checks and a larger shape sweep.
