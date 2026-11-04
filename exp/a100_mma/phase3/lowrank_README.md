# Fused compile-time low-rank staging proof of lowering

`lowrank_staging.py` generates native OpenAssembler kernels for Strassen rank 7
and a classical rank 8 control. A single warp computes a 32x32x32 macro product.
Eight base 16x16 quadrants are loaded once and cached in registers. Compile-time
`+1/-1` transforms produce packed FP16 values, store them directly into XOR
swizzled shared memory, and immediately feed `ldmatrix`/`mma.sync`. There are no
device coefficient loads or global transform buffers. Only one 16x16 product
tile is live before FP32 reconstruction into the four output quadrants.

The packed transform selector is an experiment-local subclass; the main
compiler is not changed. `add/sub.rn.f16x2` lowers to `HADD2`, with an explicit
packed sign XOR for subtraction because the existing encoder has no negative
register-source HADD2 template. All emitted instructions are encoded by
OpenAssembler and accepted by `nvdisasm`; ptxas supplies a separate reference
cubin and the ELF container.

Rank 7 uses 14 HMMA instructions instead of 16, but its generated code has 940
static instructions and 94 registers, versus 879 instructions and 74 registers
for rank 8. These numbers expose transform/reconstruction overhead. This is a
small proof of compiler lowering and precision control, not a competitive GEMM.
The one-warp tile, fixed K=32, synchronous staging, and full-tile M/N restriction
are deliberate limits. Extending it to the direct multistage kernel requires
larger product tiles, transform reuse across products, and a resource cost model.

Both algorithms use FP16 inputs, transforms rounded once to FP16, FP32 HMMA,
ordered FP32 reconstruction, and FP32 output. Their rounding paths differ:
low-rank transforms change the finite arithmetic even when the real-arithmetic
identity is exact. The oracle explicitly models this policy. For tested
96x96x32 inputs, the rank-7 oracle differs from untransformed direct GEMM by up
to 0.00556 for random inputs and 0.02215 for the cancellation case. Matching the
output dtype alone does not make numerical policies equivalent.

Six functional SASS checks pass: rank 7/rank 8 crossed with integer, random
floating, and cancellation inputs. Every check poisons registers, checks all
1024 outputs of CTA `(1,1)`, and verifies no writes outside that tile. Integer
checks are exact; floating checks have explicit tolerances for HMMA internal
accumulation.

The A100 hardware run in
`lowrank_gpu_20261103T233948.418796Z.json` passes all twelve full-output checks:
rank 7/rank 8, OpenAssembler/ptxas, and the three input families. Each checks all
9216 outputs and both canary regions against the matched quantization policy.
In the same worker, native rank 7/rank 8 medians were 0.007136/0.007232 ms;
ptxas medians were 0.003296/0.003328 ms. Ratios of 1.013/1.010 are small and
this 96x96x32 workload has only nine one-warp CTAs. Treat the result as a proof
of native lowering and numerical validation near the launch floor. It provides
no evidence of a competitive large-GEMM algorithm or transformer-step speedup.

From the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/a100_mma/phase3/lowrank_staging.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/a100_mma/phase3/lowrank_check.py
python3 -m modal run exp/a100_mma/phase3/lowrank_modal.py
```

The standalone Modal worker runs rank 7 and rank 8, both OpenAssembler and
ptxas, on identical A100 inputs. It checks all 9216 outputs and both guard
regions for each of three input families, against the corresponding matched
quantization oracle. Only after all twelve checks pass does it time identical
32-node graphs in seven shuffled rounds on the same worker. Results and binary
hashes are saved as `lowrank_gpu_<timestamp>.json`. No whole-step or large-GEMM
speedup should be inferred from this microprototype.
