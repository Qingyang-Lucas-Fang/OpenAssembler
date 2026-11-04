# Six-projection stage-pair confirmation

All six cases passed the GPU harness, including32 check rows per case across
four methods and eight pattern/seed combinations. Every check covers the
complete output:16,777,216 elements for square and67,108,864 for MLP/tall.
`address_stage_confirm_summary.py` records source hashes, raw timing samples,
numerical summaries and derived percentages in
`address-stage-confirm-summary.json`.

| Shape | Input | Row+B-pair,ms | Stage pair,ms | Additional latency reduction | cuBLAS,ms |
|---|---|---:|---:|---:|---:|
| Square4096³ | FP16 | 0.670822 | 0.656589 | 2.122% | 0.592230 |
| Square4096³ | BF16 | 0.650189 | 0.637082 | 2.016% | 0.560589 |
| MLP4096×16384×4096 | FP16 | 2.509158 | 2.456115 | 2.114% | 2.435533 |
| MLP4096×16384×4096 | BF16 | 2.481613 | 2.419302 | 2.511% | 2.145894 |
| Tall16384×4096×4096 | FP16 | 3.145830 | 2.720205 | 13.530% | 2.334515 |
| Tall16384×4096×4096 | BF16 | 3.081830 | 2.656051 | 13.816% | 2.135091 |

Dimensions are M×N×K. All kernels use FP32 accumulation and output. These
are same-worker comparisons for the two OA candidates; the percentages
are additional reductions from the already optimized row+B-pair kernel.
The tall-shape gain is substantially larger than square/MLP. Its cause is
unresolved; instruction count, lower register count or a cache hypothesis
alone does not establish the explanation. The generic profile wrapper is
prepared to compare the two tall FP16 kernels and their matched-PTX ptxas
references using hardware counters.

The FP16 MLP stage-pair median is0.845% above cuBLAS in this run. This is
evidence of a close result for this matched operation and shape; repeated
timing and broader contracts are needed before claiming parity. These
results do not measure a full transformer training step.

```sh
python3 -m modal run exp/a100_mma/phase3/profile_case.py --suite phase3/address-stage-confirm-suite.json --case address_stage_confirm_tall_f16 --indices 0,1
.venv/bin/python exp/a100_mma/phase3/address_stage_confirm_summary.py
```

The profile wrapper uses the actual configuration of every selected entry,
including dtype, grid, threads and dynamic shared memory. It also accepts
occupancy experiments whose alternatives use different warp arrangements;
there is no six-candidate naming or geometry assumption.
