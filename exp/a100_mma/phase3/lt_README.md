# Matched cuBLASLt heuristic baseline

`lt_modal.py` benchmarks public cuBLASLt heuristic configurations against the
existing FP32-output `cublasGemmEx` policy. This is library autotuning through
public APIs. It does not extract or reimplement proprietary kernels.

The default run requests 32 heuristics with both zero and 32 MiB workspace caps
for each of FP16 and BF16 inputs at 4096x4096x4096. Each dtype has a fresh worker
process on the same A100 container. FP32 accumulation, scale, and output are
explicit (`CUBLAS_COMPUTE_32F`, `CUDA_R_32F`, alpha=1, beta=0). Physical layouts
are column-major `C^T[N,M]`, Lt operand A=`Bstorage[K,N]` with transpose, and Lt
operand B=`Astorage[K,M]` without transpose. Thus logical output is `A @ B.T`.

`lt_abi.py` checks relevant enums and the complete algorithm/heuristic structure
declarations against the installed NVIDIA header before calling cuBLASLt. It
records the header path/hash, 64-byte algorithm and 96-byte heuristic ABI, raw
algorithm bytes, algorithm ID, tile ID/name, stage ID/name, split K, reduction,
CTA swizzle, custom option, required workspace, heuristic rank, and wave count.
AlgoCheck rejects unsupported candidates. Duplicate opaque configurations are
timed once while retaining both workspace-query rankings. Fewer than 32 results
can be returned; this is not an exhaustive search of all possible algorithms.

Every accepted unique configuration is checked across zero, integer, normal,
and scaled inputs. All 16,777,216 outputs are compared with matched GemmEx;
integers and zeros require exact results. Floating outputs use relative L2
tolerance 2e-5, and a 32x32 CPU-float64 oracle samples the quantized inputs.
Guard regions and missing/nonfinite outputs are checked. Candidates failing
correctness or graph capture are retained as rejected evidence and not timed.
`passed` means the sweep found a valid timed Lt candidate; the separate
`all_executed_candidate_checks_passed` field exposes any candidate failures.

Passing candidates and GemmEx use identical operands and 20-node CUDA graphs,
eight warm-up replay rounds, and seven shuffled timing rounds within the same
process. Results include all timing samples, round orders, and the best valid
configuration for each workspace cap. A selected minimum across candidates
should be confirmed in an independent paired run before claiming a small gain.

The saved A100-SXM4-40GB run
`lt_results_20261103T235453.187530Z.json` uses cuBLAS version 120803 and passes
the independent audit in `lt_audit.json`. The zero/32-MiB queries returned 6/5
FP16 and 8/8 BF16 configurations, despite requesting 32. After deduplication,
there were 10 FP16 and 13 BF16 algorithms. All 48 FP16 and 60 BF16 complete-output
checks pass, including the native OpenAssembler controls. Output canaries,
finite-value checks, and exact zero/integer cases all pass. Maximum full-output
relative L2 versus GemmEx is 4.06e-6; the maximum CPU-float64 sample relative L2
is 6.65e-6. There are no worker warnings, graph rejections, or attribute-read
errors in this run.

| Input | GemmEx ms | Best zero-cap query ms | Best 32-MiB-cap query ms | Native row-order/B-pair ms | Native / best Lt median |
|---|---:|---:|---:|---:|---:|
| FP16 | 0.596070 | 0.576461 | 0.588288 | 0.671693 | 1.16520 |
| BF16 | 0.574515 | 0.563917 | 0.563917 | 0.655462 | 1.16234 |

These native controls run in the same process and graph rounds as the library
controls. Native time exceeds selected Lt time in all seven rounds for both
types. The ratios above divide medians; median per-round native/Lt ratios are
1.17762 and 1.17079. These are separate descriptive statistics.

The selected FP16 configuration is algorithm 21, tile `256x128` (ID 24), stage
enum `32x3` (ID 9), split K 1, reduction scheme 0, CTA swizzle 0, and zero
workspace. The selected BF16 configuration is algorithm 6, tile `128x128`
(ID 20), stage enum `32x5` (ID 11), split K 1, reduction scheme 0, CTA swizzle 1,
and zero workspace. These are public algorithm attributes; they do not reveal
the complete internal SASS schedule or a portable algorithm identity across
library versions. Neither selected winner partitions K.

The best FP16 configuration returned by the 32-MiB-cap query uses split K 2,
reduction scheme 1, and only 4096 bytes of workspace. That query omits the faster
zero-cap winner. More allowed workspace cannot make the previously measured
zero-workspace algorithm infeasible: the best known feasible choice under
32 MiB is still 0.576461 ms when taking the union of both queries. The table's
32-MiB column describes that query's returned set, not the best attainable time
under its capacity. The BF16 workspace query also returns split K 2/3 candidates
requiring 2048/4096 bytes; the selected winner remains split K 1.

The selected Lt/GemmEx gains are small: ratios of GemmEx median to selected Lt
median are 1.03402 for FP16 and 1.01879 for BF16. The FP16 selection wins six of
seven rounds, and BF16 wins seven. Selection and assessment use the same seven
rounds across several candidates, so these gains need independent confirmation;
no statistical confidence interval or portable percentage improvement is
claimed. The roughly 16% native-to-selected-Lt time gap is consistently observed
within this run, and is a stronger target than comparing with default dispatch
in a different worker.

From the repository root:

```sh
python3 -m modal run exp/a100_mma/phase3/lt_modal.py
python3 -m modal run exp/a100_mma/phase3/lt_modal.py --requested 16 --dtype f16
```

An optional `--baseline-suite phase3/NAME.json` accepts existing experiment
build configurations with matching dimensions and adds native OpenAssembler
cubins to the same checks and timing rounds. The configuration uses the same
`build`, `dtype`, `m/n/k`, `tile_m/tile_n`, `threads`, and `smem` fields as
`modal_run.py`. Binary hashes and resource attributes are retained. Output files
are `lt_results_<timestamp>.json`; worker errors and watchdog failures remain
visible. Audit the saved run without another GPU invocation:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/a100_mma/phase3/lt_audit.py
```

API background: [NVIDIA cuBLASLt documentation](https://docs.nvidia.com/cuda/cublas/index.html#using-the-cublaslt-api).
The concrete ABI source is the installed CUDA header recorded in each result.
