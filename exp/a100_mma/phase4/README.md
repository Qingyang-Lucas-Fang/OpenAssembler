# SM80 async-copy L2-prefetch experiment

This research stays entirely under `phase4/`; existing compiler and phase3 files
are read only. It tests a concrete feature observed in cuBLAS SASS:
`LDGSTS.E.BYPASS.LTC128B.128`. It does not assume that copying a proprietary
kernel's modifier will improve a different access schedule.

`cache_probe.py` independently compiles matching PTX forms with no hint and
`.L2::64B`, `.L2::128B`, and `.L2::256B`. CUDA12.8 ptxas accepts all four for
`sm_80`. Their SASS differs only in a two-bit prefetch field, after normalizing
register operands and controls:

| PTX prefetch hint | SASS modifier | Bits 72:71 | Exact changed bits from no hint |
|---|---|---|---|
| absent | absent | 00 | none |
| `.L2::64B` | `.LTC64B` | 01 | 71 |
| `.L2::128B` | `.LTC128B` | 10 | 72 |
| `.L2::256B` | `.LTC256B` | 11 | 71, 72 |

All four field values and 80 randomized source/destination register and signed
displacement combinations round-trip exactly through nvdisasm. Eight functional
SASS checks (four hints, default/poisoned registers) copy every one of 512 shared
bytes correctly and preserve output gap sentinels. These checks establish
instruction encoding and modeled data semantics; they do not simulate L2 caches
or certify real async completion.

`cache_build.py` takes each proven native stage-pair cubin and applies this
two-bit postpass to its 64 static LDGSTS instructions. Every operand, control,
other instruction, register allocation, text length, and nontext section remains
exact. `cache_audit.py` independently checks the complete cubin byte difference,
not just the disassembly. Matching hinted PTX is assembled into a separate
ptxas reference cubin. Native candidates never use the ptxas instruction body.

The bounded suite has six stage-pair baselines (square, MLP, tall crossed with
FP16/BF16) and ten candidates: 128-byte hints on all six, with 64/256-byte hints
also tested on both square cases. Each baseline and its alternatives run in the
same GPU worker, with complete-output checks before graph timing. This isolates
the cache hint from instruction scheduling or register-allocation changes.

From the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/a100_mma/phase4/cache_probe.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/a100_mma/phase4/cache_check.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/a100_mma/phase4/cache_build.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/a100_mma/phase4/cache_audit.py
python3 -m modal run exp/a100_mma/modal_run.py --suite phase4/cache_suite.json
```

The unchanged Modal harness performs eight input-family/seed checks, full output
comparison to matched FP32-output cuBLAS, sampled CPU-float64 oracles, canaries,
and finite/missing-output checks before timing. Timing uses shuffled paired
20-node graphs and 16 warm-up replay rounds. The six suite cases together have
16 native methods and six matching PTX references.

All ten candidates subsequently passed their eight real A100 complete-output
checks. The isolated hint did not establish a useful speedup. For the 128-byte
hint, baseline/candidate median ratios were 1.00063/1.00174/1.00409 on FP16
square/MLP/tall and 0.99935/0.99601/0.99699 on BF16. Gains and losses are small,
and paired round directions vary. The FP16 256-byte square hint was about 0.69%
slower by medians. The conclusion is **no promotion**; cache-hint similarity to
cuBLAS alone does not explain its performance. `cache_hardware_summary.json`
records exact per-worker medians, round ratios/wins, checks and source hashes.
Reproduce the retained summary without another GPU run:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/a100_mma/phase4/cache_summary.py
```

NVIDIA's [PTX cp.async specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async)
describes these qualifiers as hints for additional L2 fetches. They do not alter
memory consistency and do not replace commit, wait-group, or CTA synchronization.
The cached official specification used here is `docs/research/.ptx.txt`, section
9.7.10.28.3.1: prefetch hints were introduced in PTX7.4 and require SM80 or later.
Global operands remain actual global pointers with the same alignment and bounds.

Other cuBLAS observations remain separate hypotheses. Its `.CONSTANT` modifier
does not imply a reusable input contract for arbitrary training data; no bits
for that modifier are enabled here. A source `R.64+UR` form is disassemblable,
but changing to it requires constructing correct uniform state, proving address
semantics, measuring register/issue overhead, and checking independent lanes.
Neither a register-bank remap nor uniform address substitution is inferred from
L2 hints. CTA ordering can be revisited separately on the tall case whose counters
indicate memory costs. The subsequent [fragment-window experiment](fragment_README.md)
targets load bursts and register lifetimes instead.
