# B-fragment windows with stage-pair scheduling

These six square-kernel experiments bound simultaneously live B fragments.
All four A quads are retained for each K16 slice. The output accumulation order
for each element stays unchanged; columns are consumed in increasing order and
K slices retain their original chronological order.

| Candidate | Live B x4 quads | Load/consume order | Native registers | Static non-NOP instructions |
|---|---:|---|---:|---:|
| Passing stage-pair baseline | 4 | load all eight columns, consume | 216 | 1595 |
| window2 | 1 | load two columns, consume, reuse slot | 204 | 1579 |
| window4 | 2 | load four columns, consume, reuse both slots | 208 | 1579 |
| window4 prefetch | 2 | consume pair, load next into retired slot, compute other pair | 208 | 1579 |

Every candidate retains the same 512 static HMMA, 128 LDSM.x4, and 64 LDGSTS
instructions for the stage-pair loop plus peeled tail. Reducing live values
removes 16 fragment MOVs. It also introduces more tensor epochs: native code
retains the existing required HMMA-to-observer drain before each intervening
LDSM. Compared with the baseline, window2 adds 48 drain NOPs, window4 adds 16,
and prefetch adds 32. Fewer registers and fewer load bursts can therefore lose
to additional synchronization and issue gaps. This is a measured hardware
question, not an instruction-count speed prediction.

The prefetch variant never overwrites an enclosing B quad while either half
still has a reader. Both columns are consumed by all four A rows before that
slot is reused; the other B quad remains live to feed current work. The
existing HalfBSelector tracks the enclosing quad while exposing both implicit
HMMA half-quad reads to scheduling. Source LDSM-address WAR completion edges,
the five ordinary load scoreboard IDs, copy-group waits, CTA barriers, and
tensor epoch drains remain enabled. No dependency edge is deleted.

All six variants (three schedules crossed with FP16/BF16) compile and exactly
round-trip native tensor, memory, synchronization operands and controls through
nvdisasm. Six poisoned-register SASS checks each validate all 4096 integer
outputs of a 64x64x128 CTA and output guard sentinels. A source-provenance check
compares the per-output K/column sequence using B-load addresses, independent of
register names. Functional checks do not model real tensor/load timing.

All six candidates subsequently passed eight complete-output A100 checks each,
including exact integer/zero tests. Each check compares all 16,777,216 outputs.
The maximum full-output relative L2 difference from matched cuBLAS is 3.99e-6;
BF16 full outputs match it exactly in this run. Every window candidate loses
to its stage-pair baseline in all seven shuffled timing rounds.

| Input | Baseline ms | window2 ms | window4 ms | window4 prefetch ms |
|---|---:|---:|---:|---:|
| FP16 | 0.641178 | 0.820838 | 0.696934 | 0.737587 |
| BF16 | 0.627046 | 0.794931 | 0.679373 | 0.713677 |

The decision is **no promotion**. The register and MOV reductions do not
compensate for the changed load/compute schedule under the retained
synchronization model. This experiment changes register lifetimes, load bursts,
and tensor epoch boundaries together; it does not establish that any one is the
sole cause. Further drain optimization requires a separately proven lifetime
model and hardware checks. `fragment_hardware_summary.json` retains per-worker
ratios, check counts, round directions, and source-result hashes.

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/a100_mma/phase4/fragment_windows.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/a100_mma/phase4/fragment_check.py
python3 -m modal run exp/a100_mma/modal_run.py --suite phase4/fragment_suite.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/a100_mma/phase4/fragment_summary.py
```

The hardware suite uses two same-worker groups: a validated stage-pair baseline
plus all three window variants for each precision at 4096x4096x4096. It requires
eight complete-output checks per method before shuffled graph timing. Cubins,
PTX, disassembly, native word/control checks, and provenance are retained under
`phase4/build/fragment_*`; functional evidence is `fragment_semantics.json`.
All helper caches and discovery writes are redirected to phase4 before encoder
construction. Production and earlier research files are not modified.
