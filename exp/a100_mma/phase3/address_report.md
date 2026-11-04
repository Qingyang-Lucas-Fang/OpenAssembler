# LDGSTS address folding and constant-stage experiment

The A100 run measured a 2.67% FP16 / 2.12% BF16 time reduction from combining
shared-pointer folding with constant-stage specialization. Shared folding
alone was about 7% slower despite fewer registers and instructions. Complete
global-pointer collapse is blocked by a measured encoding limit. No production
source or pre-existing experimental module was edited.

`address_probe.py` records every one of the 128 single-bit flips of an existing
LDGSTS seed against nvdisasm. Bits 32..43 encode a signed 12-bit global byte
displacement, and bits 44..63 encode a signed 20-bit shared byte displacement.
The register fields remain destination bits 16..23 and source bits 24..31.
The probe also records opcode/modifier/predicate/control changes and rejected
words, rather than treating unchanged disassembly as proof of a usable field.
`address-ldgsts-fields.json` preserves this complete evidence.

`address-ldgsts-roundtrips.json` adds 89 exact multi-bit tests, including signed
boundaries and deterministic random register/offset combinations. Every
nonoperand bit remains the seed value and all controls are unchanged. The
candidate encoder accepts only unguarded `LDGSTS.E.BYPASS.128`, simple shared
and 64-bit global register addresses, 16-byte aligned displacements, and the
proven signed ranges. Uniform-register addresses, ZFILL, other widths/scopes,
odd global register-pair bases and unsupported forms are rejected.

For the 128×128×64 CTA, 128 threads copy eight chunks from each operand. Chunk
`j` advances by 16 rows; the global difference is `j*32*K` bytes, already
131,072 bytes for the first nonzero chunk at K=4096. This exceeds the global
field's [-2048,2047] range. The shared difference is `j*2048` bytes because
adding 16 rows preserves `row&7` in the existing XOR swizzle. Every shared
chunk offset fits the shared field's [-524288,524287] range.

`address_shared.py` keeps the 16 global pointer chains, but forms one shared
stage base per operand and folds the other chunk offsets into LDGSTS. This
reduces the generated address instructions and register pressure. It composes
the existing paired B loads and reused-row FP32 epilogue, with the same HMMA8
schedule, operand reuse settings, FP32 accumulation/output and two shared
stages as the reference.

`address_stage.py` consumes two K tiles per loop with stage offsets 0 and
32768 known at compile time. For each tile it retains copy completion wait,
CTA barrier, next-tile copy, current-tile compute, then CTA barrier. The final
two tiles are peeled: tile T-2 copies T-1, and the last drain performs no copy.
Only even full K-tile counts and a power-of-two shared-stage stride are
accepted. This removes stage toggles and some stage-address/counter work, at
the cost of a larger instruction footprint.

| FP16 4096³ variant | Scheduled static instructions incl. NOP | Registers | Executed instructions on one warp path |
|---|---:|---:|---:|
| Row epilogue + paired B reference | 1,128 | 232 | 18,674 |
| Shared LDGSTS immediates | 1,086 | 216 | 17,764 |
| Shared immediates + constant-stage pair | 1,595 | 216 | 17,525 |

All paths execute 8,192 HMMA, 2,048 LDSM, 1,024 LDGSTS, 128 CTA barriers and
64 output stores. These analytical instruction-path counts explain what
changed; they are not measured cycles or evidence of speed. Regenerate them
with `address_counts.py`; the output is `address-dynamic-counts.json`.

Seven checks passed in 37.65 seconds (`address-stage-test.log`): every thread,
chunk, operand and stage has the same shared-copy address; unsupported
encodings are rejected; full FP16/BF16 integer GEMMs match CPU output with
ordinary and poisoned initial registers. The K256 functional cases exercise
multiple two-tile loop iterations and both peeled tiles. LDGSTS encoded words
roundtrip exactly through nvdisasm. Full 4096³ cubin compilation additionally
checks every tensor/memory/store operand, every scheduled control and the
actual words spliced into the cubin. The emulator is functional and does not
model physical pipeline hazards or bank/stall performance.

The paired hardware suites are `address-shared-suite.json` and
`address-stage-suite.json`, each containing FP16 and BF16 4096³ cases with
16 warm-up graph rounds and the existing full-output/canary/float64-sample
validation. The latter suite compares the same row+B-pair reference, shared
folding alone and stage specialization within one worker. Builds are
`../build/address_{shared,stagepair}_square_{f16,bf16}`. References remain
`../build/search_combined_square_row_bpair_{f16,bf16}`. This subagent has not
launched GPU jobs; the parent ran the following same-worker comparison on
A100-SXM4-80GB and preserved both passing results:

| FP32 accumulation/output, 4096³ | Row + paired-B reference ms | Shared folding ms | Constant-stage pair ms | Stage-pair time reduction |
|---|---:|---:|---:|---:|
| FP16 inputs | 0.680090 | 0.727910 | 0.661965 | 2.67% |
| BF16 inputs | 0.656077 | 0.700160 | 0.642150 | 2.12% |

The immutable raw records are
`../runs/20261103T235538.328481Z/address_stage_square_f16_paired.json` and
`../runs/20261103T235538.328481Z/address_stage_square_bf16_paired.json`.
Each includes seven randomized-order graph samples per method, full-output
checks on all 16,777,216 elements for four patterns and two seeds, output
canaries and float64 sample checks, plus source/cubin hashes. Both workers
returned zero. FP16 maximum full-output relative L2 across the methods was
3.9812e-6; BF16 was zero, within the 2e-5 threshold. Integer and zero cases
passed exact comparisons. The larger stage-pair code footprint did not erase
the measured benefit in this square case. Confirmation across the six
projection cases is still needed before generalizing, and no whole-step
training improvement has been measured.

The 216-register folded kernel and 232-register reference have the same
static two-CTA resource bound for 128-thread CTAs, explaining why the lower
register count alone did not increase occupancy. The bounded follow-up in
`address_occupancy_report.md` changes warp layout jointly with folding and
reaches the relevant 128-register threshold for 256-thread CTAs.

```sh
# These references also supply the known LDGSTS seed used by address_probe.py.
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/a100_mma/phase3/epilogue_confirm.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/a100_mma/phase3/search_combined.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/a100_mma/phase3/address_probe.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/a100_mma/phase3/address_shared.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/a100_mma/phase3/address_stage.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q exp/a100_mma/phase3/address_test.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/a100_mma/phase3/address_counts.py
PYTHONDONTWRITEBYTECODE=1 python3 -m modal run exp/a100_mma/modal_run.py \
  --suite phase3/address-stage-suite.json
```
