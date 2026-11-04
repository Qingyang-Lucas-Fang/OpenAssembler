# SM80 tensor sequence and register search

These experiments are isolated under `exp/`; no production compiler source is
changed. `search_sequences.py` generates PTX and installs scoped selector and
allocator overrides. Every executable instruction in the tested cubin comes
from OpenAssembler. NVIDIA ptxas supplies the comparison binary and ELF
container only.

## Candidate mechanisms

* **Paired B fragments:** one `LDSM.16.M88.4` replaces two B x2 loads. Lanes
  16–31 supply addresses for N+8; `HalfBSelector` consumes each aligned half of
  the result directly as HMMA's two-register B operand. Both physical source
  reads remain explicit in the dependency IR. Pairing alone adds one live
  address and moves register use from 228 to 232.
* **Paired B with precomputed addresses:** the lane16 adjustment is folded
  into the existing invariant B shared addresses in the prologue. This avoids
  per-K address additions and retains 228 registers: 1786 non-NOP instructions
  against the baseline's 1812 (1794 versus 1820 including tensor epoch NOPs,
  before trailing container padding). B LDSM instruction count is halved. These are
  structural facts, not a measured throughput improvement.
* **Next unrolled K-slice prefetch:** after the last use of each current A
  row, load its next slice into the retired fragment group. Prefetch next B
  fragments into an alternate register set while other current A rows still
  compute. This stays inside one shared stage; it does not speculate across
  `cp.async` completion or CTA barriers. Register use grows to 246, or 248
  with paired B. Per-output accumulation order is unchanged.
* **Joint layout/traversal:** column/row traversal, B pairing, and asymmetric
  A/B shared swizzle masks are searched together. The chosen swizzles are the
  existing validated address maps, with global-to-shared staging and fragment
  access changed consistently.
* **Register allocation:** two tensor-role preferences and two complete-quad
  permutations explore operand placement. R1 is fixed for the ABI; implicit
  register pairs/quads remain contiguous/aligned. A liveness checker rejects
  overlapping values mapped to one physical register. Preferences are
  empirical search parameters, not an assertion of an undocumented RF bank
  model. No spilling is introduced.
* **Scheduling controls:** HMMA issue gap 4/8 and reuse enabled/disabled are
  compared without weakening any proven completion edge. The epoch drain,
  shared-stage synchronization and scoreboard recycling protections remain.

| Candidate | Non-NOP instructions | Scheduled instructions including epoch NOPs | Registers | LDSM x4/x2 |
|---|---:|---:|---:|---:|
| Baseline | 1812 | 1820 | 228 | 32/64 |
| Paired B | 1792 | 1800 | 232 | 64/0 |
| Paired B, precomputed addresses | 1786 | 1794 | 228 | 64/0 |
| Next K-slice overlap | 1812 | 1838 | 246 | 32/64 |
| Paired B plus overlap | 1792 | 1818 | 248 | 64/0 |

Counts include both the steady-loop body and peeled tail in the binary; all
retain 256 HMMA instructions. Interleaving loads creates more tensor epochs,
so the conservative drain introduces extra NOPs in the overlap variants. The
tests deliberately measure this implementation cost rather than removing a
proven completion edge to improve the static count.

## Validation and scope

`search_test.py` tests full 64×64×128 outputs against an exact integer CPU
oracle for f16 and bf16, including every output a second time with physical
registers poisoned. It also checks aligned-half dependency declarations and
the allocator's ABI, group and liveness invariants. All 17 tests passed in
60.42 seconds, including the precomputed-address extension.

Builds disassemble through nvdisasm. The current builder additionally verifies
that cubin instruction words equal the OA encoder's literal output, every
control word is valid and unchanged, and every tensor/memory operand round
trips exactly after removing nvdisasm's `.reuse` annotation. Scalar IMAD
aliases printed by nvdisasm generate the existing CLI warning; the alias is
not treated as evidence of changed arithmetic. `search_audit.py` performs the
same exact word/control/operand comparison on retained builds.

`search_smoke_suite.json` and `search_large_suite.json` use the existing
same-worker full-output GPU harness. Both contain a baseline and all
alternatives for each dtype, with 16 steady warmup rounds. Hardware results
must establish correctness and performance before any candidate is promoted.
No GPU job is launched by these scripts.

## Measured A100 results

The root task executed both suites on an A100-SXM4-80GB (108 SMs). All 30
smoke candidates and all 30 large candidates passed the GPU harness. The
large cases each contain 136 check rows (all methods across eight pattern
and seed combinations), checking all 16,777,216 output elements. The raw
records are `results/search_large_f16.json` and
`results/search_large_bf16.json`; `search_gpu_summary.json` retains their
hashes, timing samples and numerical summary.

4096×4096×4096, FP16/BF16 inputs and FP32 accumulation/output:

| Candidate | FP16 milliseconds | FP16 latency reduction | BF16 milliseconds | BF16 latency reduction |
|---|---:|---:|---:|---:|
| Scalar-epilogue baseline | 0.787763 | — | 0.766310 | — |
| Paired B | 0.710656 | 9.788% | 0.695552 | 9.234% |
| Paired B, precomputed addresses | 0.756736 | 3.939% | 0.739994 | 3.434% |
| Paired B plus role1 allocation | 0.704666 | 10.549% | 0.699494 | 8.719% |
| Paired B, issue gap4 | 0.708250 | 10.094% | 0.695603 | 9.227% |
| Next K-slice overlap | 0.762419 | 3.217% | 0.758528 | 1.016% |
| Paired B plus overlap | 0.745318 | 5.388% | 0.736102 | 3.942% |
| ptxas, matched baseline PTX | 0.632218 | 19.745% | 0.625715 | 18.347% |
| cuBLAS | 0.554394 | 29.624% | 0.542515 | 29.204% |

These percentages are reductions in latency from the baseline, not changes
in TFLOP/s. Results use seven shuffled timing rounds with 20 graph nodes and
16 steady warmups. The paired-B mechanism is the main measured improvement.
Removing address instructions made the precomputed-address variant slower
than ordinary pairing: instruction count did not predict throughput. Role1
helps FP16 slightly but loses to ordinary pairing for BF16; it is not a
universal register placement rule. Likewise, the overlap variants incur
live-register and epoch-drain costs and lose to ordinary pairing.

The six-CTA smoke shape is useful for correctness, not compute-bound kernel
selection. On its FP16 run, pairing reduced 18.176 to 17.664 microseconds;
that small workload includes substantial launch/prologue/epilogue cost.

The next measurement is composition with the row epilogue on square, tall
and MLP projection shapes in both dtypes. `search_combined.py` creates only
new combined artifacts and reuses the independently retained baseline/row
builds. Its suite includes baseline, row alone, pairing alone, row+pairing,
row+pairing+role1 and row+pairing+gap4 in the same worker. Counters are still
needed to attribute the paired-load gain and preaddress regression to tensor
issue, operand collection, shared traffic or scoreboard behavior.

Reproduce with:

```sh
.venv/bin/python -m pytest exp/a100_mma/phase3/search_test.py -q -o cache_dir=exp/a100_mma/phase3/search_pytest_cache
.venv/bin/python exp/a100_mma/phase3/search_sequences.py
.venv/bin/python exp/a100_mma/phase3/search_sequences.py --large
.venv/bin/python exp/a100_mma/phase3/search_audit.py search_smoke_suite.json
.venv/bin/python -m modal run exp/a100_mma/modal_run.py --suite phase3/search_smoke_suite.json
.venv/bin/python -m modal run exp/a100_mma/modal_run.py --suite phase3/search_large_suite.json
```

No model has been used to remove a tensor dependency, shorten a shared stage
wait, omit an address WAR edge, or infer numerical equivalence from reduced
instruction count. There is no claim yet of cuBLAS parity or faster training.
