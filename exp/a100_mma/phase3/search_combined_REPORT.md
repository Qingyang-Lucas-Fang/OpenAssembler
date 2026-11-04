# Paired B loads plus row epilogue: controlled composition

`search_combined.py` composes the measured B-fragment pairing improvement
with the independently tested row epilogue. It imports the isolated
`HalfBSelector` and `NativePairSelector` through multiple inheritance, keeping
both implicit B-register reads and both implicit FP32 store-register reads
explicit in the dependency IR. All compiler overrides are process-local;
production sources and the epilogue agent's files are preserved.

The ordinary B-pair address path is used. The precomputed-address variant is
excluded because the 4096³ hardware results made it slower than ordinary
pairing, despite its lower instruction count.

`search_combined_suite.json` contains six same-worker cases: FP16 and BF16
for square=(4096,4096,4096), MLP=(4096,16384,4096), and
tall=(16384,4096,4096), where dimensions are (M,N,K). Each case compares:

1. Scalar epilogue baseline, reused from `epilogue-confirm-suite.json`.
2. Row epilogue alone, reused from the same suite.
3. Paired B loads alone.
4. Row epilogue plus paired B loads.
5. Row epilogue plus paired B loads and role1 register placement.
6. Row epilogue plus paired B loads and HMMA issue gap4.

This is 36 OA kernels: twelve existing baseline/row kernels and 24 new
combined artifacts. All new builds passed nvdisasm checks for exact
tensor/memory/store operands, exact encoded cubin words, and every encoded
control word. No GPU job was launched by the child task.

`search_combined_test.py` passed seven tests in 40.87 seconds: the three
combined variants in both dtypes, each producing a complete 64×64×128 GEMM
against an exact integer CPU oracle and again with poisoned registers, plus
an independent IR check of the implicit register dependencies and allocation
invariants.

| Variant | Non-NOP instructions | Including epoch NOPs | Registers |
|---|---:|---:|---:|
| Scalar baseline | 1812 | 1820 | 228 |
| Row epilogue alone | 1140 | 1148 | 228 |
| B pairing alone | 1792 | 1800 | 232 |
| Row plus B pairing | 1120 | 1128 | 232 |
| Row plus B pairing plus role1 | 1120 | 1128 | 236 |
| Row plus B pairing plus gap4 | 1120 | 1128 | 232 |

The counts include the steady loop and peeled tail, before container
padding. Geometry, source hashes, source compiler hashes, opcode counts and
compiler-stage timings are in each new build's `experiment.json`. Static
counts do not establish speedup; the same-worker GPU suite measures the
actual composition and its numerical behavior.

```sh
.venv/bin/python exp/a100_mma/phase3/search_combined.py
.venv/bin/python -m pytest exp/a100_mma/phase3/search_combined_test.py -q -o cache_dir=exp/a100_mma/phase3/search_combined_pytest_cache
.venv/bin/python -m modal run exp/a100_mma/modal_run.py --suite phase3/search_combined_suite.json
```
