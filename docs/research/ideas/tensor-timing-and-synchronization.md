# Clocked tensor probes and synchronization sweeps

Status: validated-not-integrated

Status applies to the stated experiment and contract, not to every possible version of the idea.

## Idea in plain language

Measure exactly when matrix operations and shared loads can safely feed the next instruction. Separately vary a synchronization field instead of assuming a scalar latency table applies to Tensor Cores.

## Findings and implementation decision

Corrected exact-SASS probes measured an approximately eight-cycle-per-operation slope for the tested dependent HMMA sequence and about 41 cycles per iteration for LDSM→LOP3→IADD3. These are elapsed costs of those sequences, not isolated instruction latency, initiation interval or peak-SM throughput.

BAR field mutations passed the limited sweep, so the experiment did not prove BAR stall six is necessary. Mutated DEPBAR threshold values 1/2/3 failed full outputs while 4/6 passed that protocol. Address-WAR removal failed earlier tests. A limited scoreboard-reuse probe passing did not justify removing retirement or establish a gain. The production compiler retains its proven HMMA8/drain15 and copy/address dependencies; a general clock-derived tensor scheduler is not implemented.

## Method and validity checks

Use explicit SASS with known register chains and clock reads, verify that generic fragment copies are absent, and vary one control at a time. Keep full-output and canary checks alongside timing. The first clock harness accidentally included copies and was rejected.

## Reproduction and command set

Run from the repository root. Use Python 3.12+ and an installed test environment. Commands that compile require the CUDA 12.8 toolkit; GPU commands additionally require an NVIDIA driver or an authenticated Modal client. Explicit skips mean a dependency/corpus is unavailable, not a passing hardware result.

```sh
.venv/bin/python -m pytest -q tests/test_backend_tensor.py tests/test_gemm_codegen.py
python3 -m json.tool docs/evidence/a100-mma/phase2/clocked-exact.json
python3 -m json.tool docs/evidence/a100-mma/phase2/summary.json
```
The first command checks the implemented safe policy; the others inspect the actual historical probe records. The original clock/mutation generator was retired, so these are not a fresh replay. A new exact-SASS probe driver is required to extend the model.

## Code and retained evidence

- [Exact corrected clock records](../../evidence/a100-mma/phase2/clocked-exact.json)
- [Synchronization sweep evidence](../../evidence/a100-mma/phase2/summary.json)
- [Current tensor controls](../../../openassembler/backend/tensor.py)
- [Dependency tests](../../../tests/test_backend_tensor.py)

## What remains open

Separate accumulator RAW latency, independent-chain issue rate, LDSM-to-HMMA delay, asynchronous visibility and barrier completion under multiple active warps. Do not weaken a proven edge based on one small passing sample.
