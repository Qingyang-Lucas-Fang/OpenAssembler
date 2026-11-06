# Scalar register dependencies, scoreboards and synchronized shuffles

Status: implemented

These are correctness fixes in the production scalar backend, exposed by the
cooperative reductions. Their validation is not a claim that merely increasing
stalls or reducing instruction count improves every kernel.

## Why the old model failed

A functional emulator can calculate the right arithmetic while missing a real
GPU timing hazard. A100 operations may still be collecting input registers or
writing results after later instructions issue. The compiler must keep values
alive and use scoreboard conditions to prevent premature reuse. A scoreboard
slot is a small dependency identifier, not a mathematical register.

The investigation found five connected defects/requirements:

1. **Do not steal a producer from an existing value.** Binding a new source
   name could retarget the previous definition even when another name still
   needed that value. The small example `f0=2; f0+=1; f1=f0; f1+=1` stored
   `(2,4)` instead of `(3,4)`. Retargeting now requires a fresh value and checks
   aliases inside packed register tuples as well as scalar names.
2. **Wait before reusing a live scoreboard slot.** Both old read/write slot
   generations are retired before new mappings are published. Selecting a read
   slot cannot erase the same instruction's newly published write dependency.
3. **Keep every in-flight reader.** A register may feed several asynchronous
   operations. Replacing a single `register→read barrier` entry lost older
   readers; the backend now retains a set and retires all aliases consistently.
4. **Protect source collection.** The tested MUFU special-math and register-delta
   SHFL shuffle forms need source write-after-read protection, so overwriting
   their input temporaries waits for collection. This was tested on hardware,
   not inferred from a printed listing alone.
5. **Implement the promised synchronization.** Full-mask PTX butterfly shuffle
   lowers to actual `WARPSYNC` followed by SHFL. Both remain ordered in the list
   scheduler, and the warp barrier drains tracked dependencies. Unsupported
   masks, modes, clamp values and predicated forms fail closed.

The WARPSYNC register-mask encoding was mined from ptxas 12.8's `-O0` shuffle
helper. Two emitted register forms established the field; nine selected
registers through R254 were checked with nvdisasm. Generated templates/maps
were promoted. No instruction bits were guessed by hand.

## Experiments, including failures

| Candidate | Observed outcome |
|---|---|
| Initial cooperative source | Four softmax shape failures; norm passed nine shapes |
| Fresh-value binding fix only | Width 1/31 repaired; large softmax still failed |
| Slot draining alone | Large softmax still failed |
| MUFU read protection alone | Large softmax still failed |
| SHFL read protection alone | Large softmax still failed |
| Every stall at least 6 | Fewer mismatches, still failed |
| Every stall 15 | Width 512 passed, width 1024 still had 2,044 wrong normal outputs |
| Protect MUFU source until its result completion | Large softmax still failed |
| Drain slots + both source protections | All six critical cases passed |
| Remove SHFL source protection from that combination | Both large cases failed again |
| Full 18 sweep, then reader sets/generation-safe retirement/actual WARPSYNC | All 18 passed |
| Promoted build, independent replay | All 18×five patterns, canaries and valid graphs passed |

Several defects coexist: a negative isolated change did not disprove a hazard.
Arithmetic and numerical gates stayed fixed. The early failed binary revisions
were not all archived, so their JSON is diagnostic history, not reproducible
positive-control evidence. Later generator snapshots and the final production
identity are retained.

## Validation and regression scope

The final production replay is A100-SXM4-80GB; the matching integrated candidate
is 40 GB. All 18 complete native streams are byte-identical, including controls,
branches and padding. The reduction families check 17,408,740 outputs with full
canaries and poisoned-after-capture graph replay. Current generator/backend/
spec fingerprints match the recorded result.

A paired A100-40GB pre-fix/current-native test of ten serial-operator cases
passes bitwise on normal/zero inputs, with unchanged register/shared/local
resources and no observed timing slowdown. Small event differences are treated
as noise. Legacy tensor hashes and direct-mode hashes also remain unchanged.
This does not establish a complete tensor-pipeline latency model.

## Reproduce

Run from the repository root. Local tests need Python 3.12+; relevant encoding
checks need the CUDA 12.8 tools. Hardware tests additionally need an authenticated
Modal client/GPU.

```sh
.venv/bin/python -m pytest -q tests/test_reduction_codegen.py \
  tests/test_backend_ops.py tests/test_backend_loops.py tests/test_backend_tensor.py
PYTHONPATH=. .venv/bin/python exp/weaknesses/reductions_warpsync_template.py
OA_REDUCTION_VARIANT=production PYTHONPATH=. .venv/bin/python exp/weaknesses/reductions_a100.py --build-only
OA_REDUCTION_VARIANT=production PYTHONPATH=. modal run exp/weaknesses/reductions_a100.py
```

The full validator first needs a prepared serial-source control; see
[softmax reproduction](cooperative-softmax.md#reproduce). To reproduce the
historical compiler regression, retain/reconstruct the actual pre-fix payload:

```sh
PYTHONPATH=. python3 exp/weaknesses/reductions_scalar_regression.py
```

This last driver checks the frozen baseline SHA before GPU submission. A fresh
current compile is not allowed to impersonate the old compiler. The original
executed driver revision is retained by hash in `reductions_history`.

## Source and evidence

- [Binding/shuffle selection](../../../openassembler/backend/ra.py),
  [scoreboard scheduler](../../../openassembler/backend/compile.py),
  [list ordering](../../../openassembler/backend/sched.py),
  [regressions](../../../tests/test_reduction_codegen.py).
- [Oracle source](../../../exp/weaknesses/reductions_warpsync.ptx),
  [oracle words/sweep](../../../exp/weaknesses/reductions_warpsync_evidence.json),
  [mining driver](../../../exp/weaknesses/reductions_warpsync_template.py).
- [Final replay](../../../exp/weaknesses/reductions_runs/a100-20261106T133842.082727Z.json),
  [production identity](../../../exp/weaknesses/reductions_production_identity.json),
  [paired scalar regression](../../../exp/weaknesses/reductions_runs/scalar-regression-20261106T134737.702713Z.json).

More aggressive schedules must preserve these demonstrated dependencies.
Removing synchronization because it makes the listing shorter is not a valid
optimization without a new semantic proof and full hardware admission.
