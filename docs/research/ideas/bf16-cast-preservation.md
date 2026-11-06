# Preserving intermediate casts as a BF16 repair attempt

Status: tested-not-selected

Retaining eager-like intermediate low-precision rounding did not repair the
original BF16 parameter gate. The option is not enabled in production, and
its failed-run timing is not an admitted speedup.

## Hypothesis and test

Graph fusion can remove a downcast followed by an upcast between operations.
Even when this improves numerical accuracy against an ideal reference, the
result may differ from the exact rounding sequence of eager execution.
Pinned PyTorch 2.7.1 provides `emulate_precision_casts` to retain that rounding.
The experiment asks whether this alone resolves the observed multi-step
key-bias mismatch without changing the optimizer or tolerance.

The AOTAutograd/Inductor/Triton application route is used; these are not native
OpenAssembler MMA SASS measurements. Flag 0/1 runs use fresh eager/compiled
processes and separate caches on the same actual **A100-SXM4-80GB**.
Seed 1729, model/data, BF16 autocast, TF32 prohibition,29 updates, AdamW policy
and all 127-tensor gates remain fixed. The eager snapshot SHA is identical in
both controls. The structured verifier agrees with an independent original
assertion loop.

## Findings

| Setting | Failed parameter tensors | Failed bias entries | Largest absolute drift |
|---|---:|---:|---:|
| Original casts, flag 0 | 2 | 6 | 0.0071577709 |
| Preserve casts, flag 1 | 1 | 5 | 0.0071797873 |

All failures are key-bias entries. With preservation, layer-zero still fails
indices 521/600/728/965/974. Fewer failing tensors is not a pass, and maximum
layer-zero drift does not improve. Neither a close loss nor a fast runtime
overrides a failed parameter comparison.

The compiled step median changes 2.96346→3.03821 ms, approximately 2.52% slower
in this single paired process-order run. Forward/loss, backward and optimizer
medians also rise. This is not a repeated performance estimate; both settings
fail numerical admission. The result establishes only that preserving these
casts is insufficient, not that the option is universally useless.

The original worker/model/harness hashes are retained. The actual raw metadata
says 80 GB; an earlier duplicate prose report's40 GB label was incorrect. Device
labels in the raw record take precedence over the requested Modal tier.

## Reproduce

Run from the repository root with an authenticated Modal client. The driver
uses the pinned 2.7.1/CUDA 12.8 image and bounds the GPU function to 900 seconds:

```sh
modal run exp/weaknesses/bf16_precision_casts.py
# Optional order-sensitivity repeat, saved as a new run:
modal run exp/weaknesses/bf16_precision_casts.py --reverse
python3.13 -m pytest -q tests/test_torchfront_numerics.py
```

Read the `verification` status for every setting; successful execution/report
serialization does not mean the training gate passed. Do not compare this
80 GB process timing against an absolute 40 GB timing as a speedup.

## Source, evidence and next questions

- [Experiment, exact configuration and primary-source reference](../../../exp/weaknesses/bf16_precision_casts.py).
- [Complete raw result](../../../exp/weaknesses/bf16_precision_casts_20261106T131714.328188Z.json).
- [Verifier](../../../openassembler/torchfront/worker.py),
  [diagnostic contract](state-verification-diagnostics.md),
  [independent epsilon sensitivity](adam-epsilon-sensitivity.md).

Remaining work needs to isolate reduction ordering, raw gradient histories and
optimizer amplification under the same policy. Enabling a knob that still
fails, loosening the gate, deleting key-bias parameters or changing epsilon
would not establish a repair of the original compiler contract.
