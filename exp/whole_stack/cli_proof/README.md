# Saved application capture/replay proof

`prove.py`, the fixture under `application/`, and `summary.json` retain an
independent execution of the real CLI workflow: capture and saved replay both
returned zero, 77 state tensors were checked including 15 gradients, 35 cubins
were labeled Inductor/Triton, and replay performed five optimizer updates with
zero new GPU code generation.

This is an integration/correctness proof, not the main latency estimate and
not native OpenAssembler SASS attribution. The canonical configuration,
commands and limitations are in the
[application-capture dossier](../../../docs/research/ideas/pytorch-application-capture.md).
The full synthetic-workload methodology is in
[WHOLE_STACK_RESULTS.md](../../../docs/research/WHOLE_STACK_RESULTS.md).
