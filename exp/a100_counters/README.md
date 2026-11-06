# A100 counter-study workspace

The canonical methods, command sets and conclusions are in
[Hardware-counter-guided tensor analysis](../../docs/research/ideas/tensor-counter-profiling.md).
The [Morton dossier](../../docs/research/ideas/morton-cta-order.md) and
[direct-pointer dossier](../../docs/research/ideas/direct-pointer-homes.md) state
which later measurements support actual implementation choices.

`run.py` is the current bounded driver; `counter-validation_*.json` are raw
records. Preflight can request a GPU. NCU availability/permissions are checked;
a failed profile is not zero stall activity. Counter replay perturbs caches,
so its duration is not the separate validated CUDA-graph benchmark latency.
No experimental candidate is automatically installed in production by this driver.
