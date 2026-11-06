# Native hybrid MLP benchmark attempts

The first A100 attempt stopped before kernel execution with
`CUDA_ERROR_INVALID_CONTEXT` while loading cubins. `torch.cuda.init()` alone
had not materialized the primary CUDA context; a small PyTorch CUDA allocation
was added before loading driver modules. No kernel timing came from this attempt.

The second attempt passed all checks for `M=512,D=512,H=2048` and completed its
timing, then stopped on the larger case's preset numerical gate. After three
AdamW steps at `M=1024,D=1024,H=4096`, the first-layer weight gradient differed
between cuBLAS and OA by `0.006364095` relative L2 and `9.83872e-6` maximum
absolute error. This exceeded the preset gradient relative-L2 threshold of
`0.005`. OA and same-generated-PTX ptxas states were bitwise identical. The
threshold was not loosened. The runner was changed to retain the full numerical
record and omit cuBLAS timings for a case that fails that gate, rather than
aborting and losing the passing earlier case. The change concerns result
handling, not arithmetic or performance policy.

Whole-step comparisons are hybrid: GEMMs use native OpenAssembler, ptxas or
cuBLAS; LayerNorm, ReLU, residual, MSE loss, AdamW, packing and FP16 casts share
the PyTorch CUDA runtime. Timing includes those shared costs and host dispatch.
This is not evidence for an entirely native OpenAssembler transformer.
