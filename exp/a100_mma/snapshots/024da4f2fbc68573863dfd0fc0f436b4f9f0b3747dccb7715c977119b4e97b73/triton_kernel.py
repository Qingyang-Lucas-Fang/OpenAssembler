"""Comparison oracle for tiling/pipeline targets; not OpenAssembler-generated code."""
import base64
import triton
import triton.language as tl


@triton.jit
def gemm(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
         BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr):
    pid = tl.program_id(0)
    nm = tl.cdiv(M, BM)
    nn = tl.cdiv(N, BN)
    if GROUP == 0:
        pm = pid // nn
        pn = pid % nn
    else:
        group_id = pid // (GROUP * nn)
        first_m = group_id * GROUP
        group_size = tl.minimum(nm - first_m, GROUP)
        pm = first_m + (pid % (GROUP * nn)) % group_size
        pn = (pid % (GROUP * nn)) // group_size
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    # All callers enforce exact tile divisibility. Both inputs are K-contiguous.
    for step in range(tl.cdiv(K, BK)):
        kk = step * BK + rk
        aa = tl.load(A + rm[:, None] * K + kk[None, :])
        bb = tl.load(B + rn[None, :] * K + kk[:, None])
        acc = tl.dot(aa, bb, acc)
    tl.store(C + rm[:, None] * N + rn[None, :], acc)


CONFIGS = [
    (32, 64, 32, 4, 3, 0),
    (64, 128, 32, 4, 4, 8),
    (128, 128, 32, 4, 4, 8),
    (128, 128, 64, 4, 3, 8),
    (128, 256, 32, 8, 3, 8),
    (128, 256, 64, 8, 3, 8),
    (64, 256, 32, 4, 4, 8),
    (256, 128, 32, 8, 3, 8),
    (128, 128, 32, 8, 4, 8),
]


def comparison_kernels(a, b, out, m, n, k):
    launches, artifacts = {}, []
    for bm, bn, bk, warps, stages, group in CONFIGS:
        if m % bm or n % bn or k % bk:
            continue
        name = f"triton_{bm}x{bn}x{bk}_w{warps}_s{stages}_g{group}"
        def launch(bm=bm, bn=bn, bk=bk, warps=warps, stages=stages, group=group):
            return gemm[(m // bm * (n // bn),)](
                a, b, out, m, n, k, bm, bn, bk, group, num_warps=warps, num_stages=stages)
        compiled = launch()
        launches[name] = launch
        artifacts.append({"name": name, "compiler": "Triton (comparison only)",
                          "triton_version": triton.__version__, "configuration": [bm, bn, bk, warps, stages, group],
                          "n_regs": getattr(compiled, "n_regs", None),
                          "n_spills": getattr(compiled, "n_spills", None),
                          "shared": getattr(compiled.metadata, "shared", None),
                          "ptx": compiled.asm["ptx"],
                          "cubin": base64.b64encode(compiled.asm["cubin"]).decode()})
    return launches, artifacts
