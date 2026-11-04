"""A 2-layer MLP forward pass (linear -> ReLU -> linear) -- the feed-forward
block inside a transformer layer, minus attention/softmax/layernorm (those
need exp/transcendentals, which openassembler.pyfront and our backend don't
support yet; see docs/STATUS.md "Gaps"). Every kernel here compiles through
openassembler.pyfront's small Python subset straight to our own SASS backend --
no ptxas code generation anywhere in this pipeline.

Shapes: x (batch x in_dim), W1 (in_dim x hidden), b1 (hidden),
        W2 (hidden x out_dim), b2 (out_dim), out (batch x out_dim).
One thread computes one output element (row, col) of its layer -- naive,
not tiled/blocked like openassembler.fastmm's GEMM kernels, which is the honest
trade for writing it in ~10 lines of Python instead of hand-tuned PTX.

linear1/linear2 launch a 2D grid: ctaid.x = row (one block per batch row,
as before), ctaid.y * ntid.x + tid.x = col -- so hidden/out_dim can exceed
CUDA's 1024-thread-per-block limit (tools/mlp_run.py picks ntid.x and
ctaid.y's count to cover hidden/out_dim between them).

    openassembler compile examples/mlp/kernels.py --kernel linear1 -o linear1.cubin
"""

from openassembler.pyfront import Array, f32, i32, kernel


@kernel
def linear1(x: Array(f32), w: Array(f32), b: Array(f32), out: Array(f32),
            batch: i32, in_dim: i32, hidden: i32):
    row = ctaid_x()
    col = tid_x() + ctaid_y() * ntid_x()
    if row < batch and col < hidden:
        acc = b[col]
        for k in range(0, in_dim):
            acc = acc + x[row * in_dim + k] * w[k * hidden + col]
        out[row * hidden + col] = acc


@kernel
def relu(x: Array(f32), out: Array(f32), n: i32):
    i = tid_x() + ctaid_x() * ntid_x()
    if i < n:
        v = x[i]
        if v < 0.0:
            v = 0.0
        out[i] = v


@kernel
def linear2(h: Array(f32), w: Array(f32), b: Array(f32), out: Array(f32),
            batch: i32, hidden: i32, out_dim: i32):
    row = ctaid_x()
    col = tid_x() + ctaid_y() * ntid_x()
    if row < batch and col < out_dim:
        acc = b[col]
        for k in range(0, hidden):
            acc = acc + h[row * hidden + k] * w[k * out_dim + col]
        out[row * out_dim + col] = acc
