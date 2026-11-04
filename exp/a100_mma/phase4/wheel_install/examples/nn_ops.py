"""Small correctness/measurement baselines, not tuned production NN kernels.

Use tools/nn_study.py to compile, validate and measure each operation. Each output
is overwritten on every invocation so repeated timing does not mutate its inputs.
"""
from openassembler.pyfront import (Array, f32, i32, kernel, tid_x, ctaid_x, ntid_x,
                                   exp, log, rsqrt, sqrt, tanh)


@kernel
def relu(x: Array(f32), out: Array(f32), n: i32):
    i = ctaid_x() * ntid_x() + tid_x()
    if i < n:
        out[i] = max(x[i], 0.0)


@kernel
def gelu(x: Array(f32), out: Array(f32), n: i32):
    i = ctaid_x() * ntid_x() + tid_x()
    if i < n:
        v = x[i]
        out[i] = 0.5 * v * (1.0 + tanh(0.7978845608 * (v + 0.044715 * v * v * v)))


@kernel
def softmax(x: Array(f32), out: Array(f32), rows: i32, width: i32):
    row = ctaid_x() * ntid_x() + tid_x()
    if row < rows:
        base = row * width
        mx = -3.402823466e38
        for j in range(width):
            mx = max(mx, x[base + j])
        denom = 0.0
        for j in range(width):
            denom += exp(x[base + j] - mx)
        for j in range(width):
            out[base + j] = exp(x[base + j] - mx) / denom


@kernel
def layernorm(x: Array(f32), out: Array(f32), rows: i32, width: i32):
    row = ctaid_x() * ntid_x() + tid_x()
    if row < rows:
        base = row * width
        mean = 0.0
        for j in range(width):
            mean += x[base + j]
        mean = mean / float(width)
        var = 0.0
        for j in range(width):
            d = x[base + j] - mean
            var += d * d
        inv = rsqrt(var / float(width) + 0.00001)
        for j in range(width):
            out[base + j] = (x[base + j] - mean) * inv


@kernel
def adam_step(w: Array(f32), g: Array(f32), m: Array(f32), v: Array(f32),
              out: Array(f32), mo: Array(f32), vo: Array(f32), n: i32):
    i = ctaid_x() * ntid_x() + tid_x()
    if i < n:
        mn = 0.9 * m[i] + 0.1 * g[i]
        vn = 0.999 * v[i] + 0.001 * g[i] * g[i]
        mo[i] = mn
        vo[i] = vn
        out[i] = w[i] - 0.001 * mn / (sqrt(vn) + 0.00000001)


@kernel
def cross_entropy(x: Array(f32), labels: Array(i32), loss: Array(f32), rows: i32, width: i32):
    row = ctaid_x() * ntid_x() + tid_x()
    if row < rows:
        base = row * width
        mx = -3.402823466e38
        for j in range(width):
            mx = max(mx, x[base + j])
        total = 0.0
        for j in range(width):
            total += exp(x[base + j] - mx)
        loss[row] = log(total) + mx - x[base + labels[row]]


@kernel
def matmul(a: Array(f32), b: Array(f32), out: Array(f32), m: i32, n: i32, k: i32):
    idx = ctaid_x() * ntid_x() + tid_x()
    if idx < m * n:
        row = idx // n
        col = idx % n
        acc = 0.0
        for j in range(k):
            acc += a[row * k + j] * b[j * n + col]
        out[idx] = acc


@kernel
def attention_scores(q: Array(f32), k: Array(f32), out: Array(f32), tokens: i32, dim: i32):
    idx = ctaid_x() * ntid_x() + tid_x()
    if idx < tokens * tokens:
        row = idx // tokens
        col = idx % tokens
        acc = 0.0
        for j in range(dim):
            acc += q[row * dim + j] * k[col * dim + j]
        out[idx] = acc * rsqrt(float(dim))
