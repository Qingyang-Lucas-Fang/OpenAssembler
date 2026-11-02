"""Cooperative CTA barriers in the PTX reference interpreter (openassembler/ptx/interp.py).

Each shared-memory kernel runs three ways and must agree bit-for-bit:
  1. the PTX interpreter (several warps, bar.sync scheduled cooperatively),
  2. the SASS emulator running ptxas's own SASS for the same PTX (the emulator's
     multi-warp BAR.SYNC is the independent model being cross-checked),
  3. a plain-Python reference of what the kernel computes.
Where our backend (openassembler.backend.ra) compiles the kernel, its SASS is
compared too; where it rejects nvcc's loops, that part is skipped with the reason.
Interpreter-only edge cases (deadlock, divergence, exits, counted barriers,
bar.arrive, multi-warp bar.red) are checked against Python references and the
PTX ISA's rules. Needs ptxas/nvdisasm/nvcc (env/cuda.sh) for the oracle parts.
"""

from __future__ import annotations

import random
import struct
import sys
from pathlib import Path

import pytest

from openassembler import oracle
from openassembler.emu.machine import Launch as SassLaunch
from openassembler.emu.machine import Machine, Memory
from openassembler.emu.program import load_listing
from openassembler.ptx.interp import Launch, NotModeled, run_kernel
from openassembler.ptx.parse import parse_module

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

needs_ptxas = pytest.mark.skipif(not (oracle.available("ptxas") and oracle.available("nvdisasm")),
                                 reason=oracle.NEEDS_TOOLKIT)
needs_nvcc = pytest.mark.skipif(not (oracle.available("nvcc") and oracle.available("ptxas")
                                     and oracle.available("nvdisasm")), reason=oracle.NEEDS_TOOLKIT)

OUT, IN, IN2 = 0x7F00_0000_0000, 0x7F00_0010_0000, 0x7F00_0020_0000


def f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def write_f32(mem: Memory, base: int, vals) -> None:
    mem.write_bytes(base, struct.pack(f"<{len(vals)}f", *vals))


def read_f32(mem: Memory, base: int, n: int) -> list[float]:
    return list(struct.unpack(f"<{n}f", bytes(mem.data.get(base + i, 0) for i in range(4 * n))))


def read_u32(mem: Memory, base: int, n: int) -> list[int]:
    return [mem.read(base + 4 * i, 4) for i in range(n)]


def pack_params(kernel, values: dict[str, int]) -> bytes:
    """CUDA param layout: each value at a multiple of its own size."""
    out = b""
    for t, name in kernel.params:
        size = 8 if t.endswith("64") else 4
        out += b"\0" * (-len(out) % size)
        out += struct.pack("<Q" if size == 8 else "<I", values[name] & ((1 << 8 * size) - 1))
    return out


def ctaids(grid):
    return [(x, y, z) for z in range(grid[2]) for y in range(grid[1]) for x in range(grid[0])]


def named(kernel, params) -> dict[str, int]:
    """Params given positionally (nvcc names them <kernel>_param_<i>) -> by name."""
    if isinstance(params, dict):
        return params
    assert len(params) == len(kernel.params)
    return {name: v for (_, name), v in zip(kernel.params, params)}


def run_interp(ptx: str, kname: str, block, grid, params, init) -> Memory:
    mod, mem = parse_module(ptx), Memory()
    params = named(mod.kernels[kname], params)
    init(mem)
    for c in ctaids(grid):
        run_kernel(mod, kname, Launch(block=block, grid=grid, ctaid=c, params=params), mem)
    return mem


def run_sass(sass_text: str, kernel, block, grid, params: dict[str, int], init) -> Memory:
    prog, mem = load_listing(sass_text), Memory()
    init(mem)
    raw = pack_params(kernel, params)
    for c in ctaids(grid):
        Machine(prog, SassLaunch(grid=grid, block=block, ctaid=c, params=raw, param_base=0x160), mem).run()
    return mem


_SASS_CACHE: dict[str, str] = {}


def ptxas_sass(ptx: str, kname: str) -> str:
    """ptxas's own build of kernel `kname` of `ptx` (ptxas -e: only that kernel, so
    load_listing needs no kernel filter -- its filter mistakes the .sectionflags
    line of a kernel with barriers for a new section), disassembled."""
    key = f"{kname}\0{ptx}"
    if key not in _SASS_CACHE:
        scratch = oracle.SCRATCH
        scratch.mkdir(parents=True, exist_ok=True)
        tag = f"test_interp_barrier_{kname}_{abs(hash(ptx)) & 0xFFFFFFFF:08x}"
        src, cub = scratch / f"{tag}.ptx", scratch / f"{tag}.cubin"
        try:
            src.write_text(ptx)
            oracle.ptxas(src, cub, "sm_80", "-e", kname)
            _SASS_CACHE[key] = oracle.disasm_text(cub)
        finally:
            src.unlink(missing_ok=True)
            cub.unlink(missing_ok=True)
    return _SASS_CACHE[key]


def our_sass(ptx: str, kname: str) -> str:
    """Our backend's SASS for kname, or pytest.skip with the backend's reason."""
    from bench_build import listing

    from openassembler.backend import ra

    try:
        code, stats = ra.compile_kernel(parse_module(ptx).kernels[kname])
    except Exception as e:  # NotSupported and friends: the backend's own clear rejection
        pytest.skip(f"our backend rejects {kname}: {type(e).__name__}: {e}")
    return listing(code, stats["labels"])


CUDA_SRC = r"""
extern "C" __global__ void rev(float* out, const float* in) {
    __shared__ float t[1024];
    int i = threadIdx.x, base = blockIdx.x * blockDim.x;
    t[i] = in[base + i];
    __syncthreads();
    out[base + i] = t[blockDim.x - 1 - i];
}
extern "C" __global__ void block_sum(float* out, const float* in) {
    __shared__ float s[256];
    int i = threadIdx.x;
    s[i] = in[blockIdx.x * blockDim.x + i];
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (i < st) s[i] += s[i + st];
        __syncthreads();
    }
    if (i == 0) out[blockIdx.x] = s[0];
}
#define T 16
extern "C" __global__ void matmul_tiled(float* c, const float* a, const float* b, int n) {
    __shared__ float As[T][T], Bs[T][T];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * T + ty, col = blockIdx.x * T + tx;
    float acc = 0.0f;
    for (int t0 = 0; t0 < n; t0 += T) {
        As[ty][tx] = a[row * n + t0 + tx];
        Bs[ty][tx] = b[(t0 + ty) * n + col];
        __syncthreads();
        for (int k = 0; k < T; ++k) acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    c[row * n + col] = acc;
}
extern "C" __global__ void early_exit(float* out, const float* in, int n) {
    __shared__ float s[256];
    int i = threadIdx.x;
    if (i >= n) return;
    s[i] = in[i] * 2.0f;
    __syncthreads();
    out[i] = s[(i + 1) % n];
}
extern "C" __global__ void sync_count(int* out, const int* in) {
    int i = threadIdx.x;
    int c = __syncthreads_count(in[i] > 0);
    int a = __syncthreads_and(in[i] > -1000);
    int o = __syncthreads_or(in[i] > 1000);
    out[i] = c * 100 + a * 10 + o;
}
"""

_PTX: dict[str, str] = {}


def cuda_ptx(kname: str) -> str:
    """nvcc --ptx of CUDA_SRC (one nvcc run for the whole module)."""
    if not _PTX:
        from openassembler.cppfront import compile_to_ptx

        src = oracle.SCRATCH / "test_interp_barrier.cu"
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_text(CUDA_SRC)
        try:
            _PTX["all"], _ = compile_to_ptx(src, entry="rev", arch="sm_80")
        finally:
            src.unlink(missing_ok=True)
    return _PTX["all"]


def three_way(ptx: str, kname: str, block, grid, params, init, read, expect, backend: bool = False):
    """Interpreter == ptxas SASS on the emulator == Python reference (and our SASS if asked)."""
    kernel = parse_module(ptx).kernels[kname]
    params = named(kernel, params)
    got_interp = read(run_interp(ptx, kname, block, grid, params, init))
    assert got_interp == expect, "PTX interpreter disagrees with the Python reference"
    got_ptxas = read(run_sass(ptxas_sass(ptx, kname), kernel, block, grid, params, init))
    assert got_ptxas == got_interp, "ptxas SASS on the emulator disagrees with the PTX interpreter"
    if backend:
        got_ours = read(run_sass(our_sass(ptx, kname), kernel, block, grid, params, init))
        assert got_ours == got_interp, "our SASS disagrees with the PTX interpreter"


HEAD = ".version 8.7\n.target sm_80\n.address_size 64\n"


def hand_rev(barrier: str) -> str:
    """Hand-written shared-memory reversal of %ntid.x (a power of two) u32 values
    around `barrier`: thread i reads slot i ^ (ntid - 1) = ntid - 1 - i. (The XOR
    form keeps ptxas from folding a negative LDS offset, `[R0.X4+-0x4]`, which
    the emulator's operand parser does not accept yet.)"""
    return HEAD + f"""
.visible .entry hrev(.param .u64 out, .param .u64 in)
{{
 .reg .pred %p<4>; .reg .b32 %r<16>; .reg .b64 %rd<8>;
 .shared .align 4 .b8 sbuf[4096];
 ld.param.u64 %rd1, [out]; ld.param.u64 %rd2, [in];
 cvta.to.global.u64 %rd1, %rd1; cvta.to.global.u64 %rd2, %rd2;
 mov.u32 %r1, %tid.x; mov.u32 %r2, %ntid.x;
 mul.wide.u32 %rd3, %r1, 4; add.s64 %rd4, %rd2, %rd3; ld.global.u32 %r3, [%rd4];
 mov.u32 %r4, sbuf; shl.b32 %r5, %r1, 2; add.u32 %r6, %r4, %r5; st.shared.u32 [%r6], %r3;
 {barrier}
 sub.u32 %r7, %r2, 1; xor.b32 %r7, %r7, %r1; shl.b32 %r8, %r7, 2; add.u32 %r9, %r4, %r8;
 ld.shared.u32 %r10, [%r9];
 add.s64 %rd5, %rd1, %rd3; st.global.u32 [%rd5], %r10;
 ret;
}}
"""


def _u32_init(n, seed=0):
    vals = [random.Random(seed * 1000 + i).getrandbits(32) for i in range(n)]
    return vals, (lambda mem: [mem.write(IN + 4 * i, 4, v) for i, v in enumerate(vals)])


@needs_ptxas
@pytest.mark.parametrize("n", [64, 256])
@pytest.mark.parametrize("barrier", ["bar.sync 0;", "barrier.sync 0;", "barrier.sync.aligned 0;",
                                     "bar.sync 5, %r2;"])
def test_hand_ptx_reversal(n, barrier):
    vals, init = _u32_init(n)
    three_way(hand_rev(barrier), "hrev", (n, 1, 1), (1, 1, 1), {"out": OUT, "in": IN}, init,
              lambda m: read_u32(m, OUT, n), vals[::-1])


def test_barrier_actually_orders_shared_memory():
    """Sanity check that the tests above can fail: with the barrier removed, warp 0
    runs to the end before warp 1 stores, so it reads never-written shared memory."""
    vals, init = _u32_init(64)
    got = read_u32(run_interp(hand_rev(""), "hrev", (64, 1, 1), (1, 1, 1),
                              {"out": OUT, "in": IN}, init), OUT, 64)
    assert got != vals[::-1] and got[:32] == [0] * 32


@needs_nvcc
@pytest.mark.parametrize("n,grid", [(64, 1), (256, 1), (128, 3)])
def test_nvcc_reversal(n, grid):
    rng = random.Random(n)
    vals = [f32(rng.uniform(-100, 100)) for _ in range(n * grid)]
    expect = [x for b in range(grid) for x in vals[b * n:(b + 1) * n][::-1]]
    three_way(cuda_ptx("rev"), "rev", (n, 1, 1), (grid, 1, 1), [OUT, IN],
              lambda m: write_f32(m, IN, vals), lambda m: read_f32(m, OUT, n * grid), expect)


@needs_nvcc
def test_nvcc_reversal_our_backend():
    n, grid = 64, 2
    vals = [float(i) for i in range(n * grid)]
    expect = [x for b in range(grid) for x in vals[b * n:(b + 1) * n][::-1]]
    three_way(cuda_ptx("rev"), "rev", (n, 1, 1), (grid, 1, 1), [OUT, IN],
              lambda m: write_f32(m, IN, vals), lambda m: read_f32(m, OUT, n * grid), expect,
              backend=True)


def _tree_sum(xs: list[float]) -> float:
    s = list(xs)
    st = len(s) // 2
    while st:
        for i in range(st):
            s[i] = f32(s[i] + s[i + st])
        st //= 2
    return s[0]


@needs_nvcc
@pytest.mark.parametrize("backend", [False, True], ids=["ptxas", "ours"])
def test_nvcc_block_tree_reduction(backend):
    """256 floats per block, log2(256) = 8 rounds of `if (i < st) ...; __syncthreads()`,
    two blocks. Random floats, so the f32 rounding of the tree order is checked too."""
    rng = random.Random(7)
    vals = [f32(rng.uniform(-10, 10)) for _ in range(512)]
    expect = [_tree_sum(vals[:256]), _tree_sum(vals[256:])]
    three_way(cuda_ptx("block_sum"), "block_sum", (256, 1, 1), (2, 1, 1), [OUT, IN],
              lambda m: write_f32(m, IN, vals), lambda m: read_f32(m, OUT, 2), expect, backend=backend)


@needs_nvcc
@pytest.mark.parametrize("backend", [False, True], ids=["ptxas", "ours"])
def test_nvcc_tiled_matmul(backend):
    """32x32 matmul as a 2x2 grid of 16x16 blocks: two tiles per block, two
    __syncthreads per tile. Small-integer inputs keep every partial sum exact, so
    the reference is independent of nvcc's fma contraction."""
    n = 32
    rng = random.Random(3)
    a = [float(rng.randrange(-4, 5)) for _ in range(n * n)]
    b = [float(rng.randrange(-4, 5)) for _ in range(n * n)]
    expect = [sum(a[r * n + k] * b[k * n + c] for k in range(n)) for r in range(n) for c in range(n)]

    def init(m):
        write_f32(m, IN, a)
        write_f32(m, IN2, b)

    three_way(cuda_ptx("matmul_tiled"), "matmul_tiled", (16, 16, 1), (2, 2, 1),
              [OUT, IN, IN2, n], init, lambda m: read_f32(m, OUT, n * n), expect,
              backend=backend)


@needs_nvcc
@pytest.mark.parametrize("n", [20, 40, 64, 100])
@pytest.mark.parametrize("backend", [False, True], ids=["ptxas", "ours"])
def test_nvcc_threads_exit_before_barrier(n, backend):
    """128 threads, threads >= n return before __syncthreads(): n=20 leaves whole
    warps exited, n=40/100 a partially exited warp. Exited threads count as arrived."""
    rng = random.Random(n)
    vals = [f32(rng.uniform(-50, 50)) for _ in range(128)]
    expect = [f32(vals[(i + 1) % n] * 2) for i in range(n)] + [0.0] * (128 - n)
    three_way(cuda_ptx("early_exit"), "early_exit", (128, 1, 1), (1, 1, 1),
              [OUT, IN, n], lambda m: write_f32(m, IN, vals),
              lambda m: read_f32(m, OUT, 128), expect, backend=backend)


def hand_red(n_extra_warp_code: str = "") -> str:
    """The bar.red forms nvcc emits for __syncthreads_count/_and/_or (copied from its
    PTX, minus the `{ .reg .pred %p1; ... }` scoping blocks the PTX parser doesn't
    take yet), plus a negated predicate: out[i] = count*1000 + and*100 + or*10 + nand."""
    return HEAD + f"""
.visible .entry hred(.param .u64 out, .param .u64 in)
{{
 .reg .pred %p<8>; .reg .b32 %r<16>; .reg .b64 %rd<8>;
 ld.param.u64 %rd1, [out]; ld.param.u64 %rd2, [in];
 cvta.to.global.u64 %rd1, %rd1; cvta.to.global.u64 %rd2, %rd2;
 mov.u32 %r1, %tid.x; mul.wide.u32 %rd3, %r1, 4; add.s64 %rd4, %rd2, %rd3;
 ld.global.u32 %r2, [%rd4];
 {n_extra_warp_code}
 setp.gt.s32 %p1, %r2, 0;     bar.red.popc.u32 %r3, 0, %p1;
 setp.gt.s32 %p2, %r2, -1000; bar.red.and.pred %p3, 0, %p2; selp.u32 %r4, 1, 0, %p3;
 setp.gt.s32 %p4, %r2, 1000;  barrier.red.or.pred %p5, 1, %p4; selp.u32 %r5, 1, 0, %p5;
 bar.red.and.pred %p6, 2, !%p4; selp.u32 %r6, 1, 0, %p6;
 mul.lo.s32 %r7, %r3, 1000; mad.lo.s32 %r7, %r4, 100, %r7; mad.lo.s32 %r7, %r5, 10, %r7;
 add.s32 %r7, %r7, %r6;
 add.s64 %rd5, %rd1, %rd3; st.global.u32 [%rd5], %r7;
 ret;
}}
"""


def _red_expect(xs):
    c = sum(x > 0 for x in xs)
    a = int(all(x > -1000 for x in xs))
    o = int(any(x > 1000 for x in xs))
    na = int(all(not x > 1000 for x in xs))
    return [c * 1000 + a * 100 + o * 10 + na] * len(xs)


def _signed_init(xs):
    return lambda m: [m.write(IN + 4 * i, 4, v & 0xFFFFFFFF) for i, v in enumerate(xs)]


@needs_ptxas
@pytest.mark.parametrize("uniform_and", [False, True])
def test_bar_red_one_warp(uniform_and):
    """bar.red on one warp, where the emulator models BAR.RED too: three-way."""
    xs = [random.Random(i).randrange(-2000, 2000) for i in range(32)]
    if uniform_and:
        xs = [abs(x) % 900 for x in xs]           # all > -1000, none > 1000
    three_way(hand_red(), "hred", (32, 1, 1), (1, 1, 1), [OUT, IN],
              _signed_init(xs), lambda m: read_u32(m, OUT, 32), _red_expect(xs))


@pytest.mark.parametrize("n", [64, 96, 256])
def test_bar_red_multi_warp(n):
    """Several warps: interpreter vs Python only (the emulator rejects BAR.RED with
    more than one warp)."""
    xs = [random.Random(i).randrange(-2000, 2000) for i in range(n)]
    got = read_u32(run_interp(hand_red(), "hred", (n, 1, 1), (1, 1, 1), [OUT, IN], _signed_init(xs)),
                   OUT, n)
    assert got == _red_expect(xs)


# ---- interpreter-only edge cases: PTX ISA rules, and clear refusals ---------------------


def tiny(body: str) -> str:
    """out[tid] = %r9 after `body`; %r1 = tid.x, %r2 = ntid.x, %r9 = 0 initially."""
    return HEAD + f"""
.visible .entry tiny(.param .u64 out)
{{
 .reg .pred %p<8>; .reg .b32 %r<16>; .reg .b64 %rd<8>;
 .shared .align 4 .b8 sbuf[4096];
 ld.param.u64 %rd1, [out]; cvta.to.global.u64 %rd1, %rd1;
 mov.u32 %r1, %tid.x; mov.u32 %r2, %ntid.x; mov.u32 %r9, 0;
 mov.u32 %r8, sbuf; shl.b32 %r7, %r1, 2; add.u32 %r6, %r8, %r7;
 {body}
 mul.wide.u32 %rd3, %r1, 4; add.s64 %rd4, %rd1, %rd3; st.global.u32 [%rd4], %r9;
 ret;
}}
"""


def run_tiny(body: str, n: int = 64, **kw) -> list[int]:
    mod, mem = parse_module(tiny(body)), Memory()
    from openassembler.ptx.interp import Interp

    it = Interp(mod, "tiny", Launch(block=(n, 1, 1), params={"out": OUT}), mem)
    it.run(**kw)
    return read_u32(mem, OUT, n)


# thread i stores i+1 to sbuf[i]; after the barrier reads its neighbour's slot (i ^ 32).
NEIGHBOUR = "add.u32 %r3, %r1, 1; st.shared.u32 [%r6], %r3; {bar} xor.b32 %r4, %r7, 128; " \
            "add.u32 %r5, %r8, %r4; ld.shared.u32 %r9, [%r5];"


def test_guard_false_threads_that_exit_count_as_arrived():
    """`@%p bar.sync 0` where warp 1's lanes 8..31 skip the barrier and exit: the
    barrier still completes (exited threads count as arrived) for the others."""
    body = "setp.lt.u32 %p1, %r1, 40; @!%p1 bra DONE; " + NEIGHBOUR.format(bar="bar.sync 0;") + " DONE:"
    got = run_tiny(body, 64)
    assert got[:8] == [i + 33 for i in range(8)]      # warp 0 lanes 0..7 read warp 1's stores
    assert got[32:40] == [i + 1 for i in range(8)]    # warp 1 lanes 0..7 read warp 0's


def test_same_barrier_from_divergent_paths():
    """Odd/even lanes reach barrier 0 at two different instructions. barrier.sync
    (non-aligned) is fine per the PTX ISA; bar.sync (= barrier.sync.aligned) is
    undefined behaviour there and is refused."""
    split = ("and.b32 %r10, %r1, 1; setp.eq.u32 %p1, %r10, 0; add.u32 %r3, %r1, 1; st.shared.u32 [%r6], %r3; "
             "@%p1 bra EVEN; {bar} bra JOIN; EVEN: {bar} JOIN: xor.b32 %r4, %r7, 128; add.u32 %r5, %r8, %r4; "
             "ld.shared.u32 %r9, [%r5];")
    got = run_tiny(split.format(bar="barrier.sync 0;"), 64)
    assert got == [(i ^ 32) + 1 for i in range(64)]
    with pytest.raises(NotModeled, match="aligned barrier from different instructions"):
        run_tiny(split.format(bar="bar.sync 0;"), 64)


def test_deadlock_is_reported_not_hung():
    """Warp 0 waits at barrier 0, warp 1 at barrier 1: neither can complete."""
    body = "setp.lt.u32 %p1, %r1, 32; @%p1 bra A; bar.sync 1; bra B; A: bar.sync 0; B:"
    with pytest.raises(NotModeled, match="CTA barrier deadlock: barrier 0: 32 threads wait but 32 "):
        run_tiny(body, 64)


def test_step_limit_with_barrier_in_loop():
    body = "LOOP: bar.sync 0; add.u32 %r9, %r9, 1; bra LOOP;"
    with pytest.raises(NotModeled, match="step limit exceeded"):
        run_tiny(body, 64, max_steps=5000)


def test_barrier_in_loop_many_rounds():
    """A bar.sync inside a uniform 10-iteration loop: 10 completions of barrier 0,
    each round reading a slot another warp wrote in the same round."""
    body = ("mov.u32 %r11, 0; LOOP: add.u32 %r3, %r9, %r1; st.shared.u32 [%r6], %r3; bar.sync 0; "
            "xor.b32 %r4, %r7, 128; add.u32 %r5, %r8, %r4; ld.shared.u32 %r12, [%r5]; bar.sync 0; "
            "add.u32 %r9, %r9, %r12; add.u32 %r11, %r11, 1; setp.lt.u32 %p2, %r11, 10; @%p2 bra LOOP;")
    got = run_tiny(body, 64)
    for i in range(64):
        a = b = 0                                   # thread i and its partner i ^ 32
        for _ in range(10):
            a, b = a + (b + (i ^ 32)), b + (a + i)
        assert got[i] == a


@pytest.mark.parametrize("producer_warp", [0, 1])
def test_counted_arrive_and_sync(producer_warp):
    """Producer/consumer on named barrier 3 with count 64 (CUTLASS-style): the producer
    warp stores and bar.arrive's (doesn't wait), the consumer warp bar.sync's and then
    reads. Correct whichever warp the interpreter happens to run first."""
    body = (f"shr.u32 %r10, %r1, 5; setp.eq.u32 %p3, %r10, 2; @%p3 bra END; "
            f"setp.eq.u32 %p1, %r10, {producer_warp}; @!%p1 bra CONS; "
            "add.u32 %r3, %r1, 100; st.shared.u32 [%r6], %r3; bar.arrive 3, 64; bra END; "
            "CONS: bar.sync 3, 64; xor.b32 %r4, %r7, 128; add.u32 %r5, %r8, %r4; ld.shared.u32 %r9, [%r5]; END:")
    got = run_tiny(body, 96)                        # warp 2 neither produces nor consumes
    cons = 1 - producer_warp
    assert got[64:] == [0] * 32
    assert got[32 * cons:32 * cons + 32] == [((32 * cons + i) ^ 32) + 100 for i in range(32)]


def test_counted_barrier_second_phase_deadlocks():
    """Count 64 with three warps all doing bar.sync: warps 0+1 complete phase one, warp 2
    alone opens phase two, which can never complete -> reported, not hung."""
    with pytest.raises(NotModeled, match="barrier 3 expects 64 threads but only 32 arrived"):
        run_tiny("bar.sync 3, 64;", 96)


@pytest.mark.parametrize("body,n,msg", [
    ("setp.lt.u32 %p1, %r1, 40; @%p1 bar.sync 1, 64;", 64, "counted barrier reached by a partial warp"),
    ("bar.sync 1, 48;", 64, "not a positive multiple of 32"),
    ("bar.sync 1, 128;", 64, "not a positive multiple of 32"),
    ("bar.red.popc.u32 %r9, 0, 64, %p1;", 32, "not a positive multiple of 32"),
    ("bar.sync 16;", 64, "out of range"),
    ("bar.sync %r1;", 64, "barrier id differs between threads"),
    ("bar.arrive 1;", 64, "expected barrier id and thread count"),
    ("bar 0;", 64, "expected exactly one of"),
    ("barrier.cluster.arrive;", 64, r"modifier\(s\) \['cluster'\] not modeled"),
    ("bar.red.add.u32 %r9, 0, %p1;", 64, r"modifier\(s\) \['add'\] not modeled"),
    ("bar.red.popc.pred %p2, 0, %p1;", 64, "expected bar.red.popc.u32"),
    ("bar.arrive 2, 64; bar.arrive 2, 64;", 64, "arrived twice"),
    ("setp.lt.u32 %p2, %r1, 32; @%p2 bar.sync 2; @!%p2 bar.red.or.pred %p3, 2, %p1;", 64,
     "mixing barrier forms"),
    ("setp.lt.u32 %p2, %r1, 16; @%p2 bra X; bar.warp.sync -1; X:", 64, "divergent __syncwarp"),
])
def test_rejected_forms(body, n, msg):
    with pytest.raises(NotModeled, match=msg):
        run_tiny("setp.ne.u32 %p1, %r1, 0; " + body, n)


def test_bar_sync_then_bar_red_same_id_sequentially():
    """A barrier id is free again once it completes: bar.sync 0 then bar.red on 0."""
    got = run_tiny("setp.ne.u32 %p1, %r1, 0; bar.sync 0; bar.red.popc.u32 %r9, 0, %p1;", 96)
    assert got == [95] * 96


def test_warp_sync_converged_is_fine():
    assert run_tiny("bar.warp.sync -1; mov.u32 %r9, %r1;", 64) == list(range(64))


# ---- openassembler.verify.run_and_diff: grid / block ------------------------------------


def test_launch_geometry_default_unchanged():
    from openassembler.verify import launch_geometry

    assert launch_geometry(64) == ((1, 1, 1), (64, 1, 1))
    assert launch_geometry(64, grid=2) == ((2, 1, 1), (32, 1, 1))
    assert launch_geometry(100, grid=3) == ((3, 1, 1), (34, 1, 1))
    assert launch_geometry(64, grid=(2, 2, 1), block=(16, 16, 1)) == ((2, 2, 1), (16, 16, 1))
    with pytest.raises(ValueError, match="1024"):
        launch_geometry(4096)
    with pytest.raises(ValueError):
        launch_geometry(64, grid=0)


@needs_nvcc
@pytest.mark.parametrize("n_elems,grid", [(64, 1), (128, 2), (256, 4), (512, 2)])
def test_run_and_diff_multi_cta_with_barrier(n_elems, grid):
    """run_and_diff on a multi-warp, multi-CTA __syncthreads kernel compiled by our
    backend: bit-exact. Also proves the per-CTA %ctaid is wired on both engines:
    the reversal is per block, so a wrong ctaid would mismatch."""
    from openassembler.backend import ra
    from openassembler.verify import run_and_diff

    mod = parse_module(cuda_ptx("rev"))
    try:
        code, stats = ra.compile_kernel(mod.kernels["rev"])
    except Exception as e:
        pytest.skip(f"our backend rejects rev: {e}")
    assert run_and_diff(mod, "rev", code, stats["labels"], n_elems=n_elems, grid=grid, seed=1) == []
    assert any(t.startswith("BAR.SYNC") for t, _ in code)
    if n_elems // grid > 32:
        # BAR.SYNC -> NOP: the emulator's warp 0 now reads shared memory before warp 1
        # stored it, and run_and_diff must notice (the check isn't vacuous)
        broken = [("NOP", c) if t.startswith("BAR.SYNC") else (t, c) for t, c in code]
        assert run_and_diff(mod, "rev", broken, stats["labels"], n_elems=n_elems, grid=grid, seed=1)
