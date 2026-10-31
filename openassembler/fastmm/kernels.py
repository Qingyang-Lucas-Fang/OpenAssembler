"""PTX generators for the kernels a block fast-matmul plan launches.

sgemm   C = A B  (fp32, row-major, leading dimensions in elements). One CTA computes a
        64 x 64 tile of C with 256 threads; each thread a 4 x 4 sub-tile. K is consumed
        8 at a time through shared memory (A stored transposed: As[k][m]); the next
        k-tile is loaded from global memory into registers while the current one is
        multiplied. Requires M % 64 == N % 64 == 0, K % 8 == 0, lda/ldb/ldc % 4 == 0.
        Grid: (N/64, M/64).

lincomb out = sum_j c_j * in_j over a rows x cols block (each operand a strided view).
        One thread handles 4 consecutive floats. Grid: (ceil(cols/1024), rows), 256
        threads; requires cols % 4 == 0 and 16-byte aligned views.

Both are plain PTX so that the same source can be compiled by ptxas and by our
compiler (openassembler.backend.ra), which is how they are compared.
"""

from __future__ import annotations

HEADER = ".version 8.7\n.target {arch}\n.address_size 64\n\n"

BM = BN = 64
BK = 8
THREADS = 256


def sgemm(arch: str = "sm_80", name: str = "sgemm64") -> str:
    L = []
    e = L.append
    e(HEADER.format(arch=arch))
    e(f".visible .entry {name}(.param .u64 pA, .param .u64 pB, .param .u64 pC, .param .u32 lda,"
      f" .param .u32 ldb, .param .u32 ldc, .param .u32 K)\n{{")
    e(" .reg .pred %p<4>; .reg .b32 %r<64>; .reg .f32 %f<64>; .reg .f32 %a<32>; .reg .f32 %b<32>;")
    e(" .reg .f32 %g<4>; .reg .b64 %rd<32>;")
    e(" .shared .align 16 .b8 As[2048]; .shared .align 16 .b8 Bs[2048];")
    e(" ld.param.u64 %rd1, [pA]; ld.param.u64 %rd2, [pB]; ld.param.u64 %rd3, [pC];")
    e(" cvta.to.global.u64 %rd1, %rd1; cvta.to.global.u64 %rd2, %rd2; cvta.to.global.u64 %rd3, %rd3;")
    e(" ld.param.u32 %r40, [lda]; ld.param.u32 %r41, [ldb]; ld.param.u32 %r42, [ldc]; ld.param.u32 %r43, [K];")
    e(" mov.u32 %r1, %tid.x; mov.u32 %r2, %ctaid.x; mov.u32 %r3, %ctaid.y;")
    e(" and.b32 %r4, %r1, 15; shr.u32 %r5, %r1, 4;")          # tx, ty
    e(" shr.u32 %r6, %r1, 3; and.b32 %r7, %r1, 7;")           # A loader: row, k
    e(" shr.u32 %r8, %r1, 6; and.b32 %r9, %r1, 63;")          # B loader: k, col
    # A element indices: (by*64 + r6) * lda + r7, and +32 rows
    e(" shl.b32 %r10, %r3, 6; add.u32 %r10, %r10, %r6; mad.lo.u32 %r11, %r10, %r40, %r7;")
    e(" shl.b32 %r12, %r40, 5; add.u32 %r13, %r11, %r12;")
    # B element indices: r8 * ldb + bx*64 + r9, and +4 rows
    e(" shl.b32 %r14, %r2, 6; add.u32 %r14, %r14, %r9; mad.lo.u32 %r15, %r8, %r41, %r14;")
    e(" shl.b32 %r16, %r41, 2; add.u32 %r17, %r15, %r16;")
    # shared-memory addresses: As[r7][r6] (+32 rows -> +128 B), Bs[r8][r9] (+4 k -> +1024 B)
    e(" mov.u32 %r18, As; mov.u32 %r19, Bs;")
    e(" shl.b32 %r20, %r7, 6; add.u32 %r20, %r20, %r6; shl.b32 %r20, %r20, 2; add.u32 %r20, %r20, %r18;")
    e(" shl.b32 %r21, %r8, 6; add.u32 %r21, %r21, %r9; shl.b32 %r21, %r21, 2; add.u32 %r21, %r21, %r19;")
    # compute-side smem bases: As + ty*16, Bs + tx*16
    e(" shl.b32 %r22, %r5, 4; add.u32 %r22, %r22, %r18; shl.b32 %r23, %r4, 4; add.u32 %r23, %r23, %r19;")
    for i in range(16):
        e(f" mov.f32 %f{i}, 0f00000000;")
    # prologue: k-tile 0 -> registers
    _gload(e, "%r11", "%r13", "%r15", "%r17", k="0", kb="0")
    e(" mov.u32 %r30, 0;")                                    # k0
    e(" LOOP:")
    e(" st.shared.f32 [%r20], %g0; st.shared.f32 [%r20+128], %g1;")
    e(" st.shared.f32 [%r21], %g2; st.shared.f32 [%r21+1024], %g3;")
    e(" bar.sync 0;")
    e(" add.u32 %r31, %r30, 8; setp.lt.u32 %p1, %r31, %r43; selp.u32 %r32, %r31, %r30, %p1;")
    e(" mul.lo.u32 %r33, %r32, %r41;")
    _gload(e, "%r11", "%r13", "%r15", "%r17", k="%r32", kb="%r33")
    for k in range(BK):
        # distinct names per k: each is assigned once, so no loop-carried home register
        a, b = [f"%a{4 * k + i}" for i in range(4)], [f"%b{4 * k + i}" for i in range(4)]
        e(f" ld.shared.v4.f32 {{{', '.join(a)}}}, [%r22+{k * 256}];")
        e(f" ld.shared.v4.f32 {{{', '.join(b)}}}, [%r23+{k * 256}];")
        for i in range(4):
            for j in range(4):
                e(f" fma.rn.f32 %f{4 * i + j}, {a[i]}, {b[j]}, %f{4 * i + j};")
    e(" bar.sync 0;")
    e(" mov.u32 %r30, %r31;")
    e(" @%p1 bra LOOP;")
    # epilogue: C[(by*64 + ty*4 + i) * ldc + bx*64 + tx*4 .. +3]
    e(" shl.b32 %r34, %r3, 6; shl.b32 %r35, %r5, 2; add.u32 %r34, %r34, %r35;")
    e(" shl.b32 %r36, %r2, 6; shl.b32 %r37, %r4, 2; add.u32 %r36, %r36, %r37;")
    e(" mad.lo.u32 %r38, %r34, %r42, %r36;")
    for i in range(4):
        if i:
            e(f" add.u32 %r38, %r38, %r42;")
        e(f" mul.wide.u32 %rd{10 + i}, %r38, 4; add.s64 %rd{14 + i}, %rd3, %rd{10 + i};")
        e(f" st.global.v4.f32 [%rd{14 + i}], {{%f{4 * i}, %f{4 * i + 1}, %f{4 * i + 2}, %f{4 * i + 3}}};")
    e(" ret;\n}")
    return "\n".join(L) + "\n"


def _gload(e, ia0, ia1, ib0, ib1, k, kb):
    """Load the k-tile starting at column k (A) / row k (B, kb = k*ldb) into %g0..%g3."""
    for n, (idx, off, base, g) in enumerate(((ia0, k, "%rd1", "%g0"), (ia1, k, "%rd1", "%g1"),
                                             (ib0, kb, "%rd2", "%g2"), (ib1, kb, "%rd2", "%g3"))):
        e(f" add.u32 %r{50 + n}, {idx}, {off}; mul.wide.u32 %rd{20 + n}, %r{50 + n}, 4;"
          f" add.s64 %rd{24 + n}, {base}, %rd{20 + n}; ld.global.f32 {g}, [%rd{24 + n}];")


def lincomb(n_in: int, arch: str = "sm_80") -> str:
    name = f"lincomb{n_in}"
    params = [".param .u64 pO", ".param .u32 ldo"]
    for j in range(n_in):
        params += [f".param .u64 pI{j}", f".param .u32 ld{j}", f".param .f32 c{j}"]
    params.append(".param .u32 cols")
    L = [HEADER.format(arch=arch), f".visible .entry {name}({', '.join(params)})\n{{",
         " .reg .pred %p<4>; .reg .b32 %r<64>; .reg .f32 %f<128>; .reg .f32 %x<4>; .reg .b64 %rd<64>;"]
    e = L.append
    e(" mov.u32 %r1, %tid.x; mov.u32 %r2, %ctaid.x; mov.u32 %r3, %ctaid.y;")
    e(" shl.b32 %r4, %r2, 10; shl.b32 %r5, %r1, 2; add.u32 %r4, %r4, %r5;")     # col
    e(" ld.param.u32 %r6, [cols]; setp.ge.u32 %p1, %r4, %r6; @%p1 bra DONE;")
    for j in range(n_in):
        e(f" ld.param.u64 %rd{2 * j + 10}, [pI{j}]; cvta.to.global.u64 %rd{2 * j + 10}, %rd{2 * j + 10};")
        e(f" ld.param.u32 %r{10 + j}, [ld{j}]; mad.lo.u32 %r{30 + j}, %r3, %r{10 + j}, %r4;")
        e(f" mul.wide.u32 %rd{2 * j + 11}, %r{30 + j}, 4; add.s64 %rd{2 * j + 11}, %rd{2 * j + 10}, %rd{2 * j + 11};")
        e(f" ld.global.v4.f32 {{%f{4 * j}, %f{4 * j + 1}, %f{4 * j + 2}, %f{4 * j + 3}}}, [%rd{2 * j + 11}];")
        e(f" ld.param.f32 %f{100 + j}, [c{j}];")
    for q in range(4):
        e(f" mul.rn.f32 %x{q}, %f{q}, %f100;")
        for j in range(1, n_in):
            e(f" fma.rn.f32 %x{q}, %f{4 * j + q}, %f{100 + j}, %x{q};")
    e(" ld.param.u64 %rd1, [pO]; cvta.to.global.u64 %rd1, %rd1; ld.param.u32 %r7, [ldo];")
    e(" mad.lo.u32 %r8, %r3, %r7, %r4; mul.wide.u32 %rd2, %r8, 4; add.s64 %rd3, %rd1, %rd2;")
    e(" st.global.v4.f32 [%rd3], {%x0, %x1, %x2, %x3};")
    e(" DONE: ret;\n}")
    return "\n".join(L) + "\n"


def sgemm128(arch: str = "sm_80", name: str = "sgemm128") -> str:
    """128 x 128 C tile per CTA, 256 threads, 8 x 8 outputs per thread (rows ty*4+{0..3},
    64+ty*4+{0..3}; same split for columns, so LDS.128 reads are conflict-free), BK = 8,
    double-buffered shared memory (one barrier per k-tile), next tile prefetched into
    registers with 128-bit global loads. Requires M, N % 128 == 0, K % 8 == 0,
    lda/ldb/ldc % 4 == 0 and 16-byte aligned views. Grid (N/128, M/128)."""
    L = []
    e = L.append
    e(HEADER.format(arch=arch))
    e(f".visible .entry {name}(.param .u64 pA, .param .u64 pB, .param .u64 pC, .param .u32 lda,"
      f" .param .u32 ldb, .param .u32 ldc, .param .u32 K)\n{{")
    e(" .reg .pred %p<4>; .reg .b32 %r<80>; .reg .f32 %f<64>; .reg .f32 %a<128>; .reg .f32 %b<128>;")
    e(" .reg .f32 %g<8>; .reg .b64 %rd<40>;")
    # buffer q at S + q*8192: As[k][m] (8 x 128 floats = 4096 B) then Bs[k][n] (4096 B)
    e(" .shared .align 16 .b8 S[16384];")
    e(" ld.param.u64 %rd1, [pA]; ld.param.u64 %rd2, [pB]; ld.param.u64 %rd3, [pC];")
    e(" cvta.to.global.u64 %rd1, %rd1; cvta.to.global.u64 %rd2, %rd2; cvta.to.global.u64 %rd3, %rd3;")
    e(" ld.param.u32 %r40, [lda]; ld.param.u32 %r41, [ldb]; ld.param.u32 %r42, [ldc]; ld.param.u32 %r43, [K];")
    e(" mov.u32 %r1, %tid.x; mov.u32 %r2, %ctaid.x; mov.u32 %r3, %ctaid.y;")
    e(" and.b32 %r4, %r1, 15; shr.u32 %r5, %r1, 4;")            # tx, ty
    e(" shr.u32 %r6, %r1, 1; and.b32 %r7, %r1, 1; shl.b32 %r7, %r7, 2;")   # A loader: row, k (0/4)
    e(" shr.u32 %r8, %r1, 5; and.b32 %r9, %r1, 31; shl.b32 %r9, %r9, 2;")  # B loader: k, col
    e(" shl.b32 %r10, %r3, 7; add.u32 %r10, %r10, %r6; mad.lo.u32 %r11, %r10, %r40, %r7;")  # A idx
    e(" shl.b32 %r14, %r2, 7; add.u32 %r14, %r14, %r9; mad.lo.u32 %r15, %r8, %r41, %r14;")  # B idx
    e(" mov.u32 %r18, S;")
    # store addresses (buffer 0): As[r7+q][r6] = S + (r7+q)*512 + r6*4 ; Bs[r8][r9] = S+4096+r8*512+r9*4
    e(" shl.b32 %r20, %r7, 9; shl.b32 %r21, %r6, 2; add.u32 %r20, %r20, %r21; add.u32 %r20, %r20, %r18;")
    e(" shl.b32 %r22, %r8, 9; shl.b32 %r23, %r9, 2; add.u32 %r22, %r22, %r23; add.u32 %r22, %r22, %r18;")
    # compute addresses (buffer 0): As + ty*16, Bs + tx*16
    e(" shl.b32 %r24, %r5, 4; add.u32 %r24, %r24, %r18; shl.b32 %r25, %r4, 4; add.u32 %r25, %r25, %r18;")
    for i in range(64):
        e(f" mov.f32 %f{i}, 0f00000000;")
    # prologue: tile 0 -> buffer 0
    _gload128(e, "0", "0")
    _sstore128(e, "%r20", "%r22")
    e(" bar.sync 0;")
    e(" mov.u32 %r30, 0; mov.u32 %r60, 0;")                   # k0, buffer byte offset (0 / 8192)
    e(" LOOP:")
    e(" add.u32 %r31, %r30, 8; setp.lt.u32 %p1, %r31, %r43; selp.u32 %r32, %r31, %r30, %p1;")
    e(" mul.lo.u32 %r33, %r32, %r41;")
    _gload128(e, "%r32", "%r33")
    e(" add.u32 %r61, %r24, %r60; add.u32 %r62, %r25, %r60;")
    for k in range(8):
        a = [f"%a{16 * k + i}" for i in range(8)]
        b = [f"%b{16 * k + i}" for i in range(8)]
        e(f" ld.shared.v4.f32 {{{', '.join(a[:4])}}}, [%r61+{k * 512}];")
        e(f" ld.shared.v4.f32 {{{', '.join(a[4:])}}}, [%r61+{k * 512 + 256}];")
        e(f" ld.shared.v4.f32 {{{', '.join(b[:4])}}}, [%r62+{4096 + k * 512}];")
        e(f" ld.shared.v4.f32 {{{', '.join(b[4:])}}}, [%r62+{4096 + k * 512 + 256}];")
        for i in range(8):
            for j in range(8):
                e(f" fma.rn.f32 %f{8 * i + j}, {a[i]}, {b[j]}, %f{8 * i + j};")
    # next tile -> the other buffer
    e(" xor.b32 %r60, %r60, 8192;")
    e(" add.u32 %r63, %r20, %r60; add.u32 %r64, %r22, %r60;")
    _sstore128(e, "%r63", "%r64")
    e(" bar.sync 0;")
    e(" mov.u32 %r30, %r31;")
    e(" @%p1 bra LOOP;")
    # epilogue: rows by*128 + {ty*4 + i, 64 + ty*4 + i}, cols bx*128 + {tx*4, 64 + tx*4}
    e(" shl.b32 %r34, %r3, 7; shl.b32 %r35, %r5, 2; add.u32 %r34, %r34, %r35;")
    e(" shl.b32 %r36, %r2, 7; shl.b32 %r37, %r4, 2; add.u32 %r36, %r36, %r37;")
    e(" mad.lo.u32 %r38, %r34, %r42, %r36;")                     # (row0, col0)
    e(" shl.b32 %r39, %r42, 6;")                                  # 64 rows
    for half in range(2):
        for i in range(4):
            row = 4 * half + i
            if half == 0 and i == 0:
                e(" mov.u32 %r50, %r38;")
            elif i == 0:
                e(" add.u32 %r50, %r38, %r39;")
            else:
                e(" add.u32 %r50, %r50, %r42;")
            for cb in range(2):
                e(f" add.u32 %r51, %r50, {64 * cb}; mul.wide.u32 %rd10, %r51, 4; add.s64 %rd11, %rd3, %rd10;")
                fs = [f"%f{8 * row + 4 * cb + q}" for q in range(4)]
                e(f" st.global.v4.f32 [%rd11], {{{', '.join(fs)}}};")
    e(" ret;\n}")
    return "\n".join(L) + "\n"


def _gload128(e, k, kb):
    """A: row r6, k + r7 .. +3 (v4);  B: row k + r8, cols r14 .. +3 (v4)  ->  %g0..%g7."""
    e(f" add.u32 %r52, %r11, {k}; mul.wide.u32 %rd20, %r52, 4; add.s64 %rd21, %rd1, %rd20;")
    e(" ld.global.v4.f32 {%g0, %g1, %g2, %g3}, [%rd21];")
    e(f" add.u32 %r53, %r15, {kb}; mul.wide.u32 %rd22, %r53, 4; add.s64 %rd23, %rd2, %rd22;")
    e(" ld.global.v4.f32 {%g4, %g5, %g6, %g7}, [%rd23];")


def _sstore128(e, aa, ab):
    for q in range(4):
        e(f" st.shared.f32 [{aa}+{q * 512}], %g{q};")
    e(f" st.shared.v4.f32 [{ab}+4096], {{%g4, %g5, %g6, %g7}};")


def fgemm(ka: int, kb: int, kc: int, arch: str = "sm_80") -> str:
    """Fused rank-1 step of a bilinear scheme, on the sgemm128 tiling:

        for each output o:  C_o += w_o * (sum_s u_s A_s) (sum_t v_t B_t)

    The A/B linear combinations are formed in registers while a k-tile is loaded
    (ka / kb 128-bit global loads per thread instead of one), and the epilogue does a
    read-modify-write of every C_o (kc of them), so no temporaries S_r, T_r, M_r and no
    separate add kernels exist. All A_s share lda, all B_t ldb, all C_o ldc. Launches of
    one plan run in stream order, so the read-modify-writes of different steps on the
    same C block cannot race. C must be zeroed before the first step."""
    name = f"fgemm_{ka}_{kb}_{kc}"
    ps = [f".param .u64 pA{s}" for s in range(ka)] + [f".param .f32 cA{s}" for s in range(ka)]
    ps += [f".param .u64 pB{t}" for t in range(kb)] + [f".param .f32 cB{t}" for t in range(kb)]
    ps += [f".param .u64 pC{o}" for o in range(kc)] + [f".param .f32 cC{o}" for o in range(kc)]
    ps += [".param .u32 lda", ".param .u32 ldb", ".param .u32 ldc", ".param .u32 K"]
    L = []
    e = L.append
    e(HEADER.format(arch=arch))
    e(f".visible .entry {name}({', '.join(ps)})\n{{")
    e(" .reg .pred %p<4>; .reg .b32 %r<80>; .reg .f32 %f<64>; .reg .f32 %a<128>; .reg .f32 %b<128>;")
    e(" .reg .f32 %g<8>; .reg .f32 %h<8>; .reg .f32 %ca<32>; .reg .f32 %cb<32>; .reg .f32 %cc<32>;")
    e(" .reg .f32 %o<4>; .reg .b64 %rd<40>; .reg .b64 %pa<32>; .reg .b64 %pb<32>; .reg .b64 %pc<32>;")
    e(" .shared .align 16 .b8 S[16384];")
    for s in range(ka):
        e(f" ld.param.u64 %pa{s}, [pA{s}]; cvta.to.global.u64 %pa{s}, %pa{s}; ld.param.f32 %ca{s}, [cA{s}];")
    for t in range(kb):
        e(f" ld.param.u64 %pb{t}, [pB{t}]; cvta.to.global.u64 %pb{t}, %pb{t}; ld.param.f32 %cb{t}, [cB{t}];")
    e(" ld.param.u32 %r40, [lda]; ld.param.u32 %r41, [ldb]; ld.param.u32 %r42, [ldc]; ld.param.u32 %r43, [K];")
    e(" mov.u32 %r1, %tid.x; mov.u32 %r2, %ctaid.x; mov.u32 %r3, %ctaid.y;")
    e(" and.b32 %r4, %r1, 15; shr.u32 %r5, %r1, 4;")
    e(" shr.u32 %r6, %r1, 1; and.b32 %r7, %r1, 1; shl.b32 %r7, %r7, 2;")
    e(" shr.u32 %r8, %r1, 5; and.b32 %r9, %r1, 31; shl.b32 %r9, %r9, 2;")
    e(" shl.b32 %r10, %r3, 7; add.u32 %r10, %r10, %r6; mad.lo.u32 %r11, %r10, %r40, %r7;")
    e(" shl.b32 %r14, %r2, 7; add.u32 %r14, %r14, %r9; mad.lo.u32 %r15, %r8, %r41, %r14;")
    e(" mov.u32 %r18, S;")
    e(" shl.b32 %r20, %r7, 9; shl.b32 %r21, %r6, 2; add.u32 %r20, %r20, %r21; add.u32 %r20, %r20, %r18;")
    e(" shl.b32 %r22, %r8, 9; shl.b32 %r23, %r9, 2; add.u32 %r22, %r22, %r23; add.u32 %r22, %r22, %r18;")
    e(" shl.b32 %r24, %r5, 4; add.u32 %r24, %r24, %r18; shl.b32 %r25, %r4, 4; add.u32 %r25, %r25, %r18;")
    for i in range(64):
        e(f" mov.f32 %f{i}, 0f00000000;")
    _gload_comb(e, ka, kb, "0", "0")
    _sstore128(e, "%r20", "%r22")
    e(" bar.sync 0;")
    e(" mov.u32 %r30, 0; mov.u32 %r60, 0;")
    e(" LOOP:")
    e(" add.u32 %r31, %r30, 8; setp.lt.u32 %p1, %r31, %r43; selp.u32 %r32, %r31, %r30, %p1;")
    e(" mul.lo.u32 %r33, %r32, %r41;")
    _gload_comb(e, ka, kb, "%r32", "%r33")
    e(" add.u32 %r61, %r24, %r60; add.u32 %r62, %r25, %r60;")
    for k in range(8):
        a = [f"%a{16 * k + i}" for i in range(8)]
        b = [f"%b{16 * k + i}" for i in range(8)]
        e(f" ld.shared.v4.f32 {{{', '.join(a[:4])}}}, [%r61+{k * 512}];")
        e(f" ld.shared.v4.f32 {{{', '.join(a[4:])}}}, [%r61+{k * 512 + 256}];")
        e(f" ld.shared.v4.f32 {{{', '.join(b[:4])}}}, [%r62+{4096 + k * 512}];")
        e(f" ld.shared.v4.f32 {{{', '.join(b[4:])}}}, [%r62+{4096 + k * 512 + 256}];")
        for i in range(8):
            for j in range(8):
                e(f" fma.rn.f32 %f{8 * i + j}, {a[i]}, {b[j]}, %f{8 * i + j};")
    e(" xor.b32 %r60, %r60, 8192;")
    e(" add.u32 %r63, %r20, %r60; add.u32 %r64, %r22, %r60;")
    _sstore128(e, "%r63", "%r64")
    e(" bar.sync 0;")
    e(" mov.u32 %r30, %r31;")
    e(" @%p1 bra LOOP;")
    e(" shl.b32 %r34, %r3, 7; shl.b32 %r35, %r5, 2; add.u32 %r34, %r34, %r35;")
    e(" shl.b32 %r36, %r2, 7; shl.b32 %r37, %r4, 2; add.u32 %r36, %r36, %r37;")
    e(" mad.lo.u32 %r38, %r34, %r42, %r36;")
    e(" shl.b32 %r39, %r42, 6;")
    for o in range(kc):
        e(f" ld.param.u64 %pc{o}, [pC{o}]; cvta.to.global.u64 %pc{o}, %pc{o}; ld.param.f32 %cc{o}, [cC{o}];")
    for half in range(2):
        for i in range(4):
            row = 4 * half + i
            if half == 0 and i == 0:
                e(" mov.u32 %r50, %r38;")
            elif i == 0:
                e(" add.u32 %r50, %r38, %r39;")
            else:
                e(" add.u32 %r50, %r50, %r42;")
            for cb in range(2):
                e(f" add.u32 %r51, %r50, {64 * cb}; mul.wide.u32 %rd10, %r51, 4;")
                fs = [f"%f{8 * row + 4 * cb + q}" for q in range(4)]
                for o in range(kc):
                    e(f" add.s64 %rd11, %pc{o}, %rd10;")
                    e(" ld.global.v4.f32 {%o0, %o1, %o2, %o3}, [%rd11];")
                    for q in range(4):
                        e(f" fma.rn.f32 %o{q}, {fs[q]}, %cc{o}, %o{q};")
                    e(" st.global.v4.f32 [%rd11], {%o0, %o1, %o2, %o3};")
    e(" ret;\n}")
    return "\n".join(L) + "\n"


def _gload_comb(e, ka, kb, k, kb_off):
    """%g0..3 = sum_s cA_s * A_s[row, k + r7 .. +3];  %g4..7 = sum_t cB_t * B_t[k + r8, cols]."""
    e(f" add.u32 %r52, %r11, {k}; mul.wide.u32 %rd20, %r52, 4;")
    e(f" add.u32 %r53, %r15, {kb_off}; mul.wide.u32 %rd22, %r53, 4;")
    for side, n, base, coef, off, g0 in (("a", ka, "%pa", "%ca", "%rd20", 0), ("b", kb, "%pb", "%cb", "%rd22", 4)):
        for s in range(n):
            e(f" add.s64 %rd21, {base}{s}, {off};")
            dst = [f"%g{g0 + q}" for q in range(4)] if s == 0 else [f"%h{q}" for q in range(4)]
            e(f" ld.global.v4.f32 {{{', '.join(dst)}}}, [%rd21];")
            if s == 0:
                for q in range(4):        # x * c is exact for c = +-1, +-2^k: no rounding added
                    e(f" mul.rn.f32 %g{g0 + q}, %g{g0 + q}, {coef}0;")
            else:
                for q in range(4):
                    e(f" fma.rn.f32 %g{g0 + q}, %h{q}, {coef}{s}, %g{g0 + q};")
