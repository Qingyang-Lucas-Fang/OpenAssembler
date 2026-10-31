"""Generate PTX probe kernels that cover the SASS instruction space.

Each probe is one small kernel. Inputs come from global memory (so ptxas
can't constant-fold), one operand optionally comes from a kernel param (which
ptxas turns into a c[0x0][...] operand) or an immediate, and results are stored
back so nothing is dead. Body snippets see these registers:

  %r1..%r4  b32 inputs        %f1..%f3  f32 (bit-cast of %r1..%r3)
  %rd6 %rd7 b64 inputs        %fd1 %fd2 f64 (bit-cast of %rd6 %rd7)
  %rk       b32 kernel param  %fk       f32 kernel param
  %rd4      global input ptr (thread's 16-byte slot)
  %rd5      global output ptr; bytes 0..7 are the result, 8..15 are scratch for store probes
  %r0       %tid.x

and must leave the result in %r10 (and optionally %r11).
"""

from __future__ import annotations

import itertools
import re

HEADER = """.version 8.7
.target {target}
.address_size 64
"""

KERNEL = """
.visible .entry {name}(.param .u64 out, .param .u64 in, .param .u32 k, .param .f32 kf)
{{
 .reg .pred %p<8>; .reg .b16 %h<16>; .reg .b32 %r<64>; .reg .f32 %f<32>;
 .reg .b64 %rd<32>; .reg .f64 %fd<16>; .reg .b32 %rk; .reg .f32 %fk;
 {shared}
 ld.param.u64 %rd1, [in]; ld.param.u64 %rd2, [out];
 ld.param.u32 %rk, [k]; ld.param.f32 %fk, [kf];
 cvta.to.global.u64 %rd1, %rd1; cvta.to.global.u64 %rd2, %rd2;
 mov.u32 %r0, %tid.x; mul.wide.u32 %rd3, %r0, 16;
 add.s64 %rd4, %rd1, %rd3; add.s64 %rd5, %rd2, %rd3;
 ld.global.v4.u32 {{%r1, %r2, %r3, %r4}}, [%rd4];
 mov.b32 %f1, %r1; mov.b32 %f2, %r2; mov.b32 %f3, %r3;
 mov.b64 %rd6, {{%r1, %r2}}; mov.b64 %rd7, {{%r3, %r4}};
 mov.b64 %fd1, %rd6; mov.b64 %fd2, %rd7;
 mov.u32 %r11, %r4;
{body}
 st.global.v2.u32 [%rd5], {{%r10, %r11}};
 ret;
}}
"""

# Second-operand variants: register, kernel param (-> cbank), immediate.
B32 = {"r": "%r2", "c": "%rk", "i": "0x1234"}
BF32 = {"r": "%f2", "c": "%fk", "i": "0f3FC00000"}


def _int_ops():
    for op, types in {
        "add": ["s32", "u32"], "sub": ["s32"], "mul.lo": ["s32", "u32"],
        "mul.hi": ["s32", "u32"], "min": ["s32", "u32"], "max": ["s32", "u32"],
        "and": ["b32"], "or": ["b32"], "xor": ["b32"],
        "shl": ["b32"], "shr": ["u32", "s32"], "div": ["s32", "u32"], "rem": ["s32", "u32"],
    }.items():
        for t, (v, b) in itertools.product(types, B32.items()):
            yield f"{op}.{t}.{v}", f" {op}.{t} %r10, %r1, {b};"
    for t, (v, b) in itertools.product(["s32", "u32"], B32.items()):
        yield f"mad.lo.{t}.{v}", f" mad.lo.{t} %r10, %r1, {b}, %r3;"
        yield f"mad.hi.{t}.{v}", f" mad.hi.{t} %r10, %r1, {b}, %r3;"
        yield f"mul.wide.{t}.{v}", f" mul.wide.{t} %rd10, %r1, {b}; mov.b64 {{%r10, %r11}}, %rd10;"
    for op in ["add", "sub", "mul.lo", "min", "max", "and", "or", "xor"]:
        t = "b64" if op in ("and", "or", "xor") else "s64"
        yield f"{op}.{t}", f" {op}.{t} %rd10, %rd6, %rd7; mov.b64 {{%r10, %r11}}, %rd10;"
    yield "shl.b64", " shl.b64 %rd10, %rd6, %r3; mov.b64 {%r10, %r11}, %rd10;"
    yield "shr.u64", " shr.u64 %rd10, %rd6, %r3; mov.b64 {%r10, %r11}, %rd10;"
    yield "shf.l", " shf.l.wrap.b32 %r10, %r1, %r2, %r3;"
    yield "shf.r", " shf.r.clamp.b32 %r10, %r1, %r2, %r3;"
    yield "not", " not.b32 %r10, %r1;"
    yield "neg", " neg.s32 %r10, %r1;"
    yield "abs", " abs.s32 %r10, %r1;"
    yield "popc", " popc.b32 %r10, %r1;"
    yield "clz", " clz.b32 %r10, %r1;"
    yield "brev", " brev.b32 %r10, %r1;"
    yield "bfe", " bfe.u32 %r10, %r1, %r2, %r3;"
    yield "bfi", " bfi.b32 %r10, %r1, %r2, %r3, %r4;"
    yield "prmt", " prmt.b32 %r10, %r1, %r2, 0x3210;"
    yield "prmt.r", " prmt.b32 %r10, %r1, %r2, %r3;"
    yield "lop3", " lop3.b32 %r10, %r1, %r2, %r3, 0x96;"
    yield "add.cc", " add.cc.u32 %r10, %r1, %r2; addc.u32 %r11, %r3, %r4;"
    yield "dp4a", " dp4a.u32.u32 %r10, %r1, %r2, %r3;"
    yield "sad", " sad.u32 %r10, %r1, %r2, %r3;"


def _cmp_ops():
    for cmp, t, (v, b) in itertools.product(
        ["eq", "ne", "lt", "le", "gt", "ge"], ["s32", "u32"], B32.items()
    ):
        yield f"setp.{cmp}.{t}.{v}", f" setp.{cmp}.{t} %p1, %r1, {b}; selp.b32 %r10, %r3, %r4, %p1;"
    for cmp, (v, b) in itertools.product(
        ["eq", "ne", "lt", "le", "gt", "ge", "equ", "neu", "ltu", "geu", "num", "nan"], BF32.items()
    ):
        yield f"setp.{cmp}.f32.{v}", f" setp.{cmp}.f32 %p1, %f1, {b}; selp.b32 %r10, %r3, %r4, %p1;"
    for cmp in ["lt", "ge", "ne"]:
        yield f"setp.{cmp}.s64", f" setp.{cmp}.s64 %p1, %rd6, %rd7; selp.b32 %r10, %r3, %r4, %p1;"
        yield f"setp.{cmp}.f64", f" setp.{cmp}.f64 %p1, %fd1, %fd2; selp.b32 %r10, %r3, %r4, %p1;"
    for comb in ["and", "or", "xor"]:
        yield f"setp.{comb}", (f" setp.lt.s32 %p2, %r3, %r4; setp.gt.{comb}.s32 %p1, %r1, %r2, %p2;"
                               " selp.b32 %r10, %r3, %r4, %p1;")
    yield "setp.2dst", " setp.lt.s32 %p1|%p2, %r1, %r2; selp.b32 %r10, %r3, %r4, %p1; selp.b32 %r11, %r3, %r4, %p2;"
    yield "pred.logic", (" setp.lt.s32 %p1, %r1, %r2; setp.lt.s32 %p2, %r3, %r4; and.pred %p3, %p1, %p2;"
                         " xor.pred %p4, %p3, %p1; not.pred %p5, %p4; selp.b32 %r10, %r3, %r4, %p5;")
    yield "set.u32", " set.lt.u32.s32 %r10, %r1, %r2;"
    yield "slct", " slct.s32.s32 %r10, %r1, %r2, %r3;"


def _float_ops():
    for op, mods in {
        "add": ["", ".rn", ".rz", ".rm", ".rp", ".sat", ".ftz"], "sub": [""],
        "mul": ["", ".rz", ".sat"], "min": ["", ".NaN"], "max": [""],
    }.items():
        for m, (v, b) in itertools.product(mods, BF32.items()):
            yield f"{op}{m}.f32.{v}", f" {op}{m}.f32 %f10, %f1, {b}; mov.b32 %r10, %f10;"
    for m, (v, b) in itertools.product([".rn", ".rz", ".rn.sat"], BF32.items()):
        yield f"fma{m}.f32.{v}", f" fma{m}.f32 %f10, %f1, {b}, %f3; mov.b32 %r10, %f10;"
    for op in ["rcp.approx", "sqrt.approx", "rsqrt.approx", "ex2.approx", "lg2.approx",
               "sin.approx", "cos.approx", "tanh.approx", "rcp.rn", "sqrt.rn"]:
        yield f"{op}.f32", f" {op}.f32 %f10, %f1; mov.b32 %r10, %f10;"
    yield "div.rn.f32", " div.rn.f32 %f10, %f1, %f2; mov.b32 %r10, %f10;"
    yield "div.approx.f32", " div.approx.f32 %f10, %f1, %f2; mov.b32 %r10, %f10;"
    yield "abs.f32", " abs.f32 %f10, %f1; mov.b32 %r10, %f10;"
    yield "neg.f32", " neg.f32 %f10, %f1; mov.b32 %r10, %f10;"
    for op in ["add", "mul", "min", "max", "div.rn", "sqrt.rn", "rcp.rn"]:
        args = "%fd1" if op in ("sqrt.rn", "rcp.rn") else "%fd1, %fd2"
        yield f"{op}.f64", f" {op}.f64 %fd10, {args}; mov.b64 {{%r10, %r11}}, %fd10;"
    yield "fma.f64", " fma.rn.f64 %fd10, %fd1, %fd2, %fd1; mov.b64 {%r10, %r11}, %fd10;"
    for t in ["f16x2", "bf16x2"]:
        for op in ["add", "mul", "min", "max"]:
            if t == "bf16x2" and op in ("add", "mul"):
                continue  # add/mul.bf16x2 need sm_90; ptxas lowers them to fma there
            yield f"{op}.{t}", f" {op}.{t} %r10, %r1, %r2;"
        yield f"fma.{t}", f" fma.rn.{t} %r10, %r1, %r2, %r3;"
    yield "add.f16", " mov.b32 {%h1, %h2}, %r1; mov.b32 {%h3, %h4}, %r2; add.f16 %h5, %h1, %h3; mov.b32 %r10, {%h5, %h2};"
    yield "testp", " testp.finite.f32 %p1, %f1; selp.b32 %r10, %r3, %r4, %p1;"
    yield "copysign", " copysign.f32 %f10, %f1, %f2; mov.b32 %r10, %f10;"


def _cvt_ops():
    ints = ["s8", "u8", "s16", "u16", "s32", "u32", "s64", "u64"]
    for d, s in itertools.product(["s32", "u32", "s64"], ints):
        src = "%rd6" if s.endswith("64") else "%r1"
        dst = "%rd10" if d.endswith("64") else "%r10"
        post = " mov.b64 {%r10, %r11}, %rd10;" if d.endswith("64") else ""
        yield f"cvt.{d}.{s}", f" cvt.{d}.{s} {dst}, {src};{post}"
    for rnd, t in itertools.product(["rni", "rzi", "rmi", "rpi"], ["s32", "u32", "s64"]):
        dst, post = ("%rd10", " mov.b64 {%r10, %r11}, %rd10;") if t == "s64" else ("%r10", "")
        yield f"cvt.{rnd}.{t}.f32", f" cvt.{rnd}.{t}.f32 {dst}, %f1;{post}"
        yield f"cvt.{rnd}.{t}.f64", f" cvt.{rnd}.{t}.f64 {dst}, %fd1;{post}"
    for rnd, t in itertools.product(["rn", "rz"], ["s32", "u32", "s64", "u64"]):
        src = "%rd6" if t.endswith("64") else "%r1"
        yield f"cvt.{rnd}.f32.{t}", f" cvt.{rnd}.f32.{t} %f10, {src}; mov.b32 %r10, %f10;"
        yield f"cvt.{rnd}.f64.{t}", f" cvt.{rnd}.f64.{t} %fd10, {src}; mov.b64 {{%r10, %r11}}, %fd10;"
    yield "cvt.f64.f32", " cvt.f64.f32 %fd10, %f1; mov.b64 {%r10, %r11}, %fd10;"
    yield "cvt.rn.f32.f64", " cvt.rn.f32.f64 %f10, %fd1; mov.b32 %r10, %f10;"
    yield "cvt.rni.f32.f32", " cvt.rni.f32.f32 %f10, %f1; mov.b32 %r10, %f10;"
    yield "cvt.rn.f16.f32", " cvt.rn.f16.f32 %h1, %f1; mov.b32 %r10, {%h1, %h1};"
    yield "cvt.f32.f16", " mov.b32 {%h1, %h2}, %r1; cvt.f32.f16 %f10, %h1; mov.b32 %r10, %f10;"
    yield "cvt.rn.f16x2.f32", " cvt.rn.f16x2.f32 %r10, %f1, %f2;"
    yield "cvt.rn.bf16x2.f32", " cvt.rn.bf16x2.f32 %r10, %f1, %f2;"


SHARED = ".shared .align 16 .b8 smem[4096];"


def _mem_ops():
    for t in ["u8", "s8", "u16", "s16", "b32", "b64", "v2.b32", "v4.b32"]:
        dst = {"b64": "%rd10", "v2.b32": "{%r10, %r11}", "v4.b32": "{%r10, %r11, %r12, %r13}"}.get(t, "%r10")
        post = " mov.b64 {%r10, %r11}, %rd10;" if t == "b64" else ""
        for sp in ["global", "global.nc", "global.cg", "global.cs", "global.lu", "global.ca"]:
            yield f"ld.{sp}.{t}", f" ld.{sp}.{t} {dst}, [%rd4+16];{post}"
        yield f"ld.shared.{t}", (f" mov.u32 %r20, smem; shl.b32 %r21, %r0, 4; add.u32 %r20, %r20, %r21;"
                                 f" st.shared.u32 [%r20], %r1; bar.sync 0; ld.shared.{t} {dst}, [%r20+16];{post}"), True
        yield f"ld.local.{t}", (f" .local .align 16 .b8 lbuf[64]; mov.u64 %rd20, lbuf; cvt.u32.u64 %r20, %rd20;"
                                f" and.b32 %r21, %r1, 3; shl.b32 %r21, %r21, 4; add.u32 %r20, %r20, %r21;"
                                f" cvt.u64.u32 %rd20, %r20; st.local.u32 [%rd20], %r2; ld.local.{t} {dst}, [%rd20];{post}")
    for t in ["u8", "u16", "b32", "b64", "v2.b32", "v4.b32"]:
        src = {"b64": "%rd6", "v2.b32": "{%r1, %r2}", "v4.b32": "{%r1, %r2, %r3, %r4}"}.get(t, "%r1")
        if t in ("u8", "u16"):
            yield f"st.global.{t}", f" cvt.{t}.u32 %h1, %r1; st.global.{t} [%rd5+8], %h1; mov.u32 %r10, 0;"
        else:
            # scratch: 16-byte aligned area past all 32 output slots (vector stores need alignment)
            yield f"st.global.{t}", (f" mul.wide.u32 %rd21, %r0, 16; add.s64 %rd21, %rd2, %rd21;"
                                     f" st.global.{t} [%rd21+0x800], {src}; mov.u32 %r10, 0;")
            # per-lane 16-byte slot; read back the first word this lane stored
            yield f"st.shared.{t}", (f" mov.u32 %r20, smem; shl.b32 %r21, %r0, 4; add.u32 %r20, %r20, %r21;"
                                     f" st.shared.{t} [%r20+64], {src}; bar.sync 0; ld.shared.u32 %r10, [%r20+64];"), True
    yield "ld.const", " .const .align 4 .b32 ctab[16]; mov.u64 %rd20, ctab; and.b32 %r20, %r1, 15; mul.wide.u32 %rd21, %r20, 4; add.s64 %rd20, %rd20, %rd21; ld.const.u32 %r10, [%rd20];"
    for op in ["add.u32", "min.s32", "max.u32", "inc.u32", "dec.u32", "and.b32", "or.b32", "xor.b32", "exch.b32"]:
        yield f"atom.global.{op}", f" atom.global.{op} %r10, [%rd4], %r2;"
        yield f"atom.shared.{op}", (f" mov.u32 %r20, smem; shl.b32 %r21, %r0, 2; add.u32 %r20, %r20, %r21;"
                                    f" st.shared.u32 [%r20], %r3; atom.shared.{op} %r10, [%r20], %r2;"
                                    f" ld.shared.u32 %r11, [%r20];"), True
        if op != "exch.b32":  # red has no .exch
            yield f"red.global.{op}", f" red.global.{op} [%rd4], %r2; mov.u32 %r10, 0;"
    yield "atom.cas.b32", " atom.global.cas.b32 %r10, [%rd4], %r2, %r3;"
    yield "atom.cas.b64", " atom.global.cas.b64 %rd10, [%rd4], %rd6, %rd7; mov.b64 {%r10, %r11}, %rd10;"
    yield "atom.add.u64", " atom.global.add.u64 %rd10, [%rd4], %rd6; mov.b64 {%r10, %r11}, %rd10;"
    yield "atom.add.f32", " atom.global.add.f32 %f10, [%rd4], %f2; mov.b32 %r10, %f10;"
    yield "atom.shared.add.f32", (" mov.u32 %r20, smem; shl.b32 %r21, %r0, 2; add.u32 %r20, %r20, %r21;"
                                  " st.shared.u32 [%r20], %r3; atom.shared.add.f32 %f10, [%r20], %f2;"
                                  " mov.b32 %r10, %f10; ld.shared.u32 %r11, [%r20];"), True
    yield "atom.add.f64", " atom.global.add.f64 %fd10, [%rd4], %fd2; mov.b64 {%r10, %r11}, %fd10;"
    yield "atom.sys", " atom.global.sys.add.u32 %r10, [%rd4], %r2;"
    for sem in ["acquire.gpu", "relaxed.gpu", "acq_rel.sys"]:
        yield f"atom.{sem}", f" atom.{sem}.global.add.u32 %r10, [%rd4], %r2;"
    for f in ["fence.sc.gpu", "fence.acq_rel.gpu", "fence.acq_rel.sys", "membar.cta", "membar.gl", "membar.sys"]:
        yield f, f" st.global.u32 [%rd5+8], %r1; {f}; ld.volatile.global.u32 %r10, [%rd4+16];"
    yield "ld.volatile", " ld.volatile.global.u32 %r10, [%rd4+16];"
    yield "ld.acquire", " ld.acquire.gpu.global.u32 %r10, [%rd4+16];"
    yield "st.release", " st.release.gpu.global.u32 [%rd5+8], %r1; mov.u32 %r10, 0;"
    yield "prefetch", " prefetch.global.L2 [%rd4+64]; mov.u32 %r10, %r1;"
    yield "cvta.shared", " mov.u64 %rd20, smem; cvta.shared.u64 %rd21, %rd20; mov.b64 {%r10, %r11}, %rd21;", True
    yield "isspacep", " isspacep.shared %p1, %rd4; selp.b32 %r10, 1, 0, %p1;"


def _warp_ops():
    for mode in ["idx", "up", "down", "bfly"]:
        for v, b in {"r": "%r2", "i": "3"}.items():
            yield f"shfl.{mode}.{v}", f" shfl.sync.{mode}.b32 %r10, %r1, {b}, 0x1f, 0xffffffff;"
    yield "shfl.p", " shfl.sync.bfly.b32 %r10|%p1, %r1, 1, 0x1f, 0xffffffff; selp.b32 %r11, 1, 0, %p1;"
    for m in ["all", "any", "uni"]:
        yield f"vote.{m}", f" setp.lt.u32 %p1, %r1, %r2; vote.sync.{m}.pred %p2, %p1, 0xffffffff; selp.b32 %r10, 1, 0, %p2;"
    yield "vote.ballot", " setp.lt.u32 %p1, %r1, %r2; vote.sync.ballot.b32 %r10, %p1, 0xffffffff;"
    yield "match.any", " match.any.sync.b32 %r10, %r1, 0xffffffff;"
    yield "match.all", " match.all.sync.b32 %r10|%p1, %r1, 0xffffffff;"
    for op in ["add.u32", "min.s32", "max.u32", "and.b32", "or.b32", "xor.b32"]:
        yield f"redux.{op}", f" redux.sync.{op} %r10, %r1, 0xffffffff;"
    yield "activemask", " activemask.b32 %r10;"
    yield "bar.sync", (" mov.u32 %r20, smem; shl.b32 %r21, %r0, 2; add.u32 %r22, %r20, %r21; st.shared.u32 [%r22], %r1;"
                     " bar.sync 0; xor.b32 %r21, %r21, 4; add.u32 %r22, %r20, %r21; ld.shared.u32 %r10, [%r22];"), True
    yield "bar.sync.n", (" mov.u32 %r20, smem; shl.b32 %r21, %r0, 2; add.u32 %r22, %r20, %r21; st.shared.u32 [%r22], %r1;"
                       " bar.sync 1, 32; xor.b32 %r21, %r21, 4; add.u32 %r22, %r20, %r21; ld.shared.u32 %r10, [%r22];"), True
    yield "bar.arrive", " bar.arrive 1, 32; mov.u32 %r10, %r1;"
    yield "bar.red.popc", " setp.lt.u32 %p1, %r1, %r2; bar.red.popc.u32 %r10, 0, %p1;"
    yield "bar.red.and", " setp.lt.u32 %p1, %r1, %r2; bar.red.and.pred %p2, 0, %p1; selp.b32 %r10, 1, 0, %p2;"
    yield "bar.warp.sync", " bar.warp.sync 0xffffffff; mov.u32 %r10, %r1;"
    sregs = ["%tid.y", "%tid.z", "%ntid.x", "%ntid.y", "%ctaid.x", "%ctaid.y", "%ctaid.z", "%nctaid.x",
             "%laneid", "%warpid", "%nwarpid", "%smid", "%nsmid", "%gridid", "%lanemask_eq", "%lanemask_lt",
             "%lanemask_ge", "%clock", "%dynamic_smem_size", "%total_smem_size"]
    for s in sregs:
        yield f"sreg.{s[1:]}", f" mov.u32 %r10, {s};"
    yield "sreg.clock64", " mov.u64 %rd10, %clock64; mov.b64 {%r10, %r11}, %rd10;"
    yield "sreg.globaltimer", " mov.u64 %rd10, %globaltimer; mov.b64 {%r10, %r11}, %rd10;"
    yield "nanosleep", " nanosleep.u32 100; mov.u32 %r10, %r1;"
    yield "trap", " setp.eq.u32 %p1, %r1, 0xdead; @%p1 trap; mov.u32 %r10, %r1;"
    yield "brkpt", " setp.eq.u32 %p1, %r1, 0xdead; @%p1 brkpt; mov.u32 %r10, %r1;"
    yield "pmevent", " pmevent 1; mov.u32 %r10, %r1;"


def _ctrl_ops():
    yield "if", " mov.u32 %r10, %r1; setp.lt.u32 %p1, %r1, %r2; @%p1 bra L1; add.u32 %r10, %r10, %r3; L1:"
    yield "ifelse", (" setp.lt.u32 %p1, %r1, %r2; @%p1 bra LA; mul.lo.u32 %r10, %r1, %r3; bra LB;"
                     " LA: add.u32 %r10, %r2, %r4; LB:")
    # Trip counts are masked so emulated runs stay short (random inputs).
    yield "loop", (" and.b32 %r1, %r1, 31; mov.u32 %r10, 0; mov.u32 %r20, 0; LOOP: add.u32 %r10, %r10, %r20; add.u32 %r20, %r20, 1;"
                   " setp.lt.u32 %p1, %r20, %r1; @%p1 bra LOOP;")
    yield "loop.nested", (" and.b32 %r1, %r1, 15; and.b32 %r2, %r2, 15; mov.u32 %r10, 0; mov.u32 %r20, 0; O: mov.u32 %r21, 0; I: xor.b32 %r10, %r10, %r21;"
                          " add.u32 %r21, %r21, 1; setp.lt.u32 %p1, %r21, %r2; @%p1 bra I; add.u32 %r20, %r20, 1;"
                          " setp.lt.u32 %p2, %r20, %r1; @%p2 bra O;")
    yield "divergent", (" mov.u32 %r10, %r1; and.b32 %r20, %r0, 1; setp.eq.u32 %p1, %r20, 0; @%p1 bra D1;"
                        " st.global.u32 [%rd5+8], %r1; atom.global.add.u32 %r10, [%rd4], 1; D1:"
                        " shfl.sync.bfly.b32 %r11, %r10, 1, 0x1f, 0xffffffff;")
    yield "brx", (" .global .align 8 .u64 jt[1]; and.b32 %r20, %r1, 3; ts: .branchtargets T0, T1, T2, T3;"
                  " brx.idx %r20, ts; T0: mov.u32 %r10, 10; bra TE; T1: mov.u32 %r10, 11; bra TE;"
                  " T2: mov.u32 %r10, 12; bra TE; T3: mov.u32 %r10, 13; TE:")
    yield "early.exit", " setp.eq.u32 %p1, %r1, 0; @%p1 ret; mov.u32 %r10, %r2;"
    yield "exit", " setp.eq.u32 %p1, %r1, 7; @%p1 exit; mov.u32 %r10, %r2;"


def _mma_ops(target: str):
    sm = int(target.removeprefix("sm_").rstrip("a"))
    if sm < 80:
        return
    yield "mma.f16", (" mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%f10,%f11,%f12,%f13},"
                      " {%r1,%r2,%r3,%r4}, {%r1,%r2}, {%f1,%f2,%f3,%f1}; mov.b32 %r10, %f10; mov.b32 %r11, %f13;")
    yield "mma.bf16", (" mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%f10,%f11,%f12,%f13},"
                       " {%r1,%r2,%r3,%r4}, {%r1,%r2}, {%f1,%f2,%f3,%f1}; mov.b32 %r10, %f10; mov.b32 %r11, %f13;")
    yield "mma.tf32", (" mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%f10,%f11,%f12,%f13},"
                       " {%r1,%r2,%r3,%r4}, {%r1,%r2}, {%f1,%f2,%f3,%f1}; mov.b32 %r10, %f10; mov.b32 %r11, %f13;")
    yield "mma.s8", (" mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%r10,%r11,%r12,%r13},"
                     " {%r1,%r2,%r3,%r4}, {%r1,%r2}, {%r1,%r2,%r3,%r4};")
    yield "mma.f64", (" mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%fd10,%fd11}, {%fd1}, {%fd2}, {%fd1,%fd2};"
                      " mov.b64 {%r10, %r11}, %fd10;")
    yield "ldmatrix", (" mov.u32 %r20, smem; shl.b32 %r21, %r0, 4; add.u32 %r20, %r20, %r21; st.shared.v4.u32 [%r20],"
                       " {%r1,%r2,%r3,%r4}; bar.sync 0; ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r10,%r11,%r12,%r13}, [%r20];"), True
    yield "ldmatrix.trans", (" mov.u32 %r20, smem; shl.b32 %r21, %r0, 4; add.u32 %r20, %r20, %r21; st.shared.v4.u32 [%r20],"
                             " {%r1,%r2,%r3,%r4}; bar.sync 0; ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%r10,%r11}, [%r20];"), True
    yield "cp.async", (" mov.u32 %r20, smem; shl.b32 %r21, %r0, 4; add.u32 %r20, %r20, %r21;"
                       " cp.async.cg.shared.global [%r20], [%rd4], 16; cp.async.commit_group; cp.async.wait_group 0;"
                       " bar.sync 0; ld.shared.u32 %r10, [%r20];"), True
    yield "cp.async.ca4", (" mov.u32 %r20, smem; shl.b32 %r21, %r0, 2; add.u32 %r20, %r20, %r21;"
                           " cp.async.ca.shared.global [%r20], [%rd4], 4; cp.async.wait_all; bar.sync 0; ld.shared.u32 %r10, [%r20];"), True


def probes(target: str):
    """Yield (name, body, needs_shared) for every probe valid on `target`."""
    for gen in (_int_ops(), _cmp_ops(), _float_ops(), _cvt_ops(), _mem_ops(), _warp_ops(),
                _ctrl_ops(), _mma_ops(target)):
        for p in gen:
            name, body, *rest = p
            yield name, body, bool(rest and rest[0])


def kernel_name(probe: str) -> str:
    return "p_" + "".join(c if c.isalnum() else "_" for c in probe)


# Only variable declarations (they carry .align); not e.g. "cp.async...shared.global [..]".
_MODULE_SCOPE = re.compile(r"\s*(\.(?:const|global)\s+\.align\s[^;]*;)")


def module(target: str, name: str, body: str, needs_shared: bool) -> str:
    """One standalone PTX module holding one probe kernel."""
    # .const/.global variables are only legal at module scope: hoist them.
    decls = "\n".join(_MODULE_SCOPE.findall(body))
    body = _MODULE_SCOPE.sub("", body)
    return HEADER.format(target=target) + decls + KERNEL.format(
        name=kernel_name(name), body=body, shared=SHARED if needs_shared else "")
