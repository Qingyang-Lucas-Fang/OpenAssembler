"""Isolated native SM80 NN-arithmetic campaign; no production mutations.

Build: .venv/bin/python exp/nn_round2/operator_suite.py --build
Worker interface: exp.nn_round2.operator_worker.run(payload, samples=15,
                                                  graph_repeats=32).
The matched compiler control is ptxas -O3 on the exact emitted PTX.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
HERE = ROOT / "exp/nn_round2"
BUILD = HERE / "operator_build"


def fp(value):
    return "0f" + struct.pack(">f", value).hex()


POINTWISE = {
    "residual_add": (2, 1, "float", (0., 0.)),
    "gated_multiply": (2, 1, "float", (0., 0.)),
    "relu": (1, 1, "float", (0., 0.)),
    "relu_backward": (2, 1, "float", (0., 0.)),
    "gelu_tanh": (1, 1, "float", (2e-5, 5e-6)),
    "gelu_tanh_backward": (2, 1, "float", (8e-5, 2e-5)),
    "silu": (1, 1, "float", (3e-5, 4e-6)),
    "silu_backward": (2, 1, "float", (5e-5, 1e-5)),
    "sigmoid": (1, 1, "float", (2e-5, 3e-6)),
    "tanh_backward": (2, 1, "float", (2e-5, 3e-6)),
    "divide": (2, 1, "float", (2e-6, 2e-7)),
    "rsqrt_scale": (2, 1, "float", (2e-5, 3e-6)),
    "dropout_mask_scale": (2, 1, "float", (0., 0.)),
    "rope_pairs": (4, 2, "float", (2e-5, 3e-6)),
    "sgd_momentum": (3, 2, "float", (2e-5, 3e-6)),
    "adam_like": (4, 3, "float", (2e-5, 2e-6)),
    "div_const7": (1, 1, "integer", (0., 0.)),
    "rem_const7": (1, 1, "integer", (0., 0.)),
}
ROWWISE = {
    "softmax": (1, 1, (5e-5, 1e-6)),
    "softmax_backward": (2, 1, (1e-4, 1e-5)),
    "layernorm": (1, 1, (1e-4, 2e-5)),
    "layernorm_backward": (2, 1, (2e-4, 5e-5)),
    "rmsnorm": (1, 1, (1e-4, 2e-5)),
    "rmsnorm_backward": (2, 1, (2e-4, 5e-5)),
    "cross_entropy": (1, 2, (1e-4, 1e-5)),
    "row_sum": (1, 1, (1e-4, 2e-5)),
}


def arithmetic(op, j):
    """An explicit FP32 sequence. No implicit contraction or policy rewrites."""
    x,z,w,v = (f"%f{32*k+j}" for k in range(4))
    y,mo,vo = (f"%f{128+32*k+j}" for k in range(3))
    a,b,c,d,e,t = (f"%f{224+8*j+k}" for k in range(6))
    zero,one = fp(0),fp(1)
    mul=lambda dst,l,r:f"mul.f32 {dst},{l},{r};"
    add=lambda dst,l,r:f"add.f32 {dst},{l},{r};"
    sub=lambda dst,l,r:f"sub.f32 {dst},{l},{r};"
    if op=="residual_add": return [add(y,x,z)]
    if op=="gated_multiply": return [mul(y,x,z)]
    if op=="relu": return [f"max.f32 {y},{x},{zero};"]
    if op=="relu_backward": return [f"setp.gt.f32 %p1,{x},{zero};",f"selp.f32 {y},{z},{zero},%p1;"]
    if op in ("gelu_tanh","gelu_tanh_backward"):
        seq=[mul(a,x,x),mul(b,a,x),mul(b,b,fp(.044715)),add(b,b,x),mul(b,b,fp(.7978845608028654)),f"tanh.approx.f32 {c},{b};",add(d,c,one)]
        if op=="gelu_tanh": return seq+[mul(d,d,fp(.5)),mul(y,x,d)]
        return seq+[mul(d,d,fp(.5)),mul(e,c,c),sub(e,one,e),mul(a,a,fp(3*.044715)),add(a,a,one),mul(a,a,fp(.7978845608028654)),mul(a,a,e),mul(a,a,x),mul(a,a,fp(.5)),add(a,a,d),mul(y,z,a)]
    if op in ("sigmoid","silu","silu_backward"):
        seq=[mul(a,x,fp(-1.4426950408889634)),f"ex2.approx.ftz.f32 {a},{a};",add(a,a,one),f"rcp.approx.ftz.f32 {b},{a};"]
        if op=="sigmoid":return seq+[f"mov.f32 {y},{b};"]
        if op=="silu":return seq+[mul(y,x,b)]
        return seq+[sub(c,one,b),mul(c,c,x),add(c,c,one),mul(c,c,b),mul(y,c,z)]
    if op=="tanh_backward": return [mul(a,x,x),sub(a,one,a),mul(y,a,z)]
    if op=="divide":return [f"div.rn.f32 {y},{x},{z};"]
    if op=="rsqrt_scale":return [f"rsqrt.approx.ftz.f32 {a},{z};",mul(y,x,a)]
    if op=="dropout_mask_scale":return [mul(a,x,z),mul(y,a,fp(1.25))]
    if op=="rope_pairs":return [mul(a,x,w),mul(b,z,v),sub(y,a,b),mul(c,x,v),mul(d,z,w),add(mo,c,d)]
    if op=="sgd_momentum":return [mul(a,w,fp(.9)),add(mo,a,z),mul(b,mo,fp(.001)),sub(y,x,b)]
    if op=="adam_like":return [mul(a,w,fp(.9)),mul(b,z,fp(.1)),add(mo,a,b),mul(c,z,z),mul(c,c,fp(.001)),mul(d,v,fp(.999)),add(vo,d,c),f"sqrt.rn.f32 {e},{vo};",add(e,e,fp(1e-8)),mul(t,mo,fp(.001)),f"div.rn.f32 {t},{t},{e};",sub(y,x,t)]
    if op in ("div_const7","rem_const7"):
        return [f"{'div' if op.startswith('div') else 'rem'}.u32 %u{j},%u{32+j},7;"]
    raise ValueError(op)


def pointwise_source(op, mode):
    if mode not in ("scalar","scalar4","vector4"):raise ValueError("unknown pointwise layout")
    nin,nout,dtype,_=POINTWISE[op]
    ept=1 if mode=="scalar" else 4
    name=f"op_{op}_{mode}"
    lines=[f".visible .entry {name}(.param .u64 X0,.param .u64 X1,.param .u64 X2,.param .u64 X3,.param .u64 Y0,.param .u64 Y1,.param .u64 Y2,.param .u32 N) {{",
           ".reg .b32 %r<16>; .reg .b32 %u<64>; .reg .f32 %f<280>; .reg .b64 %d<32>; .reg .pred %p<4>;",
           "mov.u32 %r0,%ctaid.x;mov.u32 %r1,%ntid.x;mov.u32 %r2,%tid.x;mad.lo.u32 %r3,%r0,%r1,%r2;",
           f"mul.lo.u32 %r3,%r3,{ept};ld.param.u32 %r4,[N];setp.ge.u32 %p0,%r3,%r4;@%p0 bra done;mul.wide.u32 %d0,%r3,4;"]
    for k in range(nin):lines += [f"ld.param.u64 %d{1+k},[X{k}];add.u64 %d{8+k},%d{1+k},%d0;"]
    for k in range(nout):lines += [f"ld.param.u64 %d{5+k},[Y{k}];add.u64 %d{12+k},%d{5+k},%d0;"]
    for k in range(nin):
        regs=[f"%u{32+j}" if dtype=="integer" else f"%f{32*k+j}" for j in range(ept)]
        ty="u32" if dtype=="integer" else "f32"
        if mode=="vector4":lines += [f"ld.global.v4.{ty} {{{','.join(regs)}}},[%d{8+k}];"]
        else:lines += [f"ld.global.{ty} {reg},[%d{8+k}+{4*j}];" for j,reg in enumerate(regs)]
    for j in range(ept):lines += arithmetic(op,j)
    for k in range(nout):
        regs=[f"%u{j}" if dtype=="integer" else f"%f{128+32*k+j}" for j in range(ept)]
        ty="u32" if dtype=="integer" else "f32"
        if mode=="vector4":lines += [f"st.global.v4.{ty} [%d{12+k}],{{{','.join(regs)}}};"]
        else:lines += [f"st.global.{ty} [%d{12+k}+{4*j}],{reg};" for j,reg in enumerate(regs)]
    return name,"\n".join(lines+["done:ret;}"])+"\n"


def rowwise_source(op,width,warps):
    """Full-warp row reductions, explicit synchronization retained unchanged."""
    if type(width) is not int or not 1<=width<=1024:raise ValueError("width must be1..1024")
    if warps not in (4,8):raise ValueError("warps must be4or8")
    chunks=(width+31)//32
    nin,nout,_=ROWWISE[op]
    name=f"op_{op}_w{width}_c{warps}"
    lines=[f".visible .entry {name}(.param .u64 X0,.param .u64 X1,.param .u64 X2,.param .u64 X3,.param .u64 Y0,.param .u64 Y1,.param .u64 Y2,.param .u32 N) {{",
           ".reg .b32 %r<20>; .reg .b64 %d<24>; .reg .f32 %f<240>; .reg .pred %p<4>;",
           "mov.u32 %r0,%tid.x;mov.u32 %r1,%ctaid.x;and.b32 %r2,%r0,31;shr.u32 %r3,%r0,5;",
           f"mad.lo.u32 %r4,%r1,{warps},%r3;ld.param.u32 %r9,[N];setp.ge.u32 %p0,%r4,%r9;@%p0 bra done;",
           f"mad.lo.u32 %r5,%r4,{width},%r2;mul.wide.u32 %d0,%r5,4;"]
    for k in range(nin):lines += [f"ld.param.u64 %d{1+k},[X{k}];add.u64 %d{8+k},%d{1+k},%d0;"]
    for k in range(nout):lines += [f"ld.param.u64 %d{5+k},[Y{k}];add.u64 %d{12+k},%d{5+k},%d0;"]
    tail=width%32
    if tail:lines += [f"setp.lt.u32 %p1,%r2,{tail};"]
    for k in range(nin):
        for j in range(chunks):
            dst=f"%f{32*k+j}"
            if tail and j==chunks-1:
                neutral="0fff800000" if k==0 and op in ("softmax","cross_entropy") else fp(0)
                lines += [f"mov.f32 {dst},{neutral};",f"@%p1 ld.global.f32 {dst},[%d{8+k}+{128*j}];"]
            else:lines += [f"ld.global.f32 {dst},[%d{8+k}+{128*j}];"]
    def reduce(kind,regs,dst):
        lines.append(f"mov.f32 %f{dst},%f{regs[0]};")
        for reg in regs[1:]:lines.append(f"{kind}.f32 %f{dst},%f{dst},%f{reg};")
        for delta in (16,8,4,2,1):lines.extend([f"shfl.sync.bfly.b32 %f239,%f{dst},{delta},31,0xffffffff;",f"{kind}.f32 %f{dst},%f{dst},%f239;"])
    def neutral_tail(reg):
        if tail:lines.append(f"@!%p1 mov.f32 %f{reg+chunks-1},{fp(0)};")
    if op in ("softmax","cross_entropy"):
        reduce("max",list(range(chunks)),160)
        for j in range(chunks):lines += [f"sub.f32 %f{64+j},%f{j},%f160;",f"mul.f32 %f{64+j},%f{64+j},{fp(1.4426950408889634)};",f"ex2.approx.ftz.f32 %f{64+j},%f{64+j};"]
        reduce("add",list(range(64,64+chunks)),161)
        lines.append("rcp.approx.ftz.f32 %f162,%f161;")
        for j in range(chunks):lines.append(f"mul.f32 %f{128+j},%f{64+j},%f162;")
        if op=="cross_entropy":
            # Fixed label zero: loss written only by lane zero, gradient for all columns.
            lines += [f"setp.eq.u32 %p2,%r2,0;", "lg2.approx.ftz.f32 %f164,%f161;",f"mul.f32 %f164,%f164,{fp(.6931471805599453)};","add.f32 %f164,%f164,%f160;","sub.f32 %f164,%f164,%f0;","mul.wide.u32 %d20,%r4,4;add.u64 %d21,%d6,%d20;@%p2 st.global.f32 [%d21],%f164;",f"@%p2 sub.f32 %f128,%f128,{fp(1)};"]
    elif op=="softmax_backward":
        # X is normalized probabilities; Z is dY.
        for j in range(chunks):lines.append(f"mul.f32 %f{64+j},%f{j},%f{32+j};")
        reduce("add",list(range(64,64+chunks)),160)
        for j in range(chunks):lines += [f"sub.f32 %f{96+j},%f{32+j},%f160;",f"mul.f32 %f{128+j},%f{j},%f{96+j};"]
    elif op=="row_sum":
        reduce("add",list(range(chunks)),160)
        lines += ["setp.eq.u32 %p2,%r2,0;mul.wide.u32 %d20,%r4,4;add.u64 %d21,%d5,%d20;@%p2 st.global.f32 [%d21],%f160;"]
    else:
        layer=op.startswith("layernorm")
        backward=op.endswith("backward")
        if layer:
            # Shift by the first element for stable small variance at large offsets.
            lines += [f"mul.lo.u32 %r6,%r4,{width};mul.wide.u32 %d20,%r6,4;add.u64 %d21,%d1,%d20;ld.global.f32 %f165,[%d21];"]
            for j in range(chunks):lines.append(f"sub.f32 %f{j},%f{j},%f165;")
            neutral_tail(0);reduce("add",list(range(chunks)),160)
            lines.append(f"mul.f32 %f160,%f160,{fp(1/width)};")
            for j in range(chunks):lines.append(f"sub.f32 %f{j},%f{j},%f160;")
            neutral_tail(0)
        for j in range(chunks):lines.append(f"mul.f32 %f{64+j},%f{j},%f{j};")
        reduce("add",list(range(64,64+chunks)),161)
        lines += [f"mul.f32 %f161,%f161,{fp(1/width)};",f"add.f32 %f161,%f161,{fp(1e-5)};","rsqrt.approx.ftz.f32 %f162,%f161;"]
        if backward:
            if layer:
                reduce("add",list(range(32,32+chunks)),163)
                lines.append(f"mul.f32 %f163,%f163,{fp(1/width)};")
            for j in range(chunks):lines.append(f"mul.f32 %f{96+j},%f{j},%f{32+j};")
            reduce("add",list(range(96,96+chunks)),164)
            lines += [f"mul.f32 %f164,%f164,{fp(1/width)};","mul.f32 %f164,%f164,%f162;mul.f32 %f164,%f164,%f162;"]
            for j in range(chunks):
                lines.append(f"mul.f32 %f{96+j},%f{j},%f164;")
                if layer:lines.append(f"sub.f32 %f{32+j},%f{32+j},%f163;")
                lines += [f"sub.f32 %f{96+j},%f{32+j},%f{96+j};",f"mul.f32 %f{128+j},%f{96+j},%f162;"]
        else:
            for j in range(chunks):lines.append(f"mul.f32 %f{128+j},%f{j},%f162;")
    if op!="row_sum":
        for j in range(chunks):lines.append(f"{'@%p1 ' if tail and j==chunks-1 else ''}st.global.f32 [%d12+{128*j}],%f{128+j};")
    return name,"\n".join(lines+["done:ret;}"])+"\n"


def generate_module():
    sources=[];kernels=[]
    for op in POINTWISE:
        for mode in ("scalar","scalar4","vector4"):
            name,source=pointwise_source(op,mode);sources.append(source)
            kernels.append({"kernel":name,"op":op,"mode":mode,"kind":"pointwise","elements_per_thread":1 if mode=="scalar" else 4,"inputs":POINTWISE[op][0],"outputs":POINTWISE[op][1],"dtype":POINTWISE[op][2],"rtol":POINTWISE[op][3][0],"atol":POINTWISE[op][3][1]})
    for width in (128,256,513):
        for op in ROWWISE:
            if width==256 and op not in ("rmsnorm","rmsnorm_backward","softmax","softmax_backward"):continue
            for warps in (4,8):
                name,source=rowwise_source(op,width,warps);sources.append(source)
                kernels.append({"kernel":name,"op":op,"mode":f"warp{warps}","kind":"rowwise","width":width,"warps":warps,"inputs":ROWWISE[op][0],"outputs":ROWWISE[op][1],"dtype":"float","rtol":ROWWISE[op][2][0],"atol":ROWWISE[op][2][1]})
    return ".version 8.7\n.target sm_80\n.address_size 64\n"+"\n".join(sources),kernels


def build_payload(force=False):
    from openassembler.build import Options,compile_paths
    BUILD.mkdir(parents=True,exist_ok=True)
    source,kernels=generate_module()
    path=BUILD/"operators.ptx"
    old=path.read_text() if path.exists() else None
    path.write_text(source)
    out=BUILD/"compiled"
    if force or old!=source or not (out/"manifest.json").is_file():
        result=compile_paths(path,out,Options(),o_name="operators")
        if result.rc:raise RuntimeError(f"operator build failed ({result.rc}); see {out}")
    manifest=json.loads((out/"manifest.json").read_text())
    entry=manifest["entries"][0]
    if len(entry["kernels"])!=len(kernels) or any(k["compiled_by"]!="openassembler" for k in entry["kernels"]):raise RuntimeError("all operators must compile natively; no fallback")
    payload={"schema":1,"kernels":kernels,"source_sha256":hashlib.sha256(source.encode()).hexdigest(),"manifest":manifest,"source":source}
    for method,key in (("oa","cubin"),("ptxas","reference")):
        data=(out/entry[key]).read_bytes();payload[method]=base64.b64encode(data).decode();payload[method+"_sha256"]=hashlib.sha256(data).hexdigest()
    payload["source_identities"]={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ("exp/nn_round2/operator_suite.py","exp/nn_round2/operator_worker.py","openassembler/backend/ra.py","openassembler/backend/compile.py","openassembler/backend/sched.py","openassembler/backend/fragments.py")}
    sass=(out/"sass/operators.sass").read_text();payload["sass_sha256"]=hashlib.sha256(sass.encode()).hexdigest()
    payload["static"]={"LDG128":sass.count("LDG.E.128"),"STG128":sass.count("STG.E.128"),"WARPSYNC":sass.count("WARPSYNC"),"SHFL":sass.count("SHFL")}
    (BUILD/"payload.json").write_text(json.dumps(payload))
    return payload


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--build",action="store_true");parser.add_argument("--force",action="store_true");args=parser.parse_args()
    payload=build_payload(force=args.force)
    print(json.dumps({"kernels":len(payload["kernels"]),"source_sha256":payload["source_sha256"],"payload":str(BUILD/"payload.json")},indent=2))
