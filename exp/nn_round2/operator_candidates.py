"""Round-two candidates: explicit FMA local reductions, isolated from production.

The candidate contracts multiply/add into one FMA in each per-lane dot-product
sum. Warp shuffles, synchronization, addresses, masks and final formulas remain
identical. Its floating-point rounding is explicitly FMA; correctness is checked
against the original, unchanged FP64-reference tolerances. Both compiler arms
consume exactly the same candidate PTX.

Build: .venv/bin/python exp/nn_round2/operator_candidates.py --build
Callable worker: run(payload, samples=17, graph_repeats=32, rounds=2).
"""
from __future__ import annotations
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parents)>2 else Path.cwd()
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
HERE=ROOT/"exp/nn_round2"
BUILD=HERE/"operator_fma_build"
OPS=("softmax_backward","layernorm","layernorm_backward","rmsnorm","rmsnorm_backward")


def fma_local(source,op,width):
    chunks=(width+31)//32
    groups=[(64,0,32,160)] if op=="softmax_backward" else [(64,0,0,161)]
    if op.endswith("backward") and op!="softmax_backward":groups.append((96,0,32,164))
    replacements=[]
    for product_base,left,right,dest in groups:
        products=[f"mul.f32 %f{product_base+j},%f{left+j},%f{right+j};" for j in range(chunks)]
        products_text="\n".join(products)
        if source.count(products_text)!=1:raise ValueError(f"product block not unique: {op}/{product_base}")
        accum=[f"mov.f32 %f{dest},%f{product_base};"]+[f"add.f32 %f{dest},%f{dest},%f{product_base+j};" for j in range(1,chunks)]
        accum_text="\n".join(accum)
        if source.count(accum_text)!=1:raise ValueError(f"accumulator block not unique: {op}/{product_base}")
        # At the original product location, form the local sum directly. Delete
        # only its old local reduction, retaining the following five shuffles.
        fused=[f"mul.f32 %f{dest},%f{left},%f{right};"]+[f"fma.rn.f32 %f{dest},%f{left+j},%f{right+j},%f{dest};" for j in range(1,chunks)]
        source=source.replace(products_text,"\n".join(fused),1).replace(accum_text,"",1)
        replacements.append({"product_register_base":product_base,"accumulator":dest,"local_terms":chunks,"explicit_fmas":chunks-1})
    return source,replacements


def generate_module():
    from exp.nn_round2.operator_suite import ROWWISE,rowwise_source
    sources=[];kernels=[]
    for width in (128,256,513):
        for op in OPS:
            for warps in (4,8):
                original_name,original=rowwise_source(op,width,warps)
                candidate,replacements=fma_local(original,op,width)
                for mode,body in (("baseline",original),("fma",candidate)):
                    name=original_name+"_"+mode;body=body.replace(".entry "+original_name+"(",".entry "+name+"(",1)
                    sources.append(body)
                    kernels.append({"kernel":name,"op":op,"mode":f"{mode}_warp{warps}","candidate":mode,"kind":"rowwise","width":width,"warps":warps,"inputs":ROWWISE[op][0],"outputs":ROWWISE[op][1],"dtype":"float","rtol":ROWWISE[op][2][0],"atol":ROWWISE[op][2][1],"changed_local_reductions":replacements if mode=="fma" else [],"rounding_policy":"explicit fused FMA in local dot reductions" if mode=="fma" else "unconstrained PTX mul/add; ptxas may contract"})
    return ".version 8.7\n.target sm_80\n.address_size 64\n"+"\n".join(sources),kernels


def build_payload(force=False):
    from openassembler.build import Options,compile_paths
    BUILD.mkdir(exist_ok=True,parents=True)
    text,kernels=generate_module();source=BUILD/"fma_operators.ptx";old=source.read_text() if source.exists() else None;source.write_text(text)
    out=BUILD/"compiled"
    if force or old!=text or not (out/"manifest.json").is_file():
        result=compile_paths(source,out,Options(),o_name="fma_operators")
        if result.rc:raise RuntimeError("FMA candidate build failed")
    manifest=json.loads((out/"manifest.json").read_text());entry=manifest["entries"][0]
    if len(entry["kernels"])!=len(kernels) or any(k["compiled_by"]!="openassembler" for k in entry["kernels"]):raise RuntimeError("native-only candidate build required")
    payload={"schema":1,"kernels":kernels,"source":text,"source_sha256":hashlib.sha256(text.encode()).hexdigest(),"manifest":manifest}
    for method,key in (("oa","cubin"),("ptxas","reference")):
        data=(out/entry[key]).read_bytes();payload[method]=base64.b64encode(data).decode();payload[method+"_sha256"]=hashlib.sha256(data).hexdigest()
    payload["source_identities"]={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ("exp/nn_round2/operator_suite.py","exp/nn_round2/operator_worker.py","exp/nn_round2/operator_candidates.py","openassembler/backend/ra.py","openassembler/backend/compile.py","openassembler/backend/sched.py")}
    (BUILD/"payload.json").write_text(json.dumps(payload))
    return payload


def run(payload,samples=17,graph_repeats=32,rounds=2,rows=(257,4096)):
    """Paired candidate/baseline graphs on identical inputs, both order rounds.

    Reuse the validated worker's checks without monkeypatching it: submit one
    kernel per call. Its kernel_index is therefore zero, and a seed derived from
    op/width/rows is identical across baseline/FMA and CTA geometries. Module
    load and Python work are excluded from the steady graph event interval.
    """
    from exp.nn_round2 import operator_worker
    from collections import defaultdict
    import random
    import statistics
    start=time.monotonic();results=[];runtime=None;methodology=None
    for ri in range(rounds):
        order=list(payload["kernels"])
        random.Random(20261107).shuffle(order)
        if ri%2:order.reverse()
        for k in order:
            for count in rows:
                stable_seed=int.from_bytes(hashlib.sha256(f"{k['op']}:{k['width']}:{count}:20261107".encode()).digest()[:4],"little")
                result=operator_worker.run({**payload,"kernels":[k]},samples=samples,graph_repeats=graph_repeats,rows=count,seed=stable_seed)
                runtime=result["runtime"];methodology=result["methodology"]
                for case in result["cases"]:case.update(round=ri,input_seed=stable_seed,candidate=k["candidate"],rounding_policy=k["rounding_policy"])
                results.extend(result["cases"])
    return {"schema":1,"cases":results,"runtime":runtime,"source_sha256":payload["source_sha256"],"source_identities":payload["source_identities"],"oa_sha256":payload["oa_sha256"],"ptxas_sha256":payload["ptxas_sha256"],"methodology":{**(methodology or {}),"rounds":rounds,"input_seeding":"sha256(op:width:rows:20261107), fixed across layouts/candidates/rounds","case_order":"seeded permutation then reverse; no selection in worker","candidate_change":"local product+sum contracted into explicit FMA; all warp synchronization and masks retained","candidate_numerical_contract":"original unchanged per-op FP64 tolerances; no approximation-policy relaxation"},"elapsed_seconds":time.monotonic()-start}


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--build",action="store_true");parser.add_argument("--force",action="store_true");args=parser.parse_args()
    payload=build_payload(force=args.force)
    print(json.dumps({"kernels":len(payload["kernels"]),"source_sha256":payload["source_sha256"],"payload":str(BUILD/"payload.json")},indent=2))
