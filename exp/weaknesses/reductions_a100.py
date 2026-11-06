"""A100 full-output/canary and paired-event/graph reduction experiment.

Build with `.venv/bin/python exp/weaknesses/reductions_a100.py --build-only`;
execute with `modal run exp/weaknesses/reductions_a100.py`.
"""
from __future__ import annotations
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import os

_FILE = Path(__file__).resolve()
ROOT = _FILE.parents[2] if len(_FILE.parents) > 2 else Path.cwd()
HERE = ROOT / "exp/weaknesses"
VARIANT = os.environ.get("OA_REDUCTION_VARIANT", "bind")
BUILD = HERE / ("reductions_build" if VARIANT == "bind" else f"reductions_build_{VARIANT}")
SHAPES = [(128,128),(1024,512),(1024,1024),(129,127),(133,513),(65,1023),
          (5,1),(7,31),(9,33)]
TOL = {"softmax": {"rtol": 5e-5, "atol": 1e-6},
       "layernorm": {"rtol": 1e-4, "atol": 2e-5}}


def build():
    if VARIANT == "production":
        from openassembler.kernels.reductions import ReductionPlan,generate
        generator_path=ROOT/"openassembler/kernels/reductions.py"
    else:
        from exp.weaknesses.reductions_codegen import ReductionPlan, generate, enable_experimental_shuffle
        enable_experimental_shuffle()
        generator_path=HERE/"reductions_codegen.py"
    from openassembler.build import Options, compile_paths
    BUILD.mkdir(parents=True, exist_ok=True)
    records = []
    for op in TOL:
        for rows,width in (SHAPES if VARIANT in ("bind","production") or os.environ.get("OA_REDUCTION_FULL") else SHAPES[:3]):
            name = f"{op}_{rows}x{width}"
            source = generate(ReductionPlan(rows,width,op))
            src = BUILD / f"{name}.ptx"
            src.write_text(source)
            out = BUILD / name
            if compile_paths(src,out,Options(),o_name=name).rc:
                raise RuntimeError(f"native compile failed: {name}")
            m = json.loads((out / "manifest.json").read_text())
            e = m["entries"][0]
            if any(k["compiled_by"] != "openassembler" for k in e["kernels"]):
                raise RuntimeError("fallback forbidden")
            rec = {"name": name,"op": op,"rows": rows,"width": width,
                   "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                   "source": source,"launch": ReductionPlan(rows,width,op).launch,
                   "kernels": e["kernels"]}
            rec["sass_sha256"] = hashlib.sha256((out/f"sass/{name}.sass").read_bytes()).hexdigest()
            for method,key in (("oa","cubin"),("ptxas","reference")):
                data = (out/e[key]).read_bytes()
                rec[method] = base64.b64encode(data).decode()
                rec[method+"_sha256"] = hashlib.sha256(data).hexdigest()
            records.append(rec)
    old = ROOT / "exp/whole_stack/native_ops_build"
    entry = json.loads((old/"manifest.json").read_text())["entries"][0]
    legacy = {method: base64.b64encode((old/entry[key]).read_bytes()).decode()
              for method,key in (("legacy_oa","cubin"),("legacy_ptxas","reference"))}
    payload = {"cases": records,"legacy": legacy,"variant": VARIANT,
               "codegen_sha256": hashlib.sha256(generator_path.read_bytes()).hexdigest(),
               "generator_path":str(generator_path.relative_to(ROOT))}
    payload["legacy_provenance"] = {
        method+"_sha256": hashlib.sha256(base64.b64decode(blob)).hexdigest()
        for method,blob in legacy.items()}
    payload["legacy_provenance"]["source_sha256"] = hashlib.sha256((ROOT/"examples/nn_ops.py").read_bytes()).hexdigest()
    payload["backend_sha256"] = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                               for name in ("openassembler/backend/ra.py","openassembler/backend/compile.py",
                                            "openassembler/backend/sched.py","specs/sm_80/synth_templates.json",
                                            "specs/sm_80/synth_fieldmaps.json")}
    history = HERE/"reductions_history";history.mkdir(exist_ok=True)
    (history/(payload["codegen_sha256"]+".py")).write_bytes(generator_path.read_bytes())
    (BUILD/"payload.json").write_text(json.dumps(payload))
    return payload


if __name__ == "__main__" and "--build-only" in sys.argv:
    build()
    raise SystemExit(0)

import modal
app = modal.App("openassembler-warp-reductions")
image = (modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
         .add_local_python_source("openassembler", copy=False))


@app.function(image=image, gpu="A100-40GB", timeout=600, max_containers=1, retries=0)
def execute(payload):
    import ctypes as C
    import random
    import statistics
    import time
    import traceback
    import torch
    import torch.nn.functional as F
    from openassembler.gpu.cuda_driver import Driver
    torch.manual_seed(20261106)
    torch.set_num_threads(2)
    torch.cuda.init()
    anchor = torch.empty(1,device="cuda")
    driver = Driver()
    old_modules = {k: driver.load(base64.b64decode(v)) for k,v in payload["legacy"].items()}
    results = []
    started = time.monotonic()

    def check(y,ref,tol):
        close = torch.isclose(y.double(),ref,**tol)
        return {"passed": bool(torch.isfinite(y).all() and close.all()),
                "elements_checked": y.numel(),"mismatched_elements": int((~close).sum()),
                "max_abs": float((y.double()-ref).abs().max()),
                "relative_l2": float(torch.linalg.vector_norm(y.double()-ref)/
                                     torch.linalg.vector_norm(ref).clamp_min(1e-30))}

    def summarize(samples):
        return {"samples_ms": samples,"median_ms": statistics.median(samples),
                "min_ms": min(samples),"max_ms": max(samples)}

    for ci,case in enumerate(payload["cases"]):
        row = {key:value for key,value in case.items()
               if key not in ("oa","ptxas","source")}
        modules = {}
        try:
            op,rows,width = case["op"],case["rows"],case["width"]
            shape=(rows,width)
            xstorage=torch.full((rows*width+512,),float("nan"),device="cuda")
            x=xstorage[256:-256].view(shape)
            storages={k:torch.full_like(xstorage,-12345.) for k in
                      ("oa","ptxas","legacy_oa","legacy_ptxas","torch_cuda")}
            outputs={k:v[256:-256].view(shape) for k,v in storages.items()}
            modules={k:driver.load(base64.b64decode(case[k])) for k in ("oa","ptxas")}
            functions={k:driver.function(m,op) for k,m in {**modules,**old_modules}.items()}
            row["resources"]={k:driver.func_attrs(f) for k,f in functions.items()}
            launchers={}
            for method,fn in functions.items():
                args=[C.c_uint64(x.data_ptr()),C.c_uint64(outputs[method].data_ptr())]
                if method.startswith("legacy"):
                    args.extend([C.c_int(rows),C.c_int(width)])
                    grid=(rows+127)//128
                else:
                    grid=case["launch"]["grid"][0]
                params=(C.c_void_p*len(args))(*[C.cast(C.pointer(a),C.c_void_p) for a in args])
                def launcher(fn=fn,params=params,args=args,grid=grid):
                    driver.ck(driver.cu.cuLaunchKernel(fn,grid,1,1,128,1,1,0,
                              C.c_void_p(torch.cuda.current_stream().cuda_stream),params,None),op)
                launchers[method]=launcher
            def torch_launch():
                if op=="softmax":
                    torch.softmax(x,dim=-1,out=outputs["torch_cuda"])
                else:
                    outputs["torch_cuda"]=F.layer_norm(x,(width,),eps=1e-5)
            launchers["torch_cuda"]=torch_launch
            normal=torch.randn(shape,device="cuda")
            patterns={"normal":normal,"zero":torch.zeros_like(normal),
                      "large_offset_small_variance":1000.+normal*.01,
                      "wide":normal*30.,
                      "alternating":torch.where(torch.arange(width,device="cuda")%2==0,
                                                 10000.,-10000.).expand(shape).clone()}
            row["patterns"]=[]
            for pattern,values in patterns.items():
                x.copy_(values)
                xd=x.double()
                if op=="softmax":
                    expected=torch.softmax(xd,dim=-1)
                else:
                    mean=xd.mean(dim=-1,keepdim=True)
                    expected=(xd-mean)*torch.rsqrt((xd-mean).square().mean(dim=-1,keepdim=True)+1e-5)
                checks={}
                for method,launch in launchers.items():
                    outputs[method].fill_(float("nan"))
                    launch()
                    torch.cuda.synchronize()
                    checks[method]=check(outputs[method],expected,TOL[op])
                    checks[method]["canaries"]=(bool((storages[method][:256]==-12345.).all() and
                                                         (storages[method][-256:]==-12345.).all())
                                                  if op=="softmax" or method!="torch_cuda" else None)
                row["patterns"].append({"name":pattern,"checks":checks,
                    "oa_ptxas_bitwise_equal": bool(torch.equal(outputs["oa"].view(torch.int32),
                                                               outputs["ptxas"].view(torch.int32)))})
            # All candidate outputs must pass every unchanged numerical gate.
            candidate_pass=all(p["checks"]["oa"]["passed"] and p["checks"]["oa"]["canaries"]
                               for p in row["patterns"])
            x.copy_(normal)
            eligible=[k for k in launchers if row["patterns"][0]["checks"][k]["passed"]
                      and row["patterns"][0]["checks"][k]["canaries"] is not False]
            row["candidate_passed_all_patterns"]=candidate_pass
            row["timing_inputs"]="normal pattern, checked in full before timing"
            if not candidate_pass:
                row["status"]="candidate_correctness_failure"
                results.append(row)
                print(json.dumps({"case":case["name"],"status":row["status"],"checks":row["patterns"]}),flush=True)
                continue
            for method in eligible:
                for _ in range(5):launchers[method]()
            torch.cuda.synchronize()
            rng=random.Random(20261106+ci)
            samples={k:[] for k in eligible}
            orders=[]
            for _ in range(20):
                order=list(eligible);rng.shuffle(order);orders.append(order)
                for method in order:
                    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    begin.record();launchers[method]();end.record();end.synchronize()
                    samples[method].append(begin.elapsed_time(end))
            row["events"]={k:summarize(v) for k,v in samples.items()}
            row["event_order"]=orders
            graphs={}; graph_checks={}
            for method in eligible:
                g=torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    for _ in range(32):launchers[method]()
                # Prove capture contains the operation: poison only after capture.
                outputs[method].fill_(float("nan"));g.replay();torch.cuda.synchronize()
                xd=x.double()
                if op=="softmax":ref=torch.softmax(xd,dim=-1)
                else:
                    mu=xd.mean(dim=-1,keepdim=True)
                    ref=(xd-mu)*torch.rsqrt((xd-mu).square().mean(dim=-1,keepdim=True)+1e-5)
                graph_checks[method]=check(outputs[method],ref,TOL[op])
                graph_checks[method]["canaries"]=(bool((storages[method][:256]==-12345.).all() and
                                                         (storages[method][-256:]==-12345.).all())
                                                  if op=="softmax" or method!="torch_cuda" else None)
                if graph_checks[method]["passed"] and graph_checks[method]["canaries"] is not False:
                    graphs[method]=g
            row["graph_poison_replay_checks"]=graph_checks
            if "oa" not in graphs:
                raise RuntimeError("OA graph failed poisoned replay output/canary validation")
            samples={k:[] for k in graphs};orders=[]
            for _ in range(20):
                order=list(graphs);rng.shuffle(order);orders.append(order)
                for method in order:
                    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    begin.record();graphs[method].replay();end.record();end.synchronize()
                    samples[method].append(begin.elapsed_time(end)/32)
            row["graphs"]={k:summarize(v) for k,v in samples.items()}
            row["graph_order"]=orders
            row["status"]="ok"
            print(json.dumps({"case":case["name"],"status":row["status"],
                              "graph_ms":{k:v["median_ms"] for k,v in row["graphs"].items()}}),flush=True)
            results.append(row)
        except Exception as exc:
            row.update(status="error",error=f"{type(exc).__name__}: {exc}",traceback=traceback.format_exc())
            results.append(row)
            print(json.dumps({"case":case["name"],"status":"error","error":row["error"]}),flush=True)
        finally:
            for module in modules.values():driver.unload(module)
    for module in old_modules.values():driver.unload(module)
    return {"schema":1,"cases":results,"tolerances_declared_before_run":TOL,
            "reference":"FP64 formulas on exact FP32 inputs; population variance; epsilon=1e-5",
            "timing":"5 warmup;20 shuffled paired single-launch events;20 shuffled paired graphs,32 operations/graph; poison-after-capture verified",
            "runtime":{"torch":torch.__version__,"cuda":torch.version.cuda,**driver.info().__dict__},
            "elapsed_gpu_worker_seconds":time.monotonic()-started}


@app.local_entrypoint()
def main():
    payload=json.loads((BUILD/"payload.json").read_text())
    result=execute.remote(payload)
    result["codegen_sha256"]=payload["codegen_sha256"]
    result["variant"]=payload["variant"]
    result["legacy_provenance"]=payload["legacy_provenance"]
    result["backend_sha256"]=payload["backend_sha256"]
    result["generator_path"]=payload["generator_path"]
    result["timestamp_utc"]=datetime.now(timezone.utc).isoformat()
    result["harness_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    directory=HERE/"reductions_runs";directory.mkdir(exist_ok=True)
    path=directory/f"a100-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}.json"
    path.write_text(json.dumps(result,indent=2)+"\n")
    print(f"evidence: {path}")
