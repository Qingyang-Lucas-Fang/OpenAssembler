"""GPU worker for matched native OA/PTXAS NN-arithmetic experiments.

Only invoked in an A100 worker; imports torch lazily. No production monkeypatches.
"""
from __future__ import annotations
import base64
import ctypes as C
import hashlib
import random
import statistics
import time
import traceback


def reference(op, inputs):
    import torch
    x,z,w,v=(a.double() for a in inputs)
    if op=="residual_add":return [(x+z).float().double()]
    if op=="gated_multiply":return [(x*z).float().double()]
    if op=="relu":return [x.clamp_min(0)]
    if op=="relu_backward":return [torch.where(x>0,z,0)]
    if op in ("gelu_tanh","gelu_tanh_backward"):
        t=torch.tanh(.7978845608028654*(x+.044715*x*x*x))
        if op=="gelu_tanh":return [.5*x*(1+t)]
        return [z*(.5*(1+t)+.5*x*(1-t*t)*.7978845608028654*(1+3*.044715*x*x))]
    if op=="silu":return [x*torch.sigmoid(x)]
    if op=="sigmoid":return [torch.sigmoid(x)]
    if op=="silu_backward":
        s=torch.sigmoid(x);return [z*s*(1+x*(1-s))]
    if op=="tanh_backward":return [z*(1-x*x)]
    if op=="divide":return [x/z]
    if op=="rsqrt_scale":return [x*torch.rsqrt(z)]
    if op=="dropout_mask_scale":return [(inputs[0]*inputs[1]*1.25).double()]
    if op=="rope_pairs":return [x*w-z*v,x*v+z*w]
    if op=="sgd_momentum":
        m=.9*w+z;return [x-.001*m,m]
    if op=="adam_like":
        m=.9*w+.1*z;vv=.999*v+.001*z*z
        return [x-.001*m/(torch.sqrt(vv)+1e-8),m,vv]
    if op in ("div_const7","rem_const7"):
        bits=inputs[0].to(torch.int64)&0xffffffff
        return [bits//7 if op.startswith("div") else bits%7]
    if op=="softmax":return [torch.softmax(x,-1)]
    if op=="softmax_backward":return [x*(z-(x*z).sum(-1,keepdim=True))]
    if op.startswith("layernorm"):
        centered=x-x.mean(-1,keepdim=True);r=torch.rsqrt(centered.square().mean(-1,keepdim=True)+1e-5)
        if op=="layernorm":return [centered*r]
        return [r*(z-z.mean(-1,keepdim=True)-centered*r*r*(centered*z).mean(-1,keepdim=True))]
    if op.startswith("rmsnorm"):
        r=torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-5)
        if op=="rmsnorm":return [x*r]
        return [r*(z-x*r*r*(x*z).mean(-1,keepdim=True))]
    if op=="cross_entropy":
        p=torch.softmax(x,-1);p[:,0]-=1
        return [p,torch.logsumexp(x,-1)-x[:,0]]
    if op=="row_sum":return [x.sum(-1)]
    raise ValueError(op)


def make_inputs(op, shape, pattern, seed):
    import torch
    generator=torch.Generator(device="cuda");generator.manual_seed(seed)
    inputs=[torch.randn(shape,generator=generator,device="cuda") for _ in range(4)]
    if op in ("div_const7","rem_const7"):
        inputs=[torch.randint(0,2**32,shape,generator=generator,device="cuda",dtype=torch.int64).to(torch.int32) for _ in range(4)]
        if pattern=="zero":inputs[0].zero_()
        elif pattern=="structured":
            values=torch.tensor([0,1,6,7,8,2**31-1,2**31,2**32-1],device="cuda",dtype=torch.int64).to(torch.int32)
            inputs[0].copy_(values[torch.arange(inputs[0].numel(),device="cuda")%len(values)].view(shape))
        return inputs
    if pattern=="zero":
        for t in inputs:t.zero_()
    elif pattern=="structured":
        pattern_tensor=torch.where(torch.arange(inputs[0].numel(),device="cuda")%2==0,3.,-3.).view(shape)
        inputs[0].copy_(pattern_tensor)
        inputs[1].copy_(pattern_tensor*.25)
        if op.startswith("layernorm"):
            inputs[0].copy_(1000.+pattern_tensor*.01)
    if op in ("divide","rsqrt_scale"):inputs[1].abs_().add_(.125)
    if op=="adam_like":inputs[3].abs_().mul_(.01).add_(.05)
    if op=="tanh_backward":inputs[0].tanh_()
    if op=="dropout_mask_scale":inputs[1].copy_((inputs[1]>0).float())
    if op=="rope_pairs":
        angle=inputs[2].clone();inputs[2].copy_(torch.cos(angle));inputs[3].copy_(torch.sin(angle))
    if op=="softmax_backward":inputs[0].copy_(torch.softmax(inputs[0],-1))
    return inputs


def run(payload, samples=15, graph_repeats=32, sizes=(65536,1048576), rows=257,
        include_kernels=None, launch_threads=(128,256), seed=20261107):
    import torch
    from openassembler.gpu.cuda_driver import Driver
    torch.set_num_threads(2);torch.cuda.init();anchor=torch.empty(1,device="cuda")
    driver=Driver();start=time.monotonic();results=[]
    if driver.info().cc!="8.0":raise RuntimeError("native campaign requires SM80/A100")
    modules={method:driver.load(base64.b64decode(payload[method])) for method in ("oa","ptxas")}
    rng=random.Random(seed)
    def numerical(actual,expected,k):
        diff=actual.double()-expected.double()
        close=torch.isclose(actual.double(),expected.double(),rtol=k["rtol"],atol=k["atol"])
        return {"passed":bool(torch.isfinite(actual).all() and close.all()),"elements_checked":actual.numel(),"mismatched_elements":int((~close).sum()),"max_abs":float(diff.abs().max()),"relative_l2":float(torch.linalg.vector_norm(diff)/torch.linalg.vector_norm(expected.double()).clamp_min(1e-30))}
    def summary(values):return {"samples_ms":values,"median_ms":statistics.median(values),"min_ms":min(values),"max_ms":max(values)}
    try:
        for kernel_index,k in enumerate(payload["kernels"]):
            if include_kernels is not None and k["kernel"] not in include_kernels:continue
            configurations=[(size,threads) for size in sizes for threads in launch_threads] if k["kind"]=="pointwise" else [(rows,k["warps"]*32)]
            if k["kind"]=="pointwise" and k["mode"]=="scalar":configurations.append((65539,128))
            for count,threads in configurations:
                if k["kind"]=="pointwise" and k["elements_per_thread"]==4 and count%4:raise ValueError("four-element layouts require N divisible by four")
                shape=(count,) if k["kind"]=="pointwise" else (count,k["width"])
                row={"kernel":k["kernel"],"op":k["op"],"mode":k["mode"],"shape":list(shape),"threads":threads,"rtol":k["rtol"],"atol":k["atol"],"status":"not_run"}
                try:
                    functions={m:driver.function(mod,k["kernel"]) for m,mod in modules.items()}
                    row["resources"]={m:driver.func_attrs(fn) for m,fn in functions.items()}
                    dtype=torch.int32 if k["dtype"]=="integer" else torch.float32
                    outshapes=[shape for _ in range(k["outputs"])]
                    if k["op"]=="row_sum":outshapes=[(count,)]
                    if k["op"]=="cross_entropy":outshapes=[shape,(count,)]
                    storages={};outputs={}
                    for method in modules:
                        storages[method]=[];outputs[method]=[]
                        for outshape in outshapes:
                            n=1
                            for s in outshape:n*=s
                            storage=torch.full((n+512,),-12345,device="cuda",dtype=dtype)
                            output=storage[256:-256].view(outshape)
                            storages[method].append(storage);outputs[method].append(output)
                    def guards(method):return all(bool((s[:256]==-12345).all() and (s[-256:]==-12345).all()) for s in storages[method])
                    def launcher(method,inputs):
                        tensors=list(inputs)+outputs[method]+[outputs[method][0]]*(3-len(outputs[method]))
                        args=[C.c_uint64(t.data_ptr()) for t in tensors]+[C.c_uint32(count)]
                        params=(C.c_void_p*len(args))(*[C.cast(C.pointer(a),C.c_void_p) for a in args])
                        grid=((count+k["elements_per_thread"]*threads-1)//(k["elements_per_thread"]*threads) if k["kind"]=="pointwise" else (count+k["warps"]-1)//k["warps"])
                        fn=functions[method]
                        def launch():driver.ck(driver.cu.cuLaunchKernel(fn,grid,1,1,threads,1,1,0,C.c_void_p(torch.cuda.current_stream().cuda_stream),params,None),k["kernel"])
                        # Closure retains args and inputs for pointer lifetimes.
                        launch._refs=(args,inputs,tensors)
                        return launch
                    row["patterns"]=[];normal_inputs=None
                    for pi,pattern in enumerate(("normal","zero","structured")):
                        inputs=make_inputs(k["op"],shape,pattern,seed+kernel_index)
                        if pattern=="normal":normal_inputs=inputs
                        expected=reference(k["op"],inputs);checks={}
                        input_snapshots=[t.clone() for t in inputs]
                        for method in modules:
                            for out in outputs[method]:out.fill_(0x2468ace0 if dtype==torch.int32 else float("nan"))
                            launch=launcher(method,inputs);launch();torch.cuda.synchronize()
                            checks[method]={"outputs":[numerical(out,ref,k) for out,ref in zip(outputs[method],expected)],"canaries":guards(method)}
                            checks[method]["inputs_unchanged"]=all(torch.equal(a.view(torch.int32),b.view(torch.int32)) for a,b in zip(inputs,input_snapshots))
                            checks[method]["passed"]=checks[method]["canaries"] and checks[method]["inputs_unchanged"] and all(a["passed"] for a in checks[method]["outputs"])
                        row["patterns"].append({"pattern":pattern,"checks":checks,"oa_ptxas_bitwise_equal":all(torch.equal(a.view(torch.int32),b.view(torch.int32)) for a,b in zip(outputs["oa"],outputs["ptxas"]))})
                    eligible=[m for m in modules if all(p["checks"][m]["passed"] for p in row["patterns"])]
                    row["eligible_for_timing"]=eligible
                    if len(eligible)!=2:
                        row["status"]="correctness_failure";results.append(row);print({"operator":k["kernel"],"shape":shape,"status":row["status"]},flush=True);continue
                    inputs=normal_inputs;expected=reference(k["op"],inputs)
                    launchers={m:launcher(m,inputs) for m in eligible}
                    for m in eligible:
                        for _ in range(5):launchers[m]()
                    torch.cuda.synchronize()
                    graphs={};row["graph_checks"]={}
                    for m in eligible:
                        graph=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            for _ in range(graph_repeats):launchers[m]()
                        for out in outputs[m]:out.fill_(0x2468ace0 if dtype==torch.int32 else float("nan"))
                        graph.replay();torch.cuda.synchronize()
                        chk={"outputs":[numerical(out,ref,k) for out,ref in zip(outputs[m],expected)],"canaries":guards(m)}
                        chk["passed"]=chk["canaries"] and all(a["passed"] for a in chk["outputs"])
                        row["graph_checks"][m]=chk
                        if chk["passed"]:graphs[m]=graph
                    if len(graphs)!=2:row["status"]="graph_correctness_failure";results.append(row);continue
                    values={m:[] for m in eligible};orders=[]
                    for _ in range(samples):
                        order=list(eligible);rng.shuffle(order);orders.append(order)
                        for m in order:
                            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                            begin.record();graphs[m].replay();end.record();end.synchronize();values[m].append(begin.elapsed_time(end)/graph_repeats)
                    row["graphs"]={m:summary(v) for m,v in values.items()};row["graph_orders"]=orders
                    row["ptxas_over_oa"]=row["graphs"]["ptxas"]["median_ms"]/row["graphs"]["oa"]["median_ms"]
                    row["status"]="ok";print({"operator":k["kernel"],"shape":shape,"threads":threads,"ptxas_over_oa":row["ptxas_over_oa"]},flush=True)
                except Exception as exc:row.update(status="error",error=f"{type(exc).__name__}: {exc}",traceback=traceback.format_exc());print({"operator":k["kernel"],"error":row["error"]},flush=True)
                results.append(row)
    finally:
        for mod in modules.values():driver.unload(mod)
    return {"schema":1,"cases":results,"source_sha256":payload["source_sha256"],"oa_sha256":payload["oa_sha256"],"ptxas_sha256":payload["ptxas_sha256"],"source_identities":payload["source_identities"],"runtime":{"torch":torch.__version__,"cuda":torch.version.cuda,**driver.info().__dict__},"methodology":{"reference":"FP64 formulas on exact FP32 inputs; explicit tanh-GELU; RMS/LayerNorm epsilon=1e-5; fixed label zero cross entropy; no affine norms; Adam-like has no bias correction/weight decay","precision":"FP32 inputs/accumulation/outputs or uint32 bit patterns; no TF32","patterns":["normal","zero","structured"],"graph_repeats":graph_repeats,"paired_shuffled_samples":samples,"graph_poison_after_capture":True,"all_outputs_and_guards":True,"inputs_checked_unchanged":True,"exact_pointwise_reference":"single residual add/multiply rounded once to FP32; mask-scale follows two explicit FP32 operations","vector_contract":"scalar4/vector4 require N multiple4; vector4 pointers16bytealigned; scalar odd-tail checks separately","compiler_control":"same PTX, ptxas -O3; no vendor-library attribution"},"elapsed_seconds":time.monotonic()-start}
