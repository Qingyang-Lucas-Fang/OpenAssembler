"""Native-routed full forward/backward/AdamW experiment; GPU-only worker.

These are explicit custom mixed-precision networks. They do not claim arbitrary
model compilation or equality to unmodified autocast/FlashAttention semantics.
"""
from __future__ import annotations


def run(payload, samples=9, verification_steps=3, timed_steps=5, cache_experiment=False, training_graph_experiment=False):
    import base64
    import copy
    import ctypes as C
    from contextlib import contextmanager
    import glob
    import math
    import random
    import statistics
    import time
    import traceback
    import torch
    from torch import nn
    from torch.nn import functional as F
    from openassembler.gpu.cuda_driver import Driver

    torch.set_num_threads(2)
    torch.manual_seed(20261107)
    torch.cuda.init()
    anchor = torch.empty(1, device="cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    driver = Driver()
    if driver.info().cc != "8.0":
        raise RuntimeError("A100/SM80 required")
    modules, kernels, resources = [], {}, {}
    for key, data in payload["gemms"].items():
        for method, blob in data["kernels"].items():
            mod = driver.load(base64.b64decode(blob)); modules.append(mod)
            fn = driver.function(mod, "gemm")
            driver.ck(driver.cu.cuFuncSetAttribute(fn, 8, 65536), "shared memory opt-in")
            kernels[method, key] = fn
            resources[method + "/" + key] = driver.func_attrs(fn)
    op = payload["operator_payload"]
    opmods = {m: driver.load(base64.b64decode(op[m])) for m in ("oa", "ptxas")}
    modules.extend(opmods.values())
    opspec = {k["kernel"]: k for k in op["kernels"]}
    opfns = {}
    paths = glob.glob("/usr/local/lib/python3*/site-packages/nvidia/cublas/lib/libcublas.so*")
    blas = C.CDLL(paths[0] if paths else "libcublas.so.12")
    create = blas.cublasCreate_v2; create.argtypes = [C.POINTER(C.c_void_p)]; create.restype = C.c_int
    destroy = blas.cublasDestroy_v2; destroy.argtypes = [C.c_void_p]; destroy.restype = C.c_int
    setstream = blas.cublasSetStream_v2; setstream.argtypes = [C.c_void_p, C.c_void_p]; setstream.restype = C.c_int
    gemm = blas.cublasGemmEx
    gemm.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int,
                    C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_void_p,
                    C.c_int, C.c_int, C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int]
    gemm.restype = C.c_int
    handle = C.c_void_p()
    if create(C.byref(handle)):
        raise RuntimeError("cuBLAS initialization failed")
    setmath = blas.cublasSetMathMode
    setmath.argtypes = [C.c_void_p, C.c_int]; setmath.restype = C.c_int
    if setmath(handle, 16):
        raise RuntimeError("cuBLAS disallow reduced-precision reduction failed")
    alpha, beta = C.c_float(1), C.c_float(0)

    def summarize(values):
        return {"samples_ms": values, "median_ms": statistics.median(values),
                "min_ms": min(values), "max_ms": max(values)}

    def compare(actual, expected, gates):
        details = []
        if set(actual) != set(expected):
            return {"passed": False, "error": "state key sets differ",
                    "missing": sorted(set(expected)-set(actual)), "extra": sorted(set(actual)-set(expected))}
        for key in sorted(expected):
            a, b = actual[key], expected[key]
            if a.shape != b.shape or a.dtype != b.dtype:
                details.append({"name": key, "elements": a.numel(), "passed": False,
                                "error": "shape/dtype mismatch"})
                continue
            category = key.split("/", 1)[0]
            tol = gates[category]
            delta = a.double() - b.double()
            close = torch.isclose(a.double(), b.double(), **tol)
            finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
            details.append({"name": key, "elements": a.numel(), "passed": finite and bool(close.all()),
                            "bitwise_equal": bool(torch.equal(a.view(torch.int32), b.view(torch.int32))) if a.dtype == torch.float32 else bool(torch.equal(a,b)),
                            "max_abs": float(delta.abs().max()), "mismatched": int((~close).sum()),
                            "relative_l2": float(delta.norm()/b.double().norm().clamp_min(1e-30)), **tol})
        return {"passed": all(d["passed"] for d in details), "tensors_checked": len(details),
                "elements_checked": sum(d["elements"] for d in details), "details": details}

    class Backend:
        def __init__(self, method):
            self.method = method; self.validate = False; self.guards = []; self.checks = []
            self.trace_enabled = False; self.trace = []; self.counts = {}; self.pack_cache = None
            self.collect_raw = False; self.raw_gradients = {}
            self.cache_enabled = False; self.weight_ids = set(); self.weight_cache = {}

        def pack_weight(self, tensor):
            if not self.cache_enabled or id(tensor) not in self.weight_ids:
                return tensor.to(torch.float16).contiguous()
            key = (id(tensor), tensor._version, tuple(tensor.shape), str(tensor.dtype), str(tensor.device))
            if key not in self.weight_cache:
                # A new version invalidates this parameter's old packed value.
                self.weight_cache = {k:v for k,v in self.weight_cache.items() if k[0] != id(tensor)}
                self.weight_cache[key] = tensor.to(torch.float16).contiguous()
            return self.weight_cache[key]

        @contextmanager
        def stage(self, name):
            self.counts[name] = self.counts.get(name, 0) + 1
            if not self.trace_enabled:
                yield; return
            a,b = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record()
            try:
                yield
            finally:
                b.record(); self.trace.append((name,a,b))

        def output(self, shape):
            if not self.validate:
                return torch.empty(shape, device="cuda", dtype=torch.float32)
            size = math.prod(shape)
            store = torch.full((size+512,), -12345., device="cuda")
            out = store[256:-256].view(shape); out.fill_(float("nan")); self.guards.append(store)
            return out

        def matmul(self, a, b, name):
            assert a.is_contiguous() and b.is_contiguous() and a.dtype == b.dtype == torch.float16
            m,k = a.shape; n,kb = b.shape
            assert k == kb
            out = self.output((m,n))
            with self.stage(name):
                if self.method == "cuda_library":
                    if setstream(handle, C.c_void_p(torch.cuda.current_stream().cuda_stream)):
                        raise RuntimeError("cuBLAS stream failed")
                    rc = gemm(handle, 1, 0, n,m,k, C.byref(alpha), C.c_void_p(b.data_ptr()),2,k,
                              C.c_void_p(a.data_ptr()),2,k,C.byref(beta),C.c_void_p(out.data_ptr()),0,n,68,99)
                    if rc:
                        raise RuntimeError(f"cuBLAS GemmEx:{rc}")
                else:
                    meth = {"oa_tensor_only":"oa_direct", "ptxas_tensor_only":"ptxas"}.get(self.method,self.method)
                    fn = kernels[meth,f"{m}_{n}_{k}"]
                    args = [C.c_uint64(t.data_ptr()) for t in (a,b,out)]
                    params = (C.c_void_p*3)(*[C.cast(C.pointer(v),C.c_void_p) for v in args])
                    driver.ck(driver.cu.cuLaunchKernel(fn,m//128,n//128,1,128,1,1,65536,
                              C.c_void_p(torch.cuda.current_stream().cuda_stream),params,None),name)
            if self.validate:
                truth = a.float() @ b.float().T
                rel = float((out-truth).norm()/truth.norm().clamp_min(1e-30))
                self.checks.append({"name":name,"shape":[m,n,k],"elements":m*n,
                                    "passed":bool(torch.isfinite(out).all()) and rel < 5e-5,"relative_l2":rel})
            return out

        def operation(self, name, x, z=None):
            shape = x.shape
            x = x.contiguous(); z = x if z is None else z.contiguous()
            row = name in ("rmsnorm","rmsnorm_backward","softmax","softmax_backward")
            width = shape[-1]
            opname = f"op_{name}_w{width}_c4" if row else f"op_{name}_scalar"
            native = self.method in ("oa_legacy","oa_direct","ptxas") and opname in opspec and name != "gelu_tanh_backward"
            with self.stage(("native/" if native else "shared_cuda/")+name):
                if not native:
                    if name == "relu": return x.relu()
                    if name == "relu_backward": return torch.where(x>0,z,0.)
                    if name == "gelu_tanh": return F.gelu(x, approximate="tanh")
                    if name == "gelu_tanh_backward":
                        t = torch.tanh(.7978845608028654*(x+.044715*x*x*x))
                        return z*(.5*(1+t)+.5*x*(1-t*t)*.7978845608028654*(1+3*.044715*x*x))
                    if name == "silu": return F.silu(x)
                    if name == "silu_backward":
                        s = torch.sigmoid(x); return z*s*(1+x*(1-s))
                    if name == "rmsnorm": return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-5)
                    if name == "rmsnorm_backward":
                        r = torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-5)
                        return r*(z-x*r*r*(x*z).mean(-1,keepdim=True))
                    if name == "softmax": return torch.softmax(x,-1)
                    if name == "softmax_backward": return x*(z-(x*z).sum(-1,keepdim=True))
                    raise ValueError(name)
                out = self.output(shape)
                meth = "ptxas" if self.method == "ptxas" else "oa"
                if (meth,opname) not in opfns:
                    opfns[meth,opname] = driver.function(opmods[meth],opname)
                fn = opfns[meth,opname]
                tensors = (x,z,x,x,out,out,out)
                count = x.numel()//width if row else x.numel()
                args = [C.c_uint64(t.data_ptr()) for t in tensors] + [C.c_uint32(count)]
                params = (C.c_void_p*8)(*[C.cast(C.pointer(v),C.c_void_p) for v in args])
                grid = (count+3)//4 if row else (count+127)//128
                driver.ck(driver.cu.cuLaunchKernel(fn,grid,1,1,128,1,1,0,
                          C.c_void_p(torch.cuda.current_stream().cuda_stream),params,None),opname)
                return out

        def guard_check(self):
            torch.cuda.synchronize()
            passed = all(bool((s[:256]==-12345.).all() and (s[-256:]==-12345.).all()) for s in self.guards)
            rec = {"passed":passed,"guarded_outputs":len(self.guards),"gemm_checks":list(self.checks)}
            self.guards.clear(); self.checks.clear()
            return rec

    class Product(torch.autograd.Function):
        @staticmethod
        def forward(ctx,a,b,backend,name):
            with backend.stage(name+"/pack_forward"):
                al,bl = a.to(torch.float16).contiguous(),backend.pack_weight(b)
            ctx.save_for_backward(al,bl); ctx.backend=backend;ctx.name=name
            return backend.matmul(al,bl,name+"/gemm_forward")
        @staticmethod
        def backward(ctx,g):
            a,b = ctx.saved_tensors; backend,name=ctx.backend,ctx.name
            with backend.stage(name+"/pack_backward"):
                gl=g.to(torch.float16).contiguous();bt=b.T.contiguous();gt=gl.T.contiguous();at=a.T.contiguous()
            return backend.matmul(gl,bt,name+"/gemm_dx"),backend.matmul(gt,at,name+"/gemm_dw"),None,None

    class Unary(torch.autograd.Function):
        @staticmethod
        def forward(ctx,x,backend,name):
            ctx.backend=backend;ctx.name=name
            y=backend.operation(name,x)
            ctx.save_for_backward(y if name=="softmax" else x)
            return y
        @staticmethod
        def backward(ctx,g):
            return ctx.backend.operation(ctx.name+"_backward",ctx.saved_tensors[0],g),None,None

    class Linear(nn.Module):
        def __init__(self,d,n,backend,name):
            super().__init__();self.weight=nn.Parameter(torch.randn(n,d,device="cuda")/math.sqrt(d));self.bias=nn.Parameter(torch.zeros(n,device="cuda"));self.backend=backend;self.name=name
        def forward(self,x):
            product=Product.apply(x,self.weight,self.backend,self.name)
            with self.backend.stage(self.name+"/bias"):
                return product+self.bias

    class Block(nn.Module):
        def __init__(self,c,backend,layer):
            super().__init__();self.c=c;self.backend=backend;d,h=c["width"],c["hidden"]
            self.fc1=Linear(d,h,backend,f"block{layer}/fc1");self.fc2=Linear(h,d,backend,f"block{layer}/fc2")
            self.norm_weight=nn.Parameter(torch.ones(d,device="cuda"))
            if c["family"]=="gated":self.gate=Linear(d,h,backend,f"block{layer}/gate")
            if c["family"]=="transformer":
                self.qkv=Linear(d,3*d,backend,f"block{layer}/qkv");self.proj=Linear(d,d,backend,f"block{layer}/proj")
                self.attn_norm_weight=nn.Parameter(torch.ones(d,device="cuda"))
        def norm(self,x,w):return Unary.apply(x,self.backend,"rmsnorm")*w
        def forward(self,x):
            c,b=self.c,self.backend;d=c["width"]
            if c["family"]=="transformer":
                t,heads=c["seq"],c["heads"];hd=d//heads
                q,k,v=self.qkv(self.norm(x,self.attn_norm_weight)).view(c["batch"],t,3,heads,hd).unbind(2)
                chunks=[]
                for batch in range(c["batch"]):
                    head_results=[]
                    for head in range(heads):
                        qq,kk,vv=q[batch,:,head],k[batch,:,head],v[batch,:,head]
                        logits=Product.apply(qq,kk,b,"attention/qk")/math.sqrt(hd)
                        with b.stage("attention/causal_mask"):
                            mask=torch.triu(torch.ones(t,t,device="cuda",dtype=torch.bool),diagonal=1)
                            logits=logits.masked_fill(mask,-1e4)
                        probabilities=Unary.apply(logits,b,"softmax")
                        head_results.append(Product.apply(probabilities,vv.T,b,"attention/pv"))
                    chunks.append(torch.cat(head_results,dim=-1))
                with b.stage("attention/merge_residual"):
                    x=x+self.proj(torch.cat(chunks,dim=0))
            h=self.fc1(self.norm(x,self.norm_weight))
            h=Unary.apply(h,b,c["activation"])
            if c["family"]=="gated":h=h*self.gate(self.norm(x,self.norm_weight))
            with b.stage("mlp/residual"):
                return x+self.fc2(h)

    class Network(nn.Module):
        def __init__(self,c,backend):
            super().__init__();self.c=c;self.backend=backend
            self.blocks=nn.ModuleList([Block(c,backend,i) for i in range(c["layers"])])
            if c["family"]=="transformer":
                self.embedding=nn.Embedding(256,c["width"],device="cuda")
                self.position=nn.Embedding(c["seq"],c["width"],device="cuda")
                self.head=Linear(c["width"],256,backend,"head")
        def forward(self,x):
            if self.c["family"]=="transformer":
                with self.backend.stage("embedding/position"):
                    position=torch.arange(self.c["seq"],device="cuda")
                    x=(self.embedding(x)+self.position(position)[None]).reshape(-1,self.c["width"])
            for block in self.blocks:x=block(x)
            if self.c["family"]=="transformer":x=self.head(x)
            return x

    def snapshot(model,opt,out,loss,x):
        state={"output/prediction":out.detach().clone(),"output/loss":loss.detach().clone()}
        if x.grad is not None:state["gradient/input"]=x.grad.detach().clone()
        for name,p in model.named_parameters():
            state["parameter/"+name]=p.detach().clone()
            if p.grad is not None:state["gradient/clipped/"+name]=p.grad.detach().clone()
            if name in model.backend.raw_gradients:
                state["gradient/raw/"+name]=model.backend.raw_gradients[name].clone()
            for key,value in opt.state[p].items():
                if isinstance(value,torch.Tensor):state["optimizer/"+name+"/"+key]=value.detach().clone()
        return state

    def training_step(model,opt,x,target):
        with model.backend.stage("optimizer/zero_grad"):
            opt.zero_grad(set_to_none=False)
            if x.grad is not None:x.grad.zero_()
        out=model(x)
        with model.backend.stage("loss"):
            loss=F.cross_entropy(out,target) if model.c["family"]=="transformer" else F.mse_loss(out,target)
        with model.backend.stage("backward"):
            loss.backward()
        if model.backend.collect_raw:
            model.backend.raw_gradients = {n: p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None}
        with model.backend.stage("optimizer/clip"):
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,foreach=False)
        with model.backend.stage("optimizer/adamw"):
            opt.step()
        return out,loss

    methods=("oa_legacy","oa_direct","oa_tensor_only","ptxas","ptxas_tensor_only","cuda_library")
    reference_methods={m:("ptxas_tensor_only" if m in ("oa_tensor_only","ptxas_tensor_only","cuda_library") else "ptxas") for m in methods}
    results=[];started=time.monotonic();capture_failed=False
    try:
        for ci,c in enumerate(payload["cases"]):
            row={"case":c,"status":"not_run","gates":payload["gate"]};runners={};checks={};states={}
            try:
                torch.manual_seed(20261107+ci)
                template=Network(c,Backend("ptxas"));initial=copy.deepcopy(template.state_dict());del template
                data=torch.randint(256,(c["batch"],c["seq"]),device="cuda") if c["family"]=="transformer" else torch.randn(c["tokens"],c["width"],device="cuda")
                target=torch.randint(256,(c["tokens"],),device="cuda") if c["family"]=="transformer" else torch.randn_like(data)*.25
                for method in methods:
                    b=Backend(method);model=Network(c,b);model.load_state_dict(initial)
                    b.weight_ids={id(module.weight) for module in model.modules() if isinstance(module,Linear)}
                    x=data.detach().clone().requires_grad_(c["family"]!="transformer")
                    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,eps=1e-8,weight_decay=.01,foreach=False,capturable=True)
                    b.validate=True; b.collect_raw=True
                    for _ in range(verification_steps):out,loss=training_step(model,opt,x,target)
                    guard=b.guard_check();b.validate=False;b.collect_raw=False
                    states[method]=snapshot(model,opt,out,loss,x)
                    checks[method]={"guards":guard}
                    runners[method]=(model,opt,x,target)
                for method in methods:
                    checks[method]["reference_method"]=reference_methods[method]
                    checks[method]["state"]=compare(states[method],states[reference_methods[method]],payload["gate"])
                    checks[method]["passed"]=checks[method]["guards"]["passed"] and all(q["passed"] for q in checks[method]["guards"]["gemm_checks"]) and checks[method]["state"]["passed"]
                row["verification"]=checks;row["verification_steps"]=verification_steps
                eligible=[m for m in methods if checks[m]["passed"]]
                row["eligible_for_timing"]=eligible
                if "ptxas" not in eligible or "ptxas_tensor_only" not in eligible:
                    row["status"]="control_correctness_failure";results.append(row);continue
                # Identical numbers of updates for every admitted method. Host
                # and GPU spans are both retained; no per-stage events here.
                rng=random.Random(20261107+ci)
                for method in eligible:
                    for _ in range(3):training_step(*runners[method])
                torch.cuda.synchronize()
                times={m:[] for m in eligible};walls={m:[] for m in eligible};orders=[]
                for _ in range(samples):
                    order=list(eligible);rng.shuffle(order);orders.append(order)
                    for method in order:
                        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        wall=time.perf_counter();a.record()
                        for _ in range(timed_steps):out,loss=training_step(*runners[method])
                        b.record();b.synchronize()
                        times[method].append(a.elapsed_time(b)/timed_steps)
                        walls[method].append((time.perf_counter()-wall)*1000/timed_steps)
                row["training"]={m:{"cuda_event":summarize(times[m]),"host_wall":summarize(walls[m])} for m in eligible}
                row["orders"]=orders
                post={}
                for method in eligible:
                    model,opt,x,y=runners[method]
                    # One additional untimed update checks raw gradients without
                    # adding any cloning to the measured intervals.
                    model.backend.collect_raw=True
                    out,loss=training_step(model,opt,x,y)
                    model.backend.collect_raw=False
                    post[method]=snapshot(model,opt,out,loss,x)
                row["post_timing_verification"]={m:compare(post[m],post[reference_methods[m]],payload["gate"]) for m in eligible}
                row["admitted_training_methods"]=[m for m in eligible if row["post_timing_verification"][m]["passed"]]
                for method in eligible:
                    row["training"][method]["admitted"]=method in row["admitted_training_methods"]
                if training_graph_experiment:
                    # Capture the entire training update, not just individual
                    # kernels. First compare a replay with an uncaptured update
                    # from identical state and a changed input. Then time and
                    # recheck all model/gradient/Adam tensors across compilers.
                    tg={};tg_checks={};tg_errors={};tg_outputs={}
                    graph_methods=[m for m in row["admitted_training_methods"] if m != "cuda_library"]
                    row["training_graph_exclusions"]={"cuda_library":"Retained training-capture probes failed with legacy-stream dependencies, including a ptxas method. This variant first tests native/ptxas after recreating gradient buffers inside capture; it does not admit a library training-graph comparison."}
                    for method in graph_methods:
                        model,opt,x,y=runners[method];back=model.backend;back.raw_gradients={}
                        try:
                            # Old gradient buffers were produced on the default
                            # stream. Allocate the first backward's gradient
                            # buffers within capture, following the graph API's
                            # training contract instead of capturing zeroing of
                            # external buffers with cross-stream history.
                            opt.zero_grad(set_to_none=True)
                            x.grad=None
                            torch.cuda.synchronize()
                            graph=torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):gout,gloss=training_step(model,opt,x,y)
                            clone=Network(c,Backend(method));clone.load_state_dict(copy.deepcopy(model.state_dict()))
                            cop=torch.optim.AdamW(clone.parameters(),lr=1e-4,eps=1e-8,weight_decay=.01,foreach=False,capturable=True)
                            cop.load_state_dict(copy.deepcopy(opt.state_dict()))
                            original=x.detach().clone()
                            alternate=(original+1)%256 if c["family"]=="transformer" else -original+.125
                            x.copy_(alternate) if not x.requires_grad else x.detach().copy_(alternate)
                            cx=alternate.detach().clone().requires_grad_(x.requires_grad)
                            replay_checks=[]
                            for proof_step in range(3):
                                co,cl=training_step(clone,cop,cx,y)
                                gout.detach().fill_(float("nan"));gloss.detach().fill_(float("nan"))
                                graph.replay();torch.cuda.synchronize()
                                replay_checks.append(compare(snapshot(model,opt,gout,gloss,x),snapshot(clone,cop,co,cl,cx),payload["gate"]))
                            tg_checks[method]={**replay_checks[-1],"passed":all(check["passed"] for check in replay_checks),"consecutive_update_checks":replay_checks}
                            if tg_checks[method]["passed"]:tg[method]=graph;tg_outputs[method]=(gout,gloss)
                            del clone,cop,cx
                        except Exception as exc:
                            tg_errors[method]=f"{type(exc).__name__}: {exc}"
                            print({"training_graph_error":method,"error":tg_errors[method],"traceback":traceback.format_exc()},flush=True)
                            capture_failed=True
                            row["training_graph_errors"]=tg_errors
                            raise
                        finally:
                            if "original" in locals():
                                with torch.no_grad():x.copy_(original)
                    gt={m:[] for m in tg};gorders=[]
                    # A method is timed only with its own admitted reference.
                    tg={m:g for m,g in tg.items() if reference_methods[m] in tg}
                    for _ in range(samples):
                        order=list(tg);rng.shuffle(order);gorders.append(order)
                        for method in order:
                            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                            a.record()
                            for _ in range(timed_steps):tg[method].replay()
                            b.record();b.synchronize();gt[method].append(a.elapsed_time(b)/timed_steps)
                    gs={}
                    for method in tg:
                        model,opt,x,y=runners[method];model.backend.collect_raw=True
                        out,loss=training_step(model,opt,x,y);model.backend.collect_raw=False
                        gs[method]=snapshot(model,opt,out,loss,x)
                    row["training_graph"]={m:{"cuda_event":summarize(gt[m]),"changed_input_replay_check":tg_checks[m],
                        "post_timing_check":compare(gs[m],gs[reference_methods[m]],payload["gate"])} for m in tg}
                    for value in row["training_graph"].values():value["admitted"]=value["post_timing_check"]["passed"]
                    row["training_graph_errors"]=tg_errors;row["training_graph_checks"]=tg_checks;row["training_graph_orders"]=gorders
                # Inference on one identical immutable parameter snapshot, with
                # weight conversion deliberately included on every invocation.
                inference_candidates=[m for m in methods if checks[m]["guards"]["passed"] and all(q["passed"] for q in checks[m]["guards"]["gemm_checks"])]
                for method in inference_candidates:runners[method][0].load_state_dict(initial)
                inference={m:[] for m in inference_candidates};inference_graph={m:[] for m in inference_candidates};graphs={};graph_errors={}
                with torch.no_grad():
                    expected_outputs={};inference_checks={}
                    for method in (["ptxas","ptxas_tensor_only"]+[m for m in inference_candidates if m not in ("ptxas","ptxas_tensor_only")]):
                        if method not in inference_candidates:continue
                        model,opt,x,y=runners[method];out=model(x)
                        if method in ("ptxas","ptxas_tensor_only"):expected_outputs[method]=out.detach().clone()
                        expected=expected_outputs[reference_methods[method]]
                        inference_checks[method]=compare({"output/prediction":out},{"output/prediction":expected},payload["gate"])
                        original=x.detach().clone()
                        try:
                            graph=torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):captured=model(x)
                            # Changing input verifies that every native ctypes
                            # launch, including intermediates, is in the graph.
                            alternate=(original+1)%256 if c["family"]=="transformer" else -original+.125
                            x.copy_(alternate);alternate_expected=model(x).detach().clone()
                            captured.fill_(float("nan"));graph.replay();torch.cuda.synchronize()
                            inference_checks[method]["graph_changed_input"]=compare({"output/prediction":captured},{"output/prediction":alternate_expected},payload["gate"])
                            x.copy_(original);captured.fill_(float("nan"));graph.replay();torch.cuda.synchronize()
                            inference_checks[method]["graph"]=compare({"output/prediction":captured},{"output/prediction":expected},payload["gate"])
                            if all(inference_checks[method][key]["passed"] for key in ("graph","graph_changed_input")) and inference_checks[method]["passed"]:graphs[method]=(graph,captured)
                        except Exception as exc:graph_errors[method]=f"{type(exc).__name__}: {exc}"
                        finally:x.copy_(original)
                    for _ in range(samples):
                        order=[m for m in inference_candidates if inference_checks[m]["passed"]];rng.shuffle(order)
                        for method in order:
                            model,opt,x,y=runners[method]
                            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                            a.record()
                            for _ in range(timed_steps):model(x)
                            b.record();b.synchronize();inference[method].append(a.elapsed_time(b)/timed_steps)
                            if method in graphs:
                                a.record()
                                for _ in range(timed_steps):graphs[method][0].replay()
                                b.record();b.synchronize();inference_graph[method].append(a.elapsed_time(b)/timed_steps)
                row["inference"]={m:{"event":summarize(inference[m]) if inference[m] else None,"graph":summarize(inference_graph[m]) if inference_graph[m] else None,"check":inference_checks[m],"admitted":inference_checks[m]["passed"]} for m in inference_candidates}
                row["graph_errors"]=graph_errors
                if cache_experiment:
                    cached={}
                    with torch.no_grad():
                        for method in inference_candidates:
                            if not inference_checks[method]["passed"]:continue
                            model,opt,x,y=runners[method];back=model.backend
                            back.cache_enabled=True;back.weight_cache={}
                            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                            a.record();out=model(x);b.record();b.synchronize()
                            item={"cold_pack_and_forward_ms":a.elapsed_time(b),
                                  "check":compare({"output/prediction":out},{"output/prediction":expected_outputs[reference_methods[method]]},payload["gate"])}
                            # Validate safe cache invalidation on a real versioned
                            # parameter mutation, before any captured graph.
                            parameter=next(module.weight for module in model.modules() if isinstance(module,Linear))
                            original_weight=parameter.detach().clone()
                            parameter.view(-1)[0].add_(.125)
                            cached_changed=model(x)
                            back.cache_enabled=False;uncached_changed=model(x);back.cache_enabled=True
                            item["weight_mutation_check"]=compare({"output/prediction":cached_changed},{"output/prediction":uncached_changed},payload["gate"])
                            parameter.copy_(original_weight)
                            out=model(x)
                            item["restored_check"]=compare({"output/prediction":out},{"output/prediction":expected_outputs[reference_methods[method]]},payload["gate"])
                            graph=torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):captured=model(x)
                            original=x.detach().clone()
                            try:
                                x.copy_((original+1)%256 if c["family"]=="transformer" else -original+.125)
                                changed_truth=model(x).detach().clone()
                                captured.fill_(float("nan"));graph.replay();torch.cuda.synchronize()
                                item["graph_changed_input_check"]=compare({"output/prediction":captured},{"output/prediction":changed_truth},payload["gate"])
                            finally:x.copy_(original)
                            captured.fill_(float("nan"));graph.replay();torch.cuda.synchronize()
                            item["graph_original_check"]=compare({"output/prediction":captured},{"output/prediction":expected_outputs[reference_methods[method]]},payload["gate"])
                            item["packed_bytes"]=sum(v.numel()*v.element_size() for v in back.weight_cache.values())
                            item["admitted"]=method in graphs and all(item[k]["passed"] for k in ("check","weight_mutation_check","restored_check","graph_changed_input_check","graph_original_check"))
                            hot=[];graph_times=[]
                            if item["admitted"]:
                                # Within-method paired cache on/off timing avoids
                                # attributing a different method's ordering to cache.
                                paired={"cached":[],"uncached":[],"graph_cached":[],"graph_uncached":[]}
                                uncached_graph=graphs[method][0]
                                for _ in range(samples):
                                    order=list(paired);rng.shuffle(order)
                                    for label in order:
                                        back.cache_enabled=label=="cached"
                                        a.record()
                                        for _ in range(timed_steps):
                                            if label=="graph_cached":graph.replay()
                                            elif label=="graph_uncached":uncached_graph.replay()
                                            else:model(x)
                                        b.record();b.synchronize();paired[label].append(a.elapsed_time(b)/timed_steps)
                                item["paired_timing"]={label:summarize(values) for label,values in paired.items()}
                            cached[method]=item;back.cache_enabled=False
                    row["inference_weight_cache"]=cached
                # Stage spans are independent diagnostics, overlap/nest, and are
                # explicitly not summed to invent an application speedup.
                stages={}
                for method in eligible:
                    model,opt,x,y=runners[method];b=model.backend;b.trace=[];b.counts={};b.trace_enabled=True
                    training_step(model,opt,x,y);torch.cuda.synchronize();b.trace_enabled=False
                    grouped={}
                    for name,a,e in b.trace:grouped.setdefault(name,[]).append(a.elapsed_time(e))
                    stages[method]={"spans":{name:summarize(v) for name,v in grouped.items()},"semantic_call_counts":b.counts}
                row["stage_diagnostics"]=stages
                row["status"]="completed";print({"whole":c["name"],"eligible":eligible,"training_ms":{m:v["cuda_event"]["median_ms"] for m,v in row["training"].items()}},flush=True)
            except Exception as exc:
                row.update(status="error",error=f"{type(exc).__name__}: {exc}",traceback=traceback.format_exc());print(row["error"],flush=True)
            results.append(row)
            del runners,states
            if capture_failed:break
            torch.cuda.empty_cache()
    finally:
        if not capture_failed:
            destroy(handle)
            for mod in modules:driver.unload(mod)
    return {"schema":1,"cases":results,"runtime":{"torch":torch.__version__,"cuda":torch.version.cuda,**driver.info().__dict__},
            "resources":resources,"elapsed_seconds":time.monotonic()-started,
            "contract":"FP32 master params/normalization/activation/loss/AdamW; GEMM input rounding to FP16, FP32 accumulation/output and explicit straight-through backward. No dropout. Mask uses finite -1e4. Fixed resident data; no KV cache.",
            "scope":"Native-routed experiments, not automatic whole-program compilation. oa_tensor_only shares Torch pointwise/reductions; oa_legacy/direct use native operators when present. Embedding/masks/loss/bias/affine/packing/clipping/optimizer are shared CUDA; cuda_library uses matched GemmEx and shared Torch operations. No Inductor attribution.",
            "timing":"Paired shuffled steady updates include zeroing, casts/transposes, saved activations, forward/backward/clipping/AdamW and host enqueue gaps; data loading/initial compilation excluded. Stage spans nest and are nonadditive. Inference graph capture separately poisoned and checked.",
            "total_training_updates_per_timed_method":verification_steps+3+samples*timed_steps+1,
            "post_timing_gate_required":True,"inference_weight_cache_experiment":cache_experiment,
            "training_graph_experiment":training_graph_experiment,"capture_failed":capture_failed}
