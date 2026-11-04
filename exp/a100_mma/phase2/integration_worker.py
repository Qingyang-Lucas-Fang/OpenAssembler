"""Explicit autograd integration: generated GEMM cubins plus measured adapters.

Layouts/tails use torch transpose/contiguous and zero padding. They are part of
the timed path, not silently attributed to native tensor-kernel support.
"""
import base64,ctypes as C,json,statistics,sys,time
from pathlib import Path
import torch
import torch.nn.functional as F
from openassembler.gpu.cuda_driver import Driver


class CompilerGemm:
    def __init__(self,binaries):
        torch.cuda.init();self.context_anchor=torch.empty(1,device='cuda')
        self.drv=Driver();self.handles=[];self.functions={}
        self.launches=[]
        self.drv.cu.cuFuncSetAttribute.argtypes=[C.c_void_p,C.c_int,C.c_int]
        for record in binaries:
            module=self.drv.load(base64.b64decode(record['binary']));self.handles.append(module)
            fn=self.drv.function(module,'gemm')
            if record['smem']:self.drv.ck(self.drv.cu.cuFuncSetAttribute(fn,8,record['smem']),'smem')
            self.functions[tuple(record['shape'])]=(fn,record)

    def mm(self,a,b,bias=None):
        # Logical A[M,K] @ B[N,K]. Padded full tiles provide zero-exact tails.
        m,k=a.shape;n,kb=b.shape
        if k!=kb:raise ValueError('shape mismatch')
        pm=(m+127)//128*128;pn=(n+127)//128*128;pk=max(128,(k+63)//64*64)
        aa=F.pad(a.contiguous(),(0,pk-k,0,pm-m))
        bb=F.pad(b.contiguous(),(0,pk-k,0,pn-n))
        out=torch.empty((pm,pn),dtype=torch.float32,device=a.device)
        fn,record=self.functions[(pm,pn,pk)]
        values=[C.c_uint64(x.data_ptr()) for x in (aa,bb,out)]
        params=(C.c_void_p*3)(*[C.cast(C.pointer(x),C.c_void_p) for x in values])
        if record.get('bias_fused'):
            if bias is None:bias=torch.zeros(pn,device=a.device,dtype=torch.float32)
            else:bias=F.pad(bias.float().contiguous(),(0,pn-n))
            values.append(C.c_uint64(bias.data_ptr()))
            params=(C.c_void_p*4)(*[C.cast(C.pointer(x),C.c_void_p) for x in values])
        self.drv.ck(self.drv.cu.cuLaunchKernel(fn,pm//128,pn//128,1,128,1,1,record['smem'],
                    C.c_void_p(torch.cuda.current_stream().cuda_stream),params,None),'GEMM launch')
        result=out[:m,:n]
        return result if record.get('bias_fused') or bias is None else result+bias


def run(case):
    torch.set_num_threads(2);torch.manual_seed(73)
    dtype={'f16':torch.float16,'bf16':torch.bfloat16}[case['dtype']]
    engine=CompilerGemm(case['binaries'])
    class Linear(torch.autograd.Function):
        @staticmethod
        def forward(ctx,x,w,bias):
            ctx.save_for_backward(x,w)
            return engine.mm(x,w,bias)
        @staticmethod
        def backward(ctx,dy):
            x,w=ctx.saved_tensors
            low=dy.to(dtype)
            dx=engine.mm(low,w.T).to(dtype)
            dw=engine.mm(low.T,x.T).to(dtype)
            return dx,dw,dy.sum(0)
    m,n,k=case['logical_shape']
    x0=torch.randn((m,k),device='cuda',dtype=dtype)*0.1
    w0=torch.randn((n,k),device='cuda',dtype=dtype)*0.1
    b0=torch.randn((n,),device='cuda',dtype=torch.float32)*0.01
    target=torch.randn((m,n),device='cuda',dtype=torch.float32)*0.1
    # Reference shares the same low-precision dY policy and FP32 forward output.
    class Reference(torch.autograd.Function):
        @staticmethod
        def forward(ctx,x,w,bias):
            ctx.save_for_backward(x,w)
            return x.float()@w.float().T+bias
        @staticmethod
        def backward(ctx,dy):
            x,w=ctx.saved_tensors;low=dy.to(dtype)
            return (low.float()@w.float()).to(dtype),(low.T.float()@x.float()).to(dtype),dy.sum(0)
    torch.backends.cuda.matmul.allow_tf32=False
    def train(function,steps):
        x=x0.clone().requires_grad_();w=w0.clone().requires_grad_();b=b0.clone().requires_grad_()
        losses=[]
        for _ in range(steps):
            output=function.apply(x,w,b)
            loss=(output-target).square().mean();loss.backward();losses.append(loss.detach())
            with torch.no_grad():
                w.add_(w.grad,alpha=-0.01);b.add_(b.grad,alpha=-0.01)
            x.grad=None;w.grad=None;b.grad=None
        return output.detach(),w.detach(),b.detach(),torch.stack(losses)
    ours=train(Linear,3);reference=train(Reference,3);torch.cuda.synchronize()
    checks=[]
    for name,a,b in zip(('output','weight','bias','losses'),ours,reference):
        error=(a.float()-b.float());rel=float(torch.linalg.vector_norm(error)/torch.linalg.vector_norm(b.float()).clamp_min(1e-30))
        tolerance=0.003 if dtype==torch.float16 else 0.02
        if rel>tolerance:raise AssertionError(f'training mismatch {name}: {rel}')
        checks.append({'tensor':name,'relative_l2':rel,'max_abs':float(error.abs().max()),'tolerance':tolerance})
    # Explicit dX/dW checks as well as final state checks, with a CPU f64 oracle
    # on the quantized inputs for the padded boundary adapter.
    dy=torch.randn((m,n),device='cuda',dtype=dtype)*0.1
    for name,a,b in (('forward',x0,w0),('dX',dy,w0.T),('dW',dy.T,x0.T)):
        actual=engine.mm(a,b)
        truth=(a.cpu().double()@b.cpu().double().T).float().cuda()
        rel=float(torch.linalg.vector_norm(actual-truth)/torch.linalg.vector_norm(truth).clamp_min(1e-30))
        if rel>2e-5:raise AssertionError(f'{name} oracle mismatch {rel}')
        checks.append({'tensor':name,'relative_l2':rel,'tolerance':2e-5})
    # Host wall timing is deliberately measured: Python/ctypes, padding,
    # transposes, epilogue, gradient casts and SGD all belong to this hook.
    timings={'ours':[],'reference':[]}
    for _ in range(2):train(Linear,1);train(Reference,1)
    for round in range(5):
        for name,function in (('ours',Linear),('reference',Reference)) if round%2 else (('reference',Reference),('ours',Linear)):
            torch.cuda.synchronize();start=time.perf_counter();train(function,3);torch.cuda.synchronize()
            timings[name].append((time.perf_counter()-start)*1000/3)
    return {'name':case['name'],'logical_shape_mnk':[m,n,k],'dtype':case['dtype'],'checks':checks,
            'step_wall_ms':{key:statistics.median(values) for key,values in timings.items()},'samples_ms':timings,
            'scope':'one linear layer + squared loss + dX/dW/db + SGD; includes Python, padding and layout adapters; not a transformer step',
            'reference':'explicit matched FP32 products and low-precision dY; TF32 disabled',
            'native_tail_support':False,'native_layout_support':'RCR; other layouts use contiguous copies'}

if __name__=='__main__':
    import traceback
    try:result=run(json.loads(Path(sys.argv[1]).read_text()))
    except Exception:result={'error':traceback.format_exc()}
    Path(sys.argv[2]).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
    raise SystemExit('error' in result)
