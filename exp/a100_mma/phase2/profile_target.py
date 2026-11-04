"""A single cubin, regular launches, no graph capture: target for Nsight Compute."""
import base64
import ctypes as C
import json
import sys
from pathlib import Path
import torch
from openassembler.gpu.cuda_driver import Driver

case=json.loads(Path(sys.argv[1]).read_text())
torch.cuda.init();drv=Driver()
m,n,k=case['shape']
dtype={'f16':torch.float16,'bf16':torch.bfloat16}[case['dtype']]
torch.manual_seed(7)
a=torch.randn((m,k),device='cuda',dtype=dtype)
b=torch.randn((n,k),device='cuda',dtype=dtype)
c=torch.empty((m,n),device='cuda',dtype=torch.float32)
mod=drv.load(base64.b64decode(case[sys.argv[2]]))
fn=drv.function(mod,'gemm')
smem=case.get('smem',0)
if smem:
    drv.cu.cuFuncSetAttribute.argtypes=[C.c_void_p,C.c_int,C.c_int]
    drv.ck(drv.cu.cuFuncSetAttribute(fn,8,smem),'shared opt-in')
ptrs=[C.c_uint64(x.data_ptr()) for x in (a,b,c)]
params=(C.c_void_p*3)(*[C.cast(C.pointer(v),C.c_void_p) for v in ptrs])
for _ in range(5):
    drv.ck(drv.cu.cuLaunchKernel(fn,*case['grid'],case['threads'],1,1,smem,
                               C.c_void_p(torch.cuda.current_stream().cuda_stream),params,None),'launch')
    torch.cuda.synchronize()
if not torch.isfinite(c).all():raise RuntimeError('nonfinite profile output')
print(json.dumps({'resources':drv.func_attrs(fn),'output_norm':float(torch.linalg.vector_norm(c))}))
