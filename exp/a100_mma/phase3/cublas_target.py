"""Exact matched-precision cuBLAS target for dispatch and SASS observation.

Uses the installed vendor library, preserving it in place. No vendor binary is
copied into the OpenAssembler package or used as generated code.
"""
import argparse,ctypes as C,glob,json
import torch

p=argparse.ArgumentParser();p.add_argument('--m',type=int,default=4096);p.add_argument('--n',type=int,default=4096)
p.add_argument('--k',type=int,default=4096);p.add_argument('--dtype',default='f16',choices=('f16','bf16'))
a=p.parse_args();m,n,k=a.m,a.n,a.k
torch.manual_seed(31)
dtype=torch.float16 if a.dtype=='f16' else torch.bfloat16
x=torch.randn((m,k),device='cuda',dtype=dtype)
w=torch.randn((n,k),device='cuda',dtype=dtype)
y=torch.empty((m,n),device='cuda',dtype=torch.float32)
lib=C.CDLL('libcublas.so.12');vp=C.c_void_p;i=C.c_int
lib.cublasCreate_v2.argtypes=[C.POINTER(vp)]
lib.cublasDestroy_v2.argtypes=[vp]
lib.cublasSetStream_v2.argtypes=[vp,vp]
lib.cublasGetVersion_v2.argtypes=[vp,C.POINTER(i)]
lib.cublasGemmEx.argtypes=[vp,i,i,i,i,i,vp,vp,i,i,vp,i,i,vp,vp,i,i,i,i]
h=vp()
def ck(rc):
    if rc:raise RuntimeError(f'cuBLAS status {rc}')
ck(lib.cublasCreate_v2(C.byref(h)))
version=i();ck(lib.cublasGetVersion_v2(h,C.byref(version)))
one,zero=C.c_float(1),C.c_float(0)
ck(lib.cublasSetStream_v2(h,vp(torch.cuda.current_stream().cuda_stream)))
for _ in range(6):
    ck(lib.cublasGemmEx(h,1,0,n,m,k,C.byref(one),w.data_ptr(),2 if a.dtype=='f16' else 14,k,
                       x.data_ptr(),2 if a.dtype=='f16' else 14,k,C.byref(zero),y.data_ptr(),0,n,68,99))
    torch.cuda.synchronize()
if not bool(torch.isfinite(y).all()):raise RuntimeError('nonfinite output')
print(json.dumps({'shape_mnk':[m,n,k],'dtype':a.dtype,'accumulation':'f32','output':'f32',
                  'cublas_version':version.value,'device':torch.cuda.get_device_name(),
                  'output_norm':float(torch.linalg.vector_norm(y))}))
ck(lib.cublasDestroy_v2(h))
