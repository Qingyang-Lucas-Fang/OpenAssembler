"""ctypes cuBLASLt ABI with installed-header validation before any API call."""
from __future__ import annotations

import ctypes as C
import glob
import hashlib
import re
from pathlib import Path

ENUMS={
    'CUBLASLT_MATMUL_DESC_TRANSA':3,
    'CUBLASLT_MATMUL_DESC_TRANSB':4,
    'CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES':1,
    'CUBLASLT_ALGO_CONFIG_ID':0,
    'CUBLASLT_ALGO_CONFIG_TILE_ID':1,
    'CUBLASLT_ALGO_CONFIG_SPLITK_NUM':2,
    'CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME':3,
    'CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING':4,
    'CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION':5,
    'CUBLASLT_ALGO_CONFIG_STAGES_ID':6,
}


class Algo(C.Structure):
    _fields_=[('data',C.c_uint64*8)]


class Heuristic(C.Structure):
    _fields_=[('algo',Algo),('workspaceSize',C.c_size_t),('state',C.c_int),
              ('wavesCount',C.c_float),('reserved',C.c_int*4)]


def validate_header():
    paths=glob.glob('/usr/local/cuda/include/cublasLt.h')+glob.glob('/opt/conda/lib/python*/site-packages/nvidia/cublas/include/cublasLt.h')
    if not paths:raise RuntimeError('installed cublasLt.h required for ABI validation')
    path=Path(paths[0]);text=path.read_text()
    plain=re.sub(r'/\*.*?\*/|//[^\n]*','',text,flags=re.S)
    snippets={}
    for name,expected in ENUMS.items():
        found=re.search(r'\b'+name+r'\s*=\s*(\d+)',plain)
        if not found or int(found.group(1))!=expected:
            raise RuntimeError(f'unsupported installed-header enum {name}')
    for name,expected in (
        ('cublasLtMatmulAlgo_t','uint64_t data[8];'),
        ('cublasLtMatmulHeuristicResult_t',
         'cublasLtMatmulAlgo_t algo; size_t workspaceSize; cublasStatus_t state; float wavesCount; int reserved[4];')):
        match=re.search(r'typedef\s+struct\s*\{([^{}]*)\}\s*'+name+r'\s*;',plain,re.S)
        actual=re.sub(r'\s+',' ',match.group(1)).strip() if match else None
        if re.sub(r'\s+','',actual or '')!=re.sub(r'\s+','',expected):
            raise RuntimeError(f'unsupported installed-header struct {name}: {actual!r}')
        snippets[name]=actual
    if C.sizeof(Algo)!=64 or C.sizeof(Heuristic)!=96 or Heuristic.workspaceSize.offset!=64 or Heuristic.state.offset!=72:
        raise RuntimeError('ctypes structure layout is not the supported 64-bit cuBLASLt ABI')
    names={int(value):name for name,value in re.findall(r'\b(CUBLASLT_MATMUL_TILE_\w+)\s*=\s*(\d+)',plain)}
    stages={int(value):name for name,value in re.findall(r'\b(CUBLASLT_MATMUL_STAGES_\w+)\s*=\s*(\d+)',plain)}
    return {'path':str(path),'sha256':hashlib.sha256(text.encode()).hexdigest(),
            'enums':ENUMS,'structs':snippets,'ctypes_algo_bytes':C.sizeof(Algo),
            'ctypes_heuristic_bytes':C.sizeof(Heuristic),'tile_names':names,'stage_names':stages}


def load_library(stem):
    errors=[]
    for path in [f'{stem}.so.12']+glob.glob(f'/opt/conda/lib/python*/site-packages/nvidia/cublas/lib/{stem}.so*'):
        try:return C.CDLL(path),path
        except OSError as exc:errors.append(str(exc))
    raise RuntimeError(f'cannot load {stem}: {errors}')


def bind(lt,blas):
    vp,i,sz=C.c_void_p,C.c_int,C.c_size_t
    ap,hp=C.POINTER(Algo),C.POINTER(Heuristic)
    signatures={
        'cublasLtCreate':[C.POINTER(vp)],'cublasLtDestroy':[vp],
        'cublasLtMatmulDescCreate':[C.POINTER(vp),i,i],
        'cublasLtMatmulDescDestroy':[vp],
        'cublasLtMatmulDescSetAttribute':[vp,i,vp,sz],
        'cublasLtMatrixLayoutCreate':[C.POINTER(vp),i,C.c_uint64,C.c_uint64,C.c_int64],
        'cublasLtMatrixLayoutDestroy':[vp],
        'cublasLtMatmulPreferenceCreate':[C.POINTER(vp)],
        'cublasLtMatmulPreferenceDestroy':[vp],
        'cublasLtMatmulPreferenceSetAttribute':[vp,i,vp,sz],
        'cublasLtMatmulAlgoGetHeuristic':[vp,vp,vp,vp,vp,vp,vp,i,hp,C.POINTER(i)],
        'cublasLtMatmulAlgoConfigGetAttribute':[ap,i,vp,sz,C.POINTER(sz)],
        'cublasLtMatmulAlgoCheck':[vp,vp,vp,vp,vp,vp,ap,hp],
        'cublasLtMatmul':[vp,vp,vp,vp,vp,vp,vp,vp,vp,vp,vp,vp,ap,vp,sz,vp],
    }
    for name,args in signatures.items():
        fn=getattr(lt,name);fn.argtypes=args;fn.restype=i
    for name,args in {
        'cublasCreate_v2':[C.POINTER(vp)],'cublasDestroy_v2':[vp],
        'cublasSetStream_v2':[vp,vp],'cublasSetMathMode':[vp,i],
        'cublasGetMathMode':[vp,C.POINTER(i)],'cublasGetVersion_v2':[vp,C.POINTER(i)],
        'cublasGemmEx':[vp,i,i,i,i,i,vp,vp,i,i,vp,i,i,vp,vp,i,i,i,i],
    }.items():
        fn=getattr(blas,name);fn.argtypes=args;fn.restype=i


def check(status,operation):
    if status:raise RuntimeError(f'{operation}: cuBLAS status {status}')


def algorithm_config(lt,algo,header):
    values={}
    for name,attr in (('id',0),('tile',1),('split_k',2),('reduction_scheme',3),
                      ('cta_swizzle',4),('custom_option',5),('stages',6)):
        value=C.c_int();written=C.c_size_t()
        status=lt.cublasLtMatmulAlgoConfigGetAttribute(C.byref(algo),attr,C.byref(value),C.sizeof(value),C.byref(written))
        values[name]={'status':status,'bytes_written':written.value,'value':value.value if status==0 else None}
    tile=values['tile']['value'];stage=values['stages']['value']
    values['tile_name']=header['tile_names'].get(tile)
    values['stage_name']=header['stage_names'].get(stage)
    return values
