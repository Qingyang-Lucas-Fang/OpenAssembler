"""Nsight replay of validated legacy/direct pointer homes and matched ptxas."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import textwrap

import modal

APP = modal.App("openassembler-pointer-homes-counters")
IMAGE = (modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
         .add_local_python_source("openassembler", copy=False))
METRICS = [
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__inst_executed.sum",
    "smsp__inst_executed_pipe_tensor.sum",
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_wait_per_warp_active.pct",
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
    "dram__bytes.sum", "lts__t_sector_hit_rate.pct",
]


def worker_source():
    return textwrap.dedent(r'''
        import base64, ctypes as C, glob, json, sys
        from openassembler.gpu.cuda_driver import Driver
        spec=json.load(open(sys.argv[1]))
        driver=Driver(); driver.open()
        m,n,k=spec["shape"]
        a=driver.alloc(m*k*2); b=driver.alloc(n*k*2); c=driver.alloc(m*n*4)
        driver.memset(a,0,m*k*2); driver.memset(b,0,n*k*2); driver.memset(c,0,m*n*4)
        if spec["method"] == "cublas":
            paths=glob.glob("/usr/local/lib/python3*/site-packages/nvidia/cublas/lib/libcublas.so*")
            library=C.CDLL(paths[0] if paths else "libcublas.so.12")
            create=library.cublasCreate_v2
            create.argtypes=[C.POINTER(C.c_void_p)]; create.restype=C.c_int
            destroy=library.cublasDestroy_v2
            destroy.argtypes=[C.c_void_p]; destroy.restype=C.c_int
            gemm=library.cublasGemmEx
            gemm.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_int,C.c_int,
                           C.c_void_p,C.c_void_p,C.c_int,C.c_int,C.c_void_p,C.c_int,C.c_int,
                           C.c_void_p,C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_int]
            gemm.restype=C.c_int
            handle=C.c_void_p()
            rc=create(C.byref(handle))
            if rc: raise RuntimeError(f"cublasCreate:{rc}")
            alpha,beta=C.c_float(1),C.c_float(0)
            rc=gemm(handle,1,0,n,m,k,C.byref(alpha),C.c_void_p(b),2,k,
                    C.c_void_p(a),2,k,C.byref(beta),C.c_void_p(c),0,n,68,99)
            if rc: raise RuntimeError(f"cublasGemmEx:{rc}")
            driver.ck(driver.cu.cuCtxSynchronize(),"cublas synchronize")
            destroy(handle)
        else:
            module=driver.load(base64.b64decode(spec["cubin"]))
            fn=driver.function(module,"gemm")
            driver.ck(driver.cu.cuFuncSetAttribute(fn,8,65536),"shared opt-in")
            args=[C.c_uint64(p) for p in (a,b,c)]
            params=(C.c_void_p*3)(*[C.cast(C.pointer(arg),C.c_void_p) for arg in args])
            driver.ck(driver.cu.cuLaunchKernel(fn,m//128,n//128,1,128,1,1,65536,None,params,None),"launch")
            driver.ck(driver.cu.cuCtxSynchronize(),"synchronize")
            driver.unload(module)
        driver.free(a); driver.free(b); driver.free(c); driver.close()
    ''')


@APP.function(image=IMAGE,gpu="A100-40GB",timeout=600,max_containers=1,retries=0)
def collect(payload):
    import csv
    import io
    import shutil
    import subprocess
    import tempfile
    from openassembler.gpu.cuda_driver import Driver
    ncu=shutil.which("ncu")
    if not ncu:
        return {"status":"unavailable","reason":"no ncu in image"}
    version=subprocess.run([ncu,"--version"],capture_output=True,text=True,check=True).stdout
    rows=[]
    with tempfile.TemporaryDirectory() as td:
        root=Path(td)
        worker=root/"profile.py"
        worker.write_text(worker_source())
        for key,data in payload.items():
            for method in ("baseline","direct_home","ptxas"):
                spec={"method":method,"shape":data["shape"],"cubin":data.get(method,"")}
                job=root/"job.json"; job.write_text(json.dumps(spec))
                command=[ncu,"--clock-control","none","--target-processes","all","--csv",
                         "--page","raw","--metrics",",".join(METRICS),sys.executable,str(worker),str(job)]
                run=subprocess.run(command,capture_output=True,text=True,timeout=90)
                records=list(csv.reader(io.StringIO(run.stdout)))
                head=next((i for i,fields in enumerate(records) if fields and fields[0]=="ID"),None)
                table=records[head:] if head is not None else []
                row={"shape":data["shape"],"method":method,"returncode":run.returncode,
                     "status":"collected" if run.returncode==0 and len(table)>=3 else "failed",
                     "raw_rows":table,"stderr":run.stderr[-5000:]}
                if len(table)>=3:
                    row["metrics"]=dict(zip(table[0],table[-1]))
                rows.append(row)
    return {"schema":1,"device":Driver().info().__dict__,"ncu_version":version,"profiles":rows,
            "scope":"Single NCU replay per method/shape, separate process and zero-filled inputs; default replay cache behavior, clock-control none. Replay duration is not benchmark timing. Cubin hashes match the full-output validated pointer-home experiment payload.",
            "metrics_requested":METRICS}


@APP.local_entrypoint()
def main():
    from exp.weaknesses.gemm_pointer_homes import build_payload
    here=Path(__file__).resolve().parent
    all_payload=build_payload()
    payload={key:all_payload[key] for key in ("512_512_2048_f16","4096_4096_4096_f16","16384_4096_4096_f16")}
    result=collect.remote(payload)
    result["builds"]={key:{k:v for k,v in data.items() if k not in ("baseline","direct_home","ptxas")}
                      for key,data in payload.items()}
    result["timestamp_utc"]=datetime.now(timezone.utc).isoformat()
    output=here/("gemm_pointer_homes_counters-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")+".json")
    output.write_text(json.dumps(result,indent=2)+"\n")
    print(output)
