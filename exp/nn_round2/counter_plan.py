"""Select a bounded counter cohort from full-output-admitted campaign records.

This only prepares a payload under exp/; it never requests a GPU.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT/"exp/nn_round2"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def build_plan(operator_evidence=None, tensor_evidence=None):
    operator_evidence = Path(operator_evidence or HERE/"runs/operators-20261107T143128.366417Z.json").resolve()
    tensor_evidence = Path(tensor_evidence or HERE/"runs/tensor-20261107T143201.758879Z.json").resolve()
    operators = json.loads(operator_evidence.read_text())
    tensors = json.loads(tensor_evidence.read_text())
    if operators["status"] != "completed" or tensors["status"] != "completed":
        raise ValueError("counter controls require completed existing evidence")
    operator_payload = json.loads((HERE/"operator_build/payload.json").read_text())
    profiles = []
    wanted = (("op_gelu_tanh_scalar4",[1048576],128),
              ("op_rmsnorm_backward_w513_c4",[257,513],128))
    operator_rows = operators["result"]["cases"]
    for kernel,shape,threads in wanted:
        record = next(row for row in operator_rows if row["kernel"] == kernel and row["shape"] == shape and row["threads"] == threads)
        if record["status"] != "ok" or len(record["eligible_for_timing"]) != 2 or not all(check["passed"] for check in record["graph_checks"].values()):
            raise ValueError("operator counter candidate did not pass all retained gates: "+kernel)
        index,meta = next((index,meta) for index,meta in enumerate(operator_payload["kernels"]) if meta["kernel"] == kernel)
        for method in ("oa","ptxas"):
            binary = base64.b64decode(operator_payload[method])
            binary_sha = sha(binary)
            if binary_sha != operators["result"][method+"_sha256"]:
                raise ValueError("operator binary changed since admitted evidence")
            count = shape[0]
            grid = (count+meta["elements_per_thread"]*threads-1)//(meta["elements_per_thread"]*threads) if meta["kind"] == "pointwise" else (count+meta["warps"]-1)//meta["warps"]
            profiles.append({"id":kernel+"/"+method,"kind":"operator","method":method,
                             "kernel":kernel,"shape":shape,"threads":threads,"grid":[grid,1,1],
                             "smem":0,"operator":meta,"cubin":operator_payload[method],
                             "cubin_sha256":binary_sha,"source_sha256":operator_payload["source_sha256"],
                             "seed":20261107+index,"admitted_evidence":str(operator_evidence.relative_to(ROOT)),
                             "admitted_evidence_sha256":sha(operator_evidence.read_bytes())})
    wanted_tensor = {
        "mlp_wide_forward_f16": ("oa_legacy","oa_direct","ptxas_m_fast","ptxas_morton8"),
        "tall_f16": ("oa_legacy","oa_morton","ptxas_m_fast","ptxas_morton8"),
    }
    for case_name,methods in wanted_tensor.items():
        case = next(case for case in tensors["result"]["results"] if case["name"] == case_name)
        for method in methods:
            if method not in case["admitted_methods"]:
                raise ValueError("tensor counter candidate not admitted: "+case_name+"/"+method)
            candidate = method if not method.startswith("ptxas_") else "oa_morton" if method == "ptxas_morton8" else "oa_legacy"
            build = next(record for record in tensors["builds"] if record["case"] == case_name and record["candidate"]["name"] == candidate)
            source_id = case["methods"][method]["source_id"]
            if build["source_sha256"] != source_id:
                raise ValueError("tensor source mismatch")
            field,name = ("reference_sha256","ptxas.cubin") if method.startswith("ptxas_") else ("native_sha256","native.cubin")
            path = ROOT/build["path"]/name
            binary = path.read_bytes()
            if sha(binary) != build[field]:
                raise ValueError("tensor binary changed since admitted evidence: "+str(path))
            m,n,k = case["shape"]
            profiles.append({"id":case_name+"/"+method,"kind":"tensor","method":method,
                             "kernel":"gemm","shape":[m,n,k],"dtype":case["dtype"],"threads":128,
                             "grid":[m//128,n//128,1],"smem":65536,"seed":1729,
                             "cubin":base64.b64encode(binary).decode(),"cubin_sha256":sha(binary),
                             "source_sha256":source_id,"admitted_evidence":str(tensor_evidence.relative_to(ROOT)),
                             "admitted_evidence_sha256":sha(tensor_evidence.read_bytes()),
                             "source_plan":build["source_plan"],"build_path":build["path"]})
    return {"schema":1,"created_utc":datetime.now(timezone.utc).isoformat(),
            "profile_timeout_seconds":90,"profiles":profiles,
            "selection":"Two scalar weakness mechanisms (GELU scalar4 and RMSNorm backward), plus matched wide/tall pointer/order tensor controls. Selected after diagnostic timing, not performance claims.",
            "scope":"Counters may explain measured gaps; replay duration is not admitted benchmark timing. Every binary must retain full-output admission and pass another independent full-output check in the profiling process.",
            "source_identities":{str(path.relative_to(ROOT)):sha(path.read_bytes()) for path in (HERE/"counter_plan.py",HERE/"counter_worker.py")}}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operators",type=Path)
    parser.add_argument("--tensor",type=Path)
    parser.add_argument("--out",type=Path,default=HERE/"counter_payload.json")
    args=parser.parse_args()
    if not args.out.resolve().is_relative_to(HERE):
        raise ValueError("counter payload must remain under exp/nn_round2")
    result=build_plan(args.operators,args.tensor)
    args.out.write_text(json.dumps(result)+"\n")
    print(json.dumps({"profiles":len(result["profiles"]),"out":str(args.out)},indent=2))
