"""Four correctness-admitted baseline/FMA diagnostic profiles; no GPU dispatch."""
from datetime import datetime,timezone
import base64
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/"exp/nn_round2"


def build():
    payload=json.loads((HERE/"operator_fma_build/payload.json").read_text())
    evidence=HERE/"runs/operator-fma-confirm-20261107T144005.532354Z.json"
    run=json.loads(evidence.read_text())
    if run["status"]!="completed":
        raise ValueError("FMA confirmation must be completed")
    if hashlib.sha256(base64.b64decode(payload["oa"])).hexdigest()!=run["result"]["oa_sha256"]:
        raise ValueError("FMA binary changed after independent hardware confirmation")
    profiles=[]
    for op,warps in (("rmsnorm_backward",4),("softmax_backward",8)):
        seeds=set()
        for candidate in ("baseline","fma"):
            kernel=f"op_{op}_w513_c{warps}_{candidate}"
            meta=next(k for k in payload["kernels"] if k["kernel"]==kernel)
            checks=[row for row in run["result"]["cases"] if row["kernel"]==kernel and row["shape"]==[4096,513]]
            if len(checks)!=2 or any(row["status"]!="ok" or "oa" not in row["eligible_for_timing"] or not row["graph_checks"]["oa"]["passed"] for row in checks):
                raise ValueError("FMA counter candidate must pass both confirmation rounds: "+kernel)
            seed=checks[0]["input_seed"]
            if any(row["input_seed"]!=seed for row in checks):
                raise ValueError("confirmation changed candidate input seed")
            seeds.add(seed)
            profiles.append({"id":kernel+"/oa","kind":"operator","method":"oa_"+candidate,
                             "kernel":kernel,"shape":[4096,513],"threads":warps*32,
                             "grid":[(4096+warps-1)//warps,1,1],"smem":0,"operator":meta,
                             "cubin":payload["oa"],"cubin_sha256":payload["oa_sha256"],
                             "source_sha256":payload["source_sha256"],"seed":seed,
                             "admitted_evidence":str(evidence.relative_to(ROOT)),
                             "admitted_evidence_sha256":hashlib.sha256(evidence.read_bytes()).hexdigest(),
                             "rounding_policy":meta["rounding_policy"]})
        if len(seeds)!=1:
            raise ValueError("baseline/FMA comparison requires exactly matched input seed")
    return {"schema":1,"created_utc":datetime.now(timezone.utc).isoformat(),
            "profile_timeout_seconds":90,"profiles":profiles,
            "selection":"Independent-confirmation winners: RMSNorm backward width513 warp4 and softmax backward width513 warp8, 4096rows, OA baseline versus explicit local FMA. Original tolerances, synchronization and seeds unchanged.",
            "scope":"Diagnostic metrics for a separately timed optimization. One profiled matching launch per candidate; no NCU replay timing speed claim. XU pipe counters, if supported, are not SFU-only counters.",
            "source_identities":{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in (Path(__file__),HERE/"counter_worker.py")}}


if __name__=="__main__":
    result=build()
    path=HERE/"counter_fma_payload.json"
    path.write_text(json.dumps(result)+"\n")
    print(json.dumps({"profiles":len(result["profiles"]),"output":str(path)},indent=2))
