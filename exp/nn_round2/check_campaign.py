"""Read-only evidence/admission checks; --seal creates the initial custody manifest.

No compilation, network or GPU request. A seal cannot silently replace an
existing manifest. Regenerable build products are outside the custody list.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/"exp/nn_round2"
MANIFEST=HERE/"evidence_manifest.json"
PRIMARY={
    "operators":"operators-20261107T143128.366417Z.json",
    "tensor":"tensor-20261107T143201.758879Z.json",
    "whole":"whole-20261107T143549.109468Z.json",
    "cache":"whole-20261107T144210.754688Z.json",
    "fma":"operator-fma-20261107T143537.650425Z.json",
    "fma_confirmation":"operator-fma-confirm-20261107T144005.532354Z.json",
    "fma_standard":"operator-standard-fma-20261107T144401.126097Z.json",
    "gelu":"operator-gelu-stable-20261107T143841.822588Z.json",
    "tensor_advance_fp16":"tensor-followup-20261107T143807.885774Z.json",
    "tensor_advance_bf16":"tensor-followup-20261107T144047.961466Z.json",
    "counter_fma":"counters-20261107T144534.159647Z.json",
}


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def load_run(name):
    path=HERE/"runs"/PRIMARY[name]
    data=json.loads(path.read_text())
    assert data["status"]=="completed", (name,data["status"])
    return data["result"]


def semantic_checks():
    original=json.loads((HERE/"source_snapshot.json").read_text())
    changed=[name for name,sha in original["source_files"].items()
             if not (ROOT/name).is_file() or digest(ROOT/name)!=sha]
    assert not changed, ("production/spec/test/environment source changed",changed)
    current={str(path.relative_to(ROOT)) for folder in ("openassembler","specs","tests","env")
             for path in (ROOT/folder).rglob("*") if path.is_file() and "__pycache__" not in path.parts
             and path.suffix in (".py",".json",".sh",".toml") and "cuda-12.8" not in path.parts}
    assert current==set(original["source_files"]), ("production source-file set changed",sorted(current^set(original["source_files"])))
    operator=load_run("operators")["cases"]
    counts=Counter(row["status"] for row in operator)
    assert counts=={"ok":264,"correctness_failure":10},counts
    assert {row["op"] for row in operator if row["status"]!="ok"}=={"gelu_tanh_backward"}
    for row in operator:
        if row["status"]!="ok":assert not row.get("graphs"),"failed gate has admitted timing"
    tensor=load_run("tensor")["results"]
    assert len(tensor)==18
    for row in tensor:
        assert row["checks"] and all(check["passed"] for check in row["checks"])
        assert row["graph_checks"] and all(check["passed"] for check in row["graph_checks"])
    candidates={}
    for name,total in (("fma",240),("fma_confirmation",16),("fma_standard",64)):
        rows=load_run(name)["cases"]
        assert len(rows)==total and all(row["status"]=="ok" for row in rows)
        candidates[name]=total
    gelu=load_run("gelu")["cases"]
    stable=[row for row in gelu if "stable_ex2" in row["kernel"]]
    assert len(stable)==26 and all(row["status"]=="ok" for row in stable)
    whole=load_run("whole")["cases"]
    assert len(whole)==6 and all(row["status"]=="completed" for row in whole)
    admitted=sum(len(row["admitted_training_methods"]) for row in whole)
    assert admitted==21,admitted
    for row in whole:
        assert "oa_tensor_only" in row["admitted_training_methods"]
        for method,item in row["training"].items():
            assert item["admitted"]==(method in row["admitted_training_methods"])
            if item["admitted"]:
                assert row["verification"][method]["passed"]
                assert row["post_timing_verification"][method]["passed"]
    cache=load_run("cache")["cases"]
    cache_items=[item for row in cache for item in row["inference_weight_cache"].values()]
    assert len(cache_items)==26 and all(item["admitted"] for item in cache_items)
    for item in cache_items:
        assert all(item[key]["passed"] for key in ("check","weight_mutation_check","restored_check","graph_changed_input_check","graph_original_check"))
    profiles=load_run("counter_fma")["profiles"]
    assert len(profiles)==4 and all(p["status"]=="collected" and p["proof"]["passed"] for p in profiles)
    archive_count=0
    for path in (HERE/"source_history").glob("*.py.gz"):
        assert hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest()==path.name.removesuffix(".py.gz")
        archive_count+=1
    worker_names={"whole_worker.py","counter_worker.py","operator_worker.py","tensor_suite.py"}
    required=set()
    for path in (HERE/"runs").glob("*.json"):
        data=json.loads(path.read_text())
        for name,sha in data.get("harness_sha256",{}).items():
            if Path(name).name in worker_names:required.add(sha)
    missing=[sha for sha in required if not (HERE/"source_history"/(sha+".py.gz")).is_file()]
    assert not missing,("missing executed worker revision",missing)
    return {"production_files_unchanged":len(original["source_files"]),
            "initial_operator_status":dict(counts),"tensor_cases":len(tensor),
            "fma_cases":candidates,"stable_gelu_cases":len(stable),
            "whole_cases":len(whole),"whole_training_pairs_admitted":admitted,
            "cache_pairs_admitted":len(cache_items),"fma_counter_profiles":len(profiles),
            "source_archives_verified":archive_count,"historical_gpu_worker_revisions":len(required)}


def custody_files():
    paths=set(HERE.glob("*.py"))|set(HERE.glob("*.md"))|set(HERE.glob("*.json"))|set(HERE.glob("*.log"))
    paths.update((HERE/"runs").glob("*"));paths.update((HERE/"source_history").glob("*"))
    paths.add(HERE/".gitignore")
    return sorted(path for path in paths if path.is_file() and path!=MANIFEST)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal",action="store_true")
    args=parser.parse_args();checks=semantic_checks()
    if args.seal:
        if MANIFEST.exists():raise SystemExit("Refusing to replace the existing custody manifest")
        files=[{"path":str(path.relative_to(ROOT)),"bytes":path.stat().st_size,"sha256":digest(path)} for path in custody_files()]
        record={"schema":1,"sealed_utc":datetime.now(timezone.utc).isoformat(),"files":files,"checks":checks,
                "primary_runs":{key:"exp/nn_round2/runs/"+name for key,name in PRIMARY.items()},
                "scope":"Experiment/source/result custody. Failed gates and capture/setup attempts remain failed. Generated build caches are reproducible and excluded. No production promotion or general CUDA superiority."}
        MANIFEST.write_text(json.dumps(record,indent=2)+"\n")
    if MANIFEST.exists():
        record=json.loads(MANIFEST.read_text())
        for item in record["files"]:
            path=ROOT/item["path"]
            assert path.is_file() and path.stat().st_size==item["bytes"] and digest(path)==item["sha256"],item["path"]
        checks["custody_files_verified"]=len(record["files"])
    print(json.dumps(checks,indent=2))


if __name__=="__main__":main()
