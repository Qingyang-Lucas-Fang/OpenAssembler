"""Recover wide NCU CSV from immutable original records without another GPU run.

The original run's parser failures remain intact. This derived record records
the original hash, parser hash, raw warnings and independently checks every
profile's child proof, process success, requested metrics and kernel identity.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

from exp.nn_round2.counter_worker import (metric_query_accepted,parse_ncu_csv,
                                          parse_proof)

ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/"exp/nn_round2"
DEFAULT=HERE/"runs/counters-20261107T143821.759059Z.json"


def recover(path=DEFAULT):
    path=Path(path).resolve()
    original=json.loads(path.read_text())
    source=original["result"]
    requested=source["metrics_requested"]
    queries={name:metric_query_accepted(query) for name,query in source["metric_queries"].items()}
    profiles=[]
    for profile in source["profiles"]:
        process=profile["process"]
        parsed=parse_ncu_csv(process["stdout"])
        proof=parse_proof(process["stdout"])
        values={name:parsed["metrics"].get(name,[]) for name in requested}
        missing=[name for name in requested if not values[name]]
        available=[name for name in requested if values[name]]
        kernels={row["kernel"] for value in values.values() for row in value}
        identity_ok=kernels=={profile["kernel"]}
        proof_ok=(proof is not None and proof.get("passed") is True and
                  proof.get("cubin_sha256")==profile["cubin_sha256"] and
                  proof.get("matching_launches")==3 and proof.get("profiled_matching_launch_index")==2)
        success=process["returncode"]==0 and not process["timed_out"] and bool(available) and identity_ok and proof_ok
        item={key:value for key,value in profile.items() if key not in ("process","command","raw_rows","metric_records","metrics","proof")}
        item.update(original_parser_status=profile["status"],recovered_status="collected_partial_metrics" if success and missing else "collected" if success else "failed",
                    csv_format=parsed["format"],proof=proof,requested_metrics=values,
                    missing_requested_metrics=missing,available_requested_metrics=available,
                    total_raw_metric_columns=len(parsed["metrics"]),kernel_identity_check=identity_ok,
                    process_returncode=process["returncode"],process_elapsed_seconds=process["elapsed_seconds"],
                    warnings=[line for line in process["stdout"].splitlines() if "==WARNING==" in line],
                    raw_stdout_sha256=hashlib.sha256(process["stdout"].encode()).hexdigest())
        profiles.append(item)
    return {"schema":1,"created_utc":datetime.now(timezone.utc).isoformat(),
            "derived_from":str(path.relative_to(ROOT)),"derived_from_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "parser_sha256":hashlib.sha256((HERE/"counter_worker.py").read_bytes()).hexdigest(),
            "recovery_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "original_status":source["status"],"profiles":profiles,"runtime":source["device"],
            "ncu_version":source["version"],"metric_query_acceptance_corrected":queries,
            "finding":"Raw NCU --page raw uses a wide header/unit/data table. The original long-only parser returned no metrics despite successful profiling. Invalid SFU metric emitted a zero-exit warning and is absent; do not interpret it as zero.",
            "scope":"No new GPU execution or changed raw records. Diagnostic replay with uncontrolled cache/clocks; one profiled matching launch per method, complete mathematical outputs/guards/nonmutation checked before and after profiling. Metric timings are not benchmark admissions.",
            "counts":{"profiles":len(profiles),"successful_profiles":sum(row["recovered_status"].startswith("collected") for row in profiles),
                      "complete_metric_profiles":sum(row["recovered_status"]=="collected" for row in profiles),
                      "partial_metric_profiles":sum(row["recovered_status"]=="collected_partial_metrics" for row in profiles)}}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence",nargs="?",type=Path,default=DEFAULT)
    parser.add_argument("--out",type=Path,default=HERE/"counter_recovered_20261107T143821.759059Z.json")
    args=parser.parse_args()
    if not args.out.resolve().is_relative_to(HERE):
        raise ValueError("counter recovery output must remain under exp/nn_round2")
    result=recover(args.evidence)
    args.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"output":str(args.out),"counts":result["counts"]},indent=2))
