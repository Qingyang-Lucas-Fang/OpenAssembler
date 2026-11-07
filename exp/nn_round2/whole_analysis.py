"""Correctness-aware summaries of native-routed whole NN campaign records.

Never treat a stored timing as admitted merely because its row completed.
Training needs both initial and post-trajectory gates; inference and inference
graphs have independent gates. Methods are compared within their actual
reference groups, and diagnostic nested stage spans are not added.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/"exp/nn_round2"
DEFAULT=HERE/"runs/whole-20261107T143549.109468Z.json"


def check_passed(check):
    return isinstance(check,dict) and check.get("passed") is True


def native_guards_passed(verification):
    guard=verification.get("guards",{})
    return check_passed(guard) and all(check_passed(item) for item in guard.get("gemm_checks",[]))


def failure_summary(check):
    failures=[row for row in check.get("details",[]) if not row.get("passed")]
    categories=Counter(row.get("name","unknown").split("/",1)[0] for row in failures)
    return {"tensors_failed":len(failures),"by_category":dict(categories),
            "details":[{key:row[key] for key in ("name","elements","mismatched","max_abs","relative_l2","error") if key in row} for row in failures]}


def reference_for(case,method):
    recorded=case.get("verification",{}).get(method,{}).get("reference_method")
    if recorded:
        return recorded
    return "ptxas_tensor_only" if method in ("oa_tensor_only","ptxas_tensor_only","cuda_library") else "ptxas"


def timing_value(item,key):
    value=item.get(key)
    return value.get("median_ms") if isinstance(value,dict) else None


def ratio(baseline,candidate):
    return baseline/candidate if baseline is not None and candidate is not None and candidate>0 else None


def summarize_case(case):
    verification=case.get("verification",{})
    post=case.get("post_timing_verification",{})
    methods=sorted(verification)
    initial={method:check_passed(verification[method]) for method in methods}
    training={method:initial[method] and check_passed(post.get(method))
              and method in case.get("admitted_training_methods",[])
              and case.get("training",{}).get(method,{}).get("admitted") is True
              for method in methods}
    # A primary control must itself have passed at the same scope.
    training={method:admitted and training.get(reference_for(case,method),False)
              for method,admitted in training.items()}
    records=[]
    for method in methods:
        reference=reference_for(case,method)
        train=case.get("training",{}).get(method,{})
        infer=case.get("inference",{}).get(method,{})
        infercheck=infer.get("check",{})
        # Inference on reset immutable parameters has an independent contract.
        # A failed multi-update training state must not exclude an independently
        # checked inference output, but native guards/GEMM gates still apply.
        inference=native_guards_passed(verification[method]) and infer.get("admitted") is True and check_passed(infercheck)
        graph=inference and infer.get("graph") is not None and check_passed(infercheck.get("graph")) and check_passed(infercheck.get("graph_changed_input"))
        record={"method":method,"reference_method":reference,
                "reference_group":"shared_torch_surrounding" if reference=="ptxas_tensor_only" else "generated_native_operator_sources",
                "initial_admitted":initial[method],"training_admitted":training[method],
                "inference_admitted":inference,"inference_graph_admitted":graph,
                "training_event_ms":timing_value(train,"cuda_event"),
                "training_host_wall_ms":timing_value(train,"host_wall"),
                "inference_event_ms":timing_value(infer,"event"),
                "inference_graph_ms":timing_value(infer,"graph"),
                "initial_failure":failure_summary(verification[method].get("state",{})),
                "post_trajectory_failure":failure_summary(post.get(method,{}))}
        if not initial[method]:
            record["exclusion_reason"]="initial_full_state_or_guard_gate"
        elif not training[method]:
            record["exclusion_reason"]="post_trajectory_full_state_gate"
        records.append(record)
    by_method={record["method"]:record for record in records}
    for record in records:
        reference=by_method.get(record["reference_method"],{})
        for scope in ("inference","inference_graph"):
            record[scope+"_admitted"]=record[scope+"_admitted"] and reference.get(scope+"_admitted",False)
        train_ok=record["training_admitted"]
        infer_ok=record["inference_admitted"]
        graph_ok=record["inference_graph_admitted"]
        record["reference_over_method_training"]=ratio(reference.get("training_event_ms"),record["training_event_ms"]) if train_ok else None
        record["reference_over_method_inference"]=ratio(reference.get("inference_event_ms"),record["inference_event_ms"]) if infer_ok else None
        record["reference_over_method_inference_graph"]=ratio(reference.get("inference_graph_ms"),record["inference_graph_ms"]) if graph_ok else None
        record["within_method_host_over_graph_inference"]=ratio(record["inference_event_ms"],record["inference_graph_ms"]) if graph_ok else None
    comparisons=[]
    for method in ("oa_legacy","oa_direct","oa_tensor_only"):
        actual=by_method.get(method)
        if actual is None:
            continue
        reference=by_method.get(actual["reference_method"])
        if reference:
            comparisons.append({"candidate":method,"baseline":reference["method"],
                                "training_admitted":actual["training_admitted"],
                                "training_baseline_over_candidate":actual["reference_over_method_training"],
                                "inference_admitted":actual["inference_admitted"],
                                "inference_baseline_over_candidate":actual["reference_over_method_inference"],
                                "inference_graph_admitted":actual["inference_graph_admitted"],
                                "inference_graph_baseline_over_candidate":actual["reference_over_method_inference_graph"]})
    actual,library=by_method.get("oa_tensor_only"),by_method.get("cuda_library")
    if actual and library:
        training_ok=actual["training_admitted"] and library["training_admitted"]
        inference_ok=actual["inference_admitted"] and library["inference_admitted"]
        graph_ok=actual["inference_graph_admitted"] and library["inference_graph_admitted"]
        comparisons.append({"candidate":"oa_tensor_only","baseline":"cuda_library",
                            "training_admitted":training_ok,
                            "training_baseline_over_candidate":ratio(library["training_event_ms"],actual["training_event_ms"]) if training_ok else None,
                            "inference_admitted":inference_ok,
                            "inference_baseline_over_candidate":ratio(library["inference_event_ms"],actual["inference_event_ms"]) if inference_ok else None,
                            "inference_graph_admitted":graph_ok,
                            "inference_graph_baseline_over_candidate":ratio(library["inference_graph_ms"],actual["inference_graph_ms"]) if graph_ok else None})
    stage_records={}
    for method,data in case.get("stage_diagnostics",{}).items():
        spans=data.get("spans",{})
        top=sorted(({"name":name,"median_ms":value["median_ms"],"observed_calls":len(value["samples_ms"])} for name,value in spans.items()),key=lambda item:item["median_ms"],reverse=True)[:12]
        stage_records[method]={"training_admitted":training.get(method,False),"largest_nested_spans":top,
                               "semantic_call_counts":data.get("semantic_call_counts",{}),
                               "scope":"One separately instrumented diagnostic step. Nested/overlapping spans, host enqueue gaps and event overhead remain; do not sum. Semantic call counts are not actual CUDA kernel launch counts."}
    result={"case":case["case"],"status":case.get("status"),"methods":records,
            "comparisons":comparisons,"stage_diagnostics":stage_records,
            "initial_method_pairs":sum(initial.values()),"training_method_pairs":sum(training.values()),
            "inference_method_pairs":sum(record["inference_admitted"] for record in records),
            "inference_graph_method_pairs":sum(record["inference_graph_admitted"] for record in records)}
    cache=[]
    for method,item in case.get("inference_weight_cache",{}).items():
        expected_checks=("check","weight_mutation_check","restored_check","graph_changed_input_check","graph_original_check")
        admitted=(by_method.get(method,{}).get("inference_graph_admitted",False)
                  and item.get("admitted") is True and all(check_passed(item.get(name)) for name in expected_checks))
        timing=item.get("paired_timing",{})
        values={label:timing_value(timing,label) for label in ("cached","uncached","graph_cached","graph_uncached")}
        admitted=admitted and all(value is not None for value in values.values())
        cache.append({"method":method,"admitted":admitted,
                      "cold_pack_and_forward_ms":item.get("cold_pack_and_forward_ms"),
                      "packed_bytes":item.get("packed_bytes"),"paired_median_ms":values,
                      "host_uncached_over_cached":ratio(values["uncached"],values["cached"]) if admitted else None,
                      "graph_uncached_over_cached":ratio(values["graph_uncached"],values["graph_cached"]) if admitted else None,
                      "checks":{name:check_passed(item.get(name)) for name in expected_checks}})
    result["inference_weight_cache_summary"]=cache
    # Optional follow-ups stay explicitly separate from the original cohort.
    for name in ("training_graph","training_graph_checks","training_graph_errors","inference_weight_cache"):
        if name in case:
            result[name]=case[name]
    return result


def analyze(path=DEFAULT):
    path=Path(path).resolve()
    record=json.loads(path.read_text())
    if record.get("status") != "completed":
        raise ValueError("analyze a completed whole-workload record, retaining per-case failures")
    run=record["result"]
    cases=[summarize_case(case) for case in run["cases"]]
    return {"schema":1,"created_utc":datetime.now(timezone.utc).isoformat(),
            "evidence":str(path.relative_to(ROOT)),"evidence_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "runtime":run["runtime"],"contract":run["contract"],"scope":run["scope"],
            "counts":{"cases":len(cases),"method_case_pairs":sum(len(case["methods"]) for case in cases),
                      "initial_admitted":sum(case["initial_method_pairs"] for case in cases),
                      "training_admitted":sum(case["training_method_pairs"] for case in cases),
                      "inference_admitted":sum(case["inference_method_pairs"] for case in cases),
                      "inference_graph_admitted":sum(case["inference_graph_method_pairs"] for case in cases)},
            "cache_counts":{"method_case_pairs":sum(len(case["inference_weight_cache_summary"]) for case in cases),
                            "admitted":sum(item["admitted"] for case in cases for item in case["inference_weight_cache_summary"])},
            "total_training_updates_per_timed_method":run["total_training_updates_per_timed_method"],
            "admission_by_method":{
                method:{scope:sum(record.get(scope+"_admitted",False) for case in cases for record in case["methods"] if record["method"]==method)
                        for scope in ("initial","training","inference","inference_graph")}
                for method in sorted({record["method"] for case in cases for record in case["methods"]})},
            "cases":cases,
            "interpretation_limits":[
                "One complete process round; repeated samples within it are not independent campaign replications.",
                "Initial passing methods can fail after the timed training trajectory; those stored timings remain diagnostic and are excluded.",
                "Host-dispatched CUDA-event spans include enqueue gaps. Faster such spans can coexist with slower captured GPU graphs; neither alone identifies a specific scheduling cause.",
                "Graph comparison changes allocation/capture behavior as well as host dispatch; host-over-graph ratios are not pure launch-overhead measurements.",
                "Generated-native-operator and shared-Torch-surrounding controls have different reference groups; no cross-group scheduling attribution.",
                "State divergence identifies the failing outputs/gradients/parameters/moments, not a proven compiler root cause. Mixed-precision cast boundaries and optimizer sensitivity require isolated follow-up, not tolerance relaxation.",
                "Inference is independently admitted on reset immutable weights; training failure cannot be converted into a training claim by passing inference.",
                "Inference cache removes repeated weight casts, not activation casts or GEMM arithmetic. Cold pack+forward is one observation and not a reliable amortization or pure packing-time estimate; cached graphs require fixed parameters or recapture.",
                "These are explicit custom networks spanning MLP, gated MLP and a small decoder-transformer family, not six independent production models or an automatic compiler integration."]}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence",type=Path,nargs="?",default=DEFAULT)
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    result=analyze(args.evidence)
    output=args.out or HERE/("whole_summary_"+args.evidence.stem.removeprefix("whole-")+".json")
    if not output.resolve().is_relative_to(HERE):
        raise ValueError("whole summary must remain under exp/nn_round2")
    output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"output":str(output),"counts":result["counts"]},indent=2))
