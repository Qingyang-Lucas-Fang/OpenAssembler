"""Regression checks that failed whole-state gates never become speedups."""
from exp.nn_round2.whole_analysis import analyze, summarize_case


def _case():
    summary={"median_ms":1.,"samples_ms":[1.,1.,1.]}
    methods=("oa_direct","ptxas")
    return {"case":{"name":"synthetic_admission_test"},"status":"completed",
            "verification":{m:{"passed":True,"reference_method":"ptxas","state":{"passed":True},"guards":{"passed":True,"gemm_checks":[{"passed":True}]}} for m in methods},
            "post_timing_verification":{m:{"passed":True} for m in methods},
            "admitted_training_methods":list(methods),
            "training":{m:{"admitted":True,"cuda_event":summary,"host_wall":summary} for m in methods},
            "inference":{m:{"admitted":True,"event":summary,"graph":summary,
                            "check":{"passed":True,"graph":{"passed":True},"graph_changed_input":{"passed":True}}} for m in methods}}


def test_failed_post_trajectory_state_overrules_stored_timing_and_admitted_flag():
    case=_case()
    case["post_timing_verification"]["oa_direct"]={"passed":False,"details":[{"name":"gradient/raw/w","elements":16,"passed":False}]}
    result=summarize_case(case)
    direct=next(row for row in result["methods"] if row["method"]=="oa_direct")
    assert direct["training_event_ms"]==1.  # Diagnostic evidence retained.
    assert direct["training_admitted"] is False
    assert direct["reference_over_method_training"] is None
    assert direct["post_trajectory_failure"]["by_category"]=={"gradient":1}


def test_missing_changed_input_graph_gate_rejects_graph_but_preserves_valid_eager():
    case=_case()
    del case["inference"]["oa_direct"]["check"]["graph_changed_input"]
    result=summarize_case(case)
    direct=next(row for row in result["methods"] if row["method"]=="oa_direct")
    assert direct["inference_admitted"] is True
    assert direct["inference_graph_admitted"] is False
    assert direct["reference_over_method_inference_graph"] is None


def test_initial_failure_rejects_every_scope_and_reference_failure_rejects_pair():
    case=_case()
    case["verification"]["ptxas"]["passed"]=False
    case["verification"]["ptxas"]["guards"]["passed"]=False
    result=summarize_case(case)
    direct=next(row for row in result["methods"] if row["method"]=="oa_direct")
    assert direct["training_admitted"] is False
    assert direct["inference_admitted"] is False
    assert direct["inference_graph_admitted"] is False


def test_independent_inference_can_pass_when_training_state_failed():
    case=_case()
    case["verification"]["oa_direct"]["passed"]=False
    result=summarize_case(case)
    direct=next(row for row in result["methods"] if row["method"]=="oa_direct")
    assert direct["initial_admitted"] is False and direct["training_admitted"] is False
    assert direct["inference_admitted"] is True and direct["inference_graph_admitted"] is True


def test_frozen_whole_record_retains_all_failures_and_correct_reference_groups():
    result=analyze()
    assert result["counts"]=={"cases":6,"method_case_pairs":36,"initial_admitted":24,
                             "training_admitted":21,"inference_admitted":24,"inference_graph_admitted":24}
    assert result["admission_by_method"]["oa_tensor_only"]["training"]==6
    assert result["admission_by_method"]["oa_direct"]["training"]==1
    assert result["admission_by_method"]["oa_legacy"]["training"]==1
    for case in result["cases"]:
        tensor=next(row for row in case["methods"] if row["method"]=="oa_tensor_only")
        assert tensor["reference_method"]=="ptxas_tensor_only"
        assert tensor["inference_graph_admitted"]
        assert tensor["reference_over_method_inference_graph"]<1
