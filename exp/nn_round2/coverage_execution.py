"""Attach actual campaign coverage to the prospective 23-family arithmetic map.

Does not change prospective missing-implementation entries into passing results.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from exp.nn_round2.whole_analysis import summarize_case


ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/"exp/nn_round2"
DEFAULT_OPERATORS=HERE/"runs/operators-20261107T143128.366417Z.json"
DEFAULT_TENSOR=HERE/"runs/tensor-20261107T143201.758879Z.json"
DEFAULT_WHOLE=HERE/"runs/whole-20261107T143549.109468Z.json"
FAMILIES={
    "dense_linear": (["tensor","whole"],[],"isolated_native_and_hybrid_gemm_validated","18 isolated shape/precision cases and six custom networks cover explicit full-tile forward/dX/dW GEMMs. Arbitrary alpha/beta, strides and tails remain absent."),
    "batched_attention_products": (["whole"],[],"hybrid_gemm_composition_validated","Three transformer configurations include per-head QK/PV products and backward. The admitted OA tensor-only route uses shared Torch softmax/masking; this is not a fused native SDPA kernel."),
    "residual_bias_scale": (["operators","tensor","whole"],["residual_add","gated_multiply"],"isolated_native_and_shared_whole_coverage","Residual add and gated multiply are isolated native functions. Tensor candidates cover bias/residual/ReLU. Whole bias/residual/scale are shared CUDA operations."),
    "relu": (["operators","whole"],["relu","relu_backward"],"isolated_validated_partial_whole_admission","Forward/backward isolated functions pass. Native-all-operator wide MLP passes post-trajectory gate; small native-all-operator MLP fails it. Tensor-only routes retain shared Torch ReLU."),
    "gelu": (["operators","whole"],["gelu_tanh","gelu_tanh_backward"],"forward_validated_backward_has_failed_gates","GELU forward passes isolated gates. Ten backward configurations fail the declared mathematical gate in both OA and ptxas; their timings are excluded. Native-all-operator transformer routes fail whole-state gates."),
    "silu_gate": (["operators","whole"],["silu","silu_backward","sigmoid","gated_multiply"],"isolated_validated_whole_native_excluded","Isolated expressions pass. Native-all-operator gated network fails initial output gate; the OA tensor-only variant with shared Torch surrounding operations passes whole-state gates."),
    "softmax": (["operators","whole"],["softmax","softmax_backward"],"isolated_nonfused_forward_backward_validated","Complete finite FP32 rows and derivatives are tested. Native-all-operator transformer state admission fails; tensor-only models use shared Torch softmax."),
    "layernorm": (["operators"],["layernorm","layernorm_backward"],"isolated_nonaffine_forward_dx_validated","Only non-affine output and dX are covered; no native saved-statistics API or dGamma/dBeta implementation, and this whole cohort uses RMSNorm instead."),
    "rmsnorm": (["operators","whole"],["rmsnorm","rmsnorm_backward"],"isolated_nonaffine_forward_dx_validated_whole_gap","Isolated output/dX pass. Native-all-operator model routes generally fail state gates; successful tensor-only models share Torch RMSNorm and affine gradients. Failures do not isolate RMSNorm as the root cause."),
    "bias_gradient": (["whole"],[],"shared_cuda_only","Bias-gradient reductions execute through shared Torch autograd in all whole methods; no new native reduction implementation."),
    "loss_mse": (["whole"],[],"shared_cuda_only","MLPs include shared Torch MSE and its full derivative inside the whole timing boundary."),
    "loss_cross_entropy": (["operators","whole"],["cross_entropy"],"restricted_native_isolated_shared_whole","Native isolated kernel uses fixed label zero and per-row loss plus unaveraged gradient. Whole transformer mean cross-entropy/labels/backward remain shared Torch."),
    "gradient_clipping": (["whole"],[],"shared_cuda_only","Fixed global-norm clipping is timed; raw and clipped gradients are checked separately. No native clipping implementation is claimed."),
    "adamw": (["operators","whole"],["adam_like"],"restricted_adam_like_isolated_true_adamw_shared","Native isolated Adam-like update lacks bias correction/decay. True AdamW lr1e-4, eps1e-8, decay.01, moments and step counters execute through shared Torch and are checked in whole workloads."),
    "sgd_momentum": (["operators"],["sgd_momentum"],"isolated_restricted_update_validated","Isolated momentum/parameter update only, with no Nesterov, dampening, decay or full optimizer integration claim."),
    "elementary_math": (["operators"],["divide","rsqrt_scale","sigmoid","tanh_backward","gelu_tanh","adam_like"],"isolated_domains_and_composed_expressions_validated","Division/rsqrt and composed exp/tanh/sqrt expressions have bounded finite inputs and declared tolerances. No exhaustive exceptional-value/precision-contract claim."),
    "integer_addressing": (["operators","tensor"],["div_const7","rem_const7"],"isolated_uint32_constant7_validated","Exact unsigned quotient/remainder for divisor7 includes structured boundary patterns. This round does not exhaust every divisor, signed mode or address expression."),
    "layout_and_cast": (["whole"],[],"shared_cuda_costs_included","FP16 operand casts, transposed contiguous packing and head split/merge are present in whole latency. New native packing/cast implementations are absent."),
    "masking_position": (["operators","whole"],["rope_pairs"],"restricted_rotation_isolated_shared_whole_masks","Isolated native paired rotations use supplied cosine/sine arrays. Whole models use learned positions and shared CUDA finite causal masking; no integrated native RoPE/position convention claim."),
    "embedding_gather_scatter": (["whole"],[],"shared_cuda_only","Whole transformer embeddings/position gradients execute through shared Torch; no native gather/scatter atomic implementation."),
    "dropout_rng": (["operators"],["dropout_mask_scale"],"supplied_mask_only_rng_missing","Native test multiplies an externally supplied mask and fixed scale. No RNG, dropout layer or saved-mask training integration; whole models disable dropout."),
    "convolution_pooling": ([],[],"not_covered","No CNN convolution/pooling tests or native implementation in this round."),
    "distributed_collectives": ([],[],"not_covered","Single-GPU only; no actual multi-GPU collective/communication overlap experiment.")
}


def build(operators_path=DEFAULT_OPERATORS,tensor_path=DEFAULT_TENSOR,whole_path=DEFAULT_WHOLE):
    paths={"operators":Path(operators_path).resolve(),"tensor":Path(tensor_path).resolve(),"whole":Path(whole_path).resolve()}
    records={key:json.loads(path.read_text()) for key,path in paths.items()}
    if any(record.get("status")!="completed" for record in records.values()):
        raise ValueError("coverage execution requires completed records, retaining internal gate failures")
    operators=records["operators"]["result"]["cases"]
    tensor=records["tensor"]["result"]["results"]
    whole=[summarize_case(case) for case in records["whole"]["result"]["cases"]]
    categories=[]
    for planned in json.loads((HERE/"coverage.json").read_text())["categories"]:
        key=planned["id"]
        scopes,ops,status,description=FAMILIES[key]
        selected=[row for row in operators if row["op"] in ops]
        categories.append({"id":key,"execution_status":status,"finding":description,
                           "evidence_scopes":scopes,"operator_names":ops,
                           "isolated_operator_case_counts":dict(Counter(row["status"] for row in selected)),
                           "scope_note":"Kernel configurations are not independent model replications; shared-runtime coverage is not native OA implementation."})
    followups=[]
    followup_paths=(
        ("local_fma_initial","runs/operator-fma-20261107T143537.650425Z.json"),
        ("local_fma_independent_confirmation","runs/operator-fma-confirm-20261107T144005.532354Z.json"),
        ("stable_gelu_backward","runs/operator-gelu-stable-20261107T143841.822588Z.json"),
        ("inference_weight_cache_80gb","runs/whole-20261107T144210.754688Z.json"),
        ("early_pointer_fp16","runs/tensor-followup-20261107T143807.885774Z.json"),
        ("early_pointer_bf16","runs/tensor-followup-20261107T144047.961466Z.json"),
        ("diagnostic_counters_recovered","counter_recovered_20261107T143821.759059Z.json"),
        ("diagnostic_fma_counters","runs/counters-20261107T144534.159647Z.json"),
    )
    for kind,relative in followup_paths:
        path=HERE/relative
        if not path.exists():
            continue
        record=json.loads(path.read_text())
        data=record.get("result",record)
        item={"kind":kind,"path":str(path.relative_to(ROOT)),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
              "status":record.get("status","derived_counter_recovery"),
              "scope":"Separate experiment/control/shape record; counts are not pooled into independent-replication totals."}
        if "cases" in data and kind not in ("inference_weight_cache_80gb",):
            item["configurations"]=len(data["cases"])
            item["configuration_status"]=dict(Counter(row.get("status","not_recorded") for row in data["cases"]))
        if kind=="inference_weight_cache_80gb":
            case_summaries=[summarize_case(case) for case in data["cases"]]
            item["workloads"]=len(case_summaries)
            item["inference_cache_method_pairs"]=sum(len(case["inference_weight_cache_summary"]) for case in case_summaries)
            item["admitted_inference_cache_pairs"]=sum(row["admitted"] for case in case_summaries for row in case["inference_weight_cache_summary"])
            item["device"]=data["runtime"]
        if kind.startswith("early_pointer"):
            item["tensor_cases"]=len(data.get("results",[]))
            item["full_pattern_checks"]=sum(len(row["checks"]) for row in data.get("results",[]))
            item["graph_checks"]=sum(len(row["graph_checks"]) for row in data.get("results",[]))
        if "profiles" in data:
            item["diagnostic_profiles"]=len(data["profiles"])
        followups.append(item)
    return {"schema":1,"created_utc":datetime.now(timezone.utc).isoformat(),
            "plan":"exp/nn_round2/coverage.json","prospective_plan_status":"planned",
            "evidence":{key:{"path":str(path.relative_to(ROOT)),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()} for key,path in paths.items()},
            "counts":{"arithmetic_families":len(categories),"isolated_named_operations":len({row["op"] for row in operators}),
                      "isolated_configurations":len(operators),"isolated_configuration_status":dict(Counter(row["status"] for row in operators)),
                      "tensor_shape_precision_cases":len(tensor),"tensor_pattern_method_checks":sum(len(row["checks"]) for row in tensor),
                      "tensor_graph_checks":sum(len(row["graph_checks"]) for row in tensor),
                      "whole_configurations":len(whole),"whole_method_pairs":sum(len(row["methods"]) for row in whole),
                      "whole_training_admitted":sum(row["training_method_pairs"] for row in whole)},
            "categories":categories,
            "separate_followup_records":followups,
            "followup_note":"Original GEMM/whole/operator failures remain recorded above. Follow-up numerical fixes, local FMA and inference caching have their own restricted contracts and dossiers; a passing follow-up does not retroactively admit an original failure.",
            "limit":"This is a current coverage map, not evidence that every NN arithmetic family is implemented or universally faster. Native, shared-runtime and uncovered categories stay distinct."}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operators",type=Path,default=DEFAULT_OPERATORS)
    parser.add_argument("--tensor",type=Path,default=DEFAULT_TENSOR)
    parser.add_argument("--whole",type=Path,default=DEFAULT_WHOLE)
    parser.add_argument("--out",type=Path,default=HERE/"coverage_execution_20261107.json")
    args=parser.parse_args()
    if not args.out.resolve().is_relative_to(HERE):
        raise ValueError("coverage output must remain under exp/nn_round2")
    result=build(args.operators,args.tensor,args.whole)
    args.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"output":str(args.out),"counts":result["counts"]},indent=2))
