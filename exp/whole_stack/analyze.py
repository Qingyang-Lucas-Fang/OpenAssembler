"""Derive whole-stack timing and profiler summaries without changing raw evidence.

The original profiler tables include GPU user annotations alongside CUDA
kernels. This analyzer excludes those annotations, separates memory operations,
and quarantines unfamiliar event names. It never sums CPU operator attribution
with device events. Fused-kernel categories describe names, not exact semantic
fractions of the fused operations.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


METHODS = ("eager", "openassembler_cli_compiled")
ANNOTATION_PREFIXES = ("stage/", "Optimizer.")
KERNEL_PREFIXES = (
    "void ", "ampere_", "cutlass_", "fmha_", "triton_", "sm80_",
    "cublasLt::", "flash_", "sgemm_", "hgemm_", "s1688gemm_", "s16816gemm_",
)
MEMORY_PREFIXES = ("Memcpy", "Memset", "[CUDA memcpy", "[CUDA memset")
CATEGORY_DESCRIPTIONS = {
    "gemm": "GEMM/GEMV and split-K reduction kernel names; not individual HMMA instruction time",
    "fused_attention": "Fused attention kernels; softmax and attention matmuls remain inseparable",
    "softmax_loss": "Softmax/log-softmax/loss kernels, including fused forms",
    "gelu_activation": "GELU forward/backward kernels, including fused forms",
    "norm_residual_embedding_fused": "Triton names mentioning norm/rsqrt/embedding; fused contents are not split",
    "embedding": "Embedding and index-select kernels",
    "optimizer_or_grad_clip_multi_tensor": "Multi-tensor kernels used by AdamW or gradient clipping",
    "reduction": "Named standalone reductions and norms",
    "pointwise": "Named elementwise/fill/arange kernels",
    "copy_layout_kernel": "CUDA kernels implementing copies, cat, or layout changes; distinct from DMA events",
    "generic_triton_mixed": "Other Triton fused/reduction kernels; semantic assignment intentionally unresolved",
    "other_recognized_kernel": "Recognized real-kernel name family without a narrower assignment",
}


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def distribution(values):
    clean = [float(value) for value in values if numeric(value)]
    if not clean:
        return None
    return {"n": len(clean), "median": statistics.median(clean), "min": min(clean),
            "max": max(clean), "values": clean}


def category(name):
    lower = name.lower()
    if re.search(r"gemm|gemv|matmul|splitkreduce", lower):
        return "gemm"
    if re.search(r"fmha|pytorch_flash::flash_|flash.*attention|attention.*kernel", lower):
        return "fused_attention"
    if re.search(r"softmax|nll_loss|cross_entropy", lower):
        return "softmax_loss"
    if "gelu" in lower:
        return "gelu_activation"
    if name.startswith("triton_"):
        if any(token in lower for token in ("embedding", "rsqrt", "layer_norm", "rmsnorm")):
            return "norm_residual_embedding_fused"
        return "generic_triton_mixed"
    if "embedding" in lower or "indexselect" in lower:
        return "embedding"
    if "multi_tensor_apply_kernel" in lower or "adam" in lower:
        return "optimizer_or_grad_clip_multi_tensor"
    if any(token in lower for token in ("direct_copy", "catarray", "copy_kernel", "transpose")):
        return "copy_layout_kernel"
    if any(token in lower for token in ("reduce_kernel", "reduction", "lpnorm", "normtwo", "sum_kernel")):
        return "reduction"
    if any(token in lower for token in ("elementwise_kernel", "fill_kernel", "arange")):
        return "pointwise"
    return "other_recognized_kernel"


def filtered_profiler(raw):
    if not isinstance(raw, dict):
        return None
    kernels, memory, annotations, unknown = [], [], [], []
    for event in raw.get("cuda_kernels", []):
        name = event.get("name", "")
        if not isinstance(name, str) or not numeric(event.get("cuda_us")):
            unknown.append(event)
            continue
        row = {"name": name, "count": event.get("count", 0), "cuda_us": event["cuda_us"]}
        if event.get("is_user_annotation") or name.startswith(ANNOTATION_PREFIXES):
            annotations.append(row)
        elif name.startswith(MEMORY_PREFIXES):
            memory.append(row)
        elif name.startswith(KERNEL_PREFIXES):
            row["category"] = category(name)
            kernels.append(row)
        else:
            unknown.append(row)
    by_category = defaultdict(lambda: {"cuda_us": 0.0, "launches": 0, "kernel_names": []})
    for event in kernels:
        item = by_category[event["category"]]
        item["cuda_us"] += event["cuda_us"]
        item["launches"] += event["count"]
        item["kernel_names"].append(event["name"])
    kernel_us = sum(event["cuda_us"] for event in kernels)
    for item in by_category.values():
        item["fraction_of_recognized_kernel_time"] = item["cuda_us"] / kernel_us if kernel_us else None
    total_raw = sum(event["cuda_us"] for event in raw.get("cuda_kernels", [])
                    if numeric(event.get("cuda_us")))
    # These checks guard the correction itself, not benchmark correctness.
    assert not any(event["name"].startswith(ANNOTATION_PREFIXES) for event in kernels)
    assert not any(event["name"].startswith(MEMORY_PREFIXES) for event in kernels)
    partition_us = kernel_us + sum(event["cuda_us"] for event in memory + annotations + unknown
                                   if numeric(event.get("cuda_us")))
    assert math.isclose(total_raw, partition_us, rel_tol=1e-10, abs_tol=1e-6)
    return {
        "recognized_cuda_kernel_time_us": kernel_us,
        "recognized_cuda_kernel_launches": sum(event["count"] for event in kernels),
        "categories": dict(sorted(by_category.items(), key=lambda pair: pair[1]["cuda_us"], reverse=True)),
        "recognized_kernels": sorted(kernels, key=lambda event: event["cuda_us"], reverse=True),
        "memory_operations": memory,
        "memory_operation_time_us": sum(event["cuda_us"] for event in memory),
        "memory_operation_count": sum(event["count"] for event in memory),
        "excluded_user_annotations": annotations,
        "excluded_annotation_time_us": sum(event["cuda_us"] for event in annotations),
        "quarantined_unknown_events": unknown,
        "raw_cuda_event_time_us": total_raw,
        "partition_selfcheck_passed": True,
        "cpu_operator_attribution": raw.get("cpu_operator_attribution", []),
        "scope": "post-timing diagnostic; kernel durations may overlap across streams; CPU operator attribution is separate and must not be added; category labels preserve fused groups",
    }


def decode_check(row):
    case = row.get("case", {})
    if case.get("mode") != "infer":
        return None
    ids = {method: row.get(method, {}).get("timing", {}).get("decode_final_token_ids") for method in METHODS}
    expected_length = case.get("seq")
    shapes_valid = all(
        isinstance(value, list) and len(value) == 1 and isinstance(value[0], list)
        and len(value[0]) == expected_length and bool(value[0])
        and all(isinstance(token, int) and not isinstance(token, bool)
                    and 0 <= token < case.get("vocab", 0) for token in value[0])
        for value in ids.values()
    )
    decode_lengths = [row.get(method, {}).get("timing", {}).get("decode_tokens") for method in METHODS]
    covers_all_generated = all(isinstance(length, int) and 0 < length <= expected_length
                               for length in decode_lengths) and len(set(decode_lengths)) == 1
    exact = shapes_valid and ids[METHODS[0]] == ids[METHODS[1]]
    return {
        "ids_present_and_shape_valid": shapes_valid,
        "all_generated_tokens_retained": covers_all_generated,
        "exact_token_ids_equal": exact,
        "passed": shapes_valid and covers_all_generated and exact,
        "missing_ids_cannot_pass": True,
        "expected_final_context_shape": [1, expected_length],
        "decode_tokens": decode_lengths,
        "final_context_sha256": {
            method: hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()
            if value is not None else None for method, value in ids.items()
        },
    }


def method_summary(row, method):
    record = row.get(method, {})
    timing = record.get("timing", {})
    case = row.get("case", {})
    fields = {
        "worker_wall_s": record.get("worker_wall_s"),
        "program_execution_s_including_compile_and_profile": record.get("report", {}).get("seconds"),
        "peak_allocated_bytes_including_profile": timing.get("peak_allocated_bytes"),
        "final_gradients_available": timing.get("final_gradients_available"),
        "tf32": timing.get("tf32"),
        "float32_matmul_precision": timing.get("float32_matmul_precision"),
        "parameters": timing.get("parameters"),
        "optimizer_steps_including_warmup_and_profile": record.get("report", {}).get("optimizer_steps"),
    }
    if case.get("mode") == "train":
        stages = timing.get("stages", {})
        fields["timing_medians_ms"] = {
            key: stages[key].get("median_ms") for key in ("batch", "forward_loss", "backward", "optimizer", "end_to_end")
            if isinstance(stages.get(key), dict)
        }
        fields["timing_samples_ms"] = {
            key: stages[key].get("samples_ms", []) for key in fields["timing_medians_ms"]
        }
        fields["tokens_per_second"] = stages.get("tokens_per_second")
        fields["profiles"] = {"training_step": filtered_profiler(stages.get("profiler"))}
    else:
        fields["timing_medians_ms"] = {
            key: timing.get(key, {}).get("median_ms") for key in ("prefill", "fixed_window_recompute_decode")
        }
        fields["timing_samples_ms"] = {
            key: timing.get(key, {}).get("samples_ms", []) for key in fields["timing_medians_ms"]
        }
        fields["prefill_tokens_per_second"] = timing.get("prefill_tokens_per_second")
        fields["decode_tokens_per_second"] = timing.get("decode_tokens_per_second")
        fields["profiles"] = {"prefill": filtered_profiler(timing.get("prefill_profiler")),
                              "decode": filtered_profiler(timing.get("decode_profiler"))}
    return fields


def analyze(raw, source_path, expected_cases=10, expected_rounds=2):
    per_round = []
    groups = defaultdict(list)
    for row in raw.get("cases", []):
        case = row.get("case", {})
        record = {"name": case.get("name"), "round": row.get("round", 1), "case": case,
                  "verification": row.get("verification", {}), "decode_check": decode_check(row),
                  "precision": row.get("precision"), "source_sha256": row.get("source_sha256"),
                  "process_order": row.get("process_order"),
                  "methods": {method: method_summary(row, method) for method in METHODS}}
        gradients_checked = all(record["methods"][method].get("final_gradients_available", 0) > 0
                                for method in METHODS) if case.get("mode") == "train" else None
        record["final_gradient_presence_passed"] = gradients_checked
        record["eligible_for_performance_claim"] = (
            record["verification"].get("status") == "ok"
            and (record["decode_check"] is None or record["decode_check"]["passed"])
            and gradients_checked is not False
        )
        ratios = {}
        for metric, eager_value in record["methods"][METHODS[0]]["timing_medians_ms"].items():
            compiled_value = record["methods"][METHODS[1]]["timing_medians_ms"].get(metric)
            if numeric(eager_value) and numeric(compiled_value) and compiled_value > 0:
                ratios[metric] = eager_value / compiled_value
        record["eager_over_compiled_time_ratio"] = ratios
        per_round.append(record)
        groups[case.get("name")].append(record)
    workloads = []
    for name, rows in groups.items():
        eligible = [row for row in rows if row["eligible_for_performance_claim"]]
        methods = {}
        for method in METHODS:
            metric_names = sorted(set().union(*(row["methods"][method]["timing_medians_ms"] for row in rows)))
            rate_names = ("tokens_per_second",) if rows[0]["case"].get("mode") == "train" else (
                "prefill_tokens_per_second", "decode_tokens_per_second")
            profile_names = sorted(set().union(*(row["methods"][method]["profiles"] for row in rows)))
            profile_aggregates = {}
            for profile_name in profile_names:
                profiles = [row["methods"][method]["profiles"].get(profile_name) for row in eligible]
                profiles = [profile for profile in profiles if profile is not None]
                category_names = sorted(set().union(*(profile["categories"] for profile in profiles)))
                profile_aggregates[profile_name] = {
                    "verified_round_recognized_kernel_time_us": distribution([
                        profile["recognized_cuda_kernel_time_us"] for profile in profiles]),
                    "verified_round_recognized_kernel_launches": distribution([
                        profile["recognized_cuda_kernel_launches"] for profile in profiles]),
                    "verified_round_memory_operation_time_us": distribution([
                        profile["memory_operation_time_us"] for profile in profiles]),
                    "verified_round_category_time_us": {
                        key: distribution([profile["categories"].get(key, {}).get("cuda_us", 0.0)
                                           for profile in profiles]) for key in category_names
                    },
                    "verified_round_category_launches": {
                        key: distribution([profile["categories"].get(key, {}).get("launches", 0)
                                           for profile in profiles]) for key in category_names
                    },
                    "unknown_events_in_verified_rounds": sum(
                        len(profile["quarantined_unknown_events"]) for profile in profiles),
                }
            methods[method] = {
                "verified_round_median_ms": {
                    metric: distribution([row["methods"][method]["timing_medians_ms"].get(metric) for row in eligible])
                    for metric in metric_names
                },
                "all_round_median_ms_diagnostic": {
                    metric: distribution([row["methods"][method]["timing_medians_ms"].get(metric) for row in rows])
                    for metric in metric_names
                },
                "worker_wall_s": distribution([row["methods"][method].get("worker_wall_s") for row in rows]),
                "program_execution_s_including_compile_and_profile": distribution([
                    row["methods"][method].get("program_execution_s_including_compile_and_profile") for row in rows]),
                "peak_allocated_bytes_including_profile": distribution([
                    row["methods"][method].get("peak_allocated_bytes_including_profile") for row in rows]),
                "verified_throughput": {
                    key: distribution([row["methods"][method].get(key) for row in eligible]) for key in rate_names
                },
                "profile_aggregates": profile_aggregates,
            }
        metric_names = sorted(set().union(*(row["eager_over_compiled_time_ratio"] for row in rows)))
        round_ids = [row["round"] for row in rows]
        rounds_complete = len(round_ids) == expected_rounds and set(round_ids) == set(range(1, expected_rounds + 1))
        workloads.append({"name": name, "case": rows[0]["case"], "rounds_recorded": len(rows),
                          "verified_rounds": len(eligible),
                          "unique_expected_rounds_present": rounds_complete,
                          "complete_and_verified": len(eligible) == expected_rounds and rounds_complete,
                          "round_statuses": [{"round": row["round"],
                                               "verification": row["verification"],
                                               "decode_check": row["decode_check"],
                                               "eligible": row["eligible_for_performance_claim"]} for row in rows],
                          "methods": methods,
                          "verified_eager_over_compiled_time_ratio": {
                              metric: distribution([row["eager_over_compiled_time_ratio"].get(metric) for row in eligible])
                              for metric in metric_names}})
    bad_rounds = [row for row in per_round if not row["eligible_for_performance_claim"]]
    all_complete = len(workloads) == expected_cases and all(row["complete_and_verified"] for row in workloads)
    return {
        "schema": 1, "derived_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_input": str(source_path.resolve()),
        "raw_input_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "raw_status": raw.get("status"), "runtime": raw.get("runtime"),
        "expected_workload_count": expected_cases, "expected_rounds_per_workload": expected_rounds,
        "workload_count": len(workloads), "round_count": len(per_round),
        "failed_or_incomplete_verification_round_count": len(bad_rounds),
        "execution_errors": raw.get("errors", []),
        "all_expected_workloads_complete_and_verified": all_complete and not raw.get("errors"),
        "compiler_attribution": raw.get("compiler_attribution"),
        "baseline_attribution": raw.get("baseline_attribution"),
        "ratio_definition": "eager time / compiled time; greater than one means compiled frontend worker is faster",
        "aggregation": "median and range of within-round medians; paired ratios computed within each round; failed numerical rounds remain diagnostic only",
        "annotation_exclusion_prefixes": list(ANNOTATION_PREFIXES),
        "recognized_kernel_prefixes": list(KERNEL_PREFIXES),
        "category_descriptions": CATEGORY_DESCRIPTIONS,
        "scope_limits": [
            "Whole-model compiled path is TorchInductor/Triton managed by the OA frontend, not native OA SASS.",
            "CUDA-event resident-data step timing excludes initialization, compile, zero_grad at step start, and final readback; may include host enqueue gaps.",
            "Program and worker wall time include compilation, profiling, and snapshots and are not isolated compile-time measurements.",
            "Final training state and clipped gradients are checked after one additional post-timing diagnostic optimizer step.",
            "Inference decode recomputes a fixed window, batch one, without a KV cache.",
            "Profiler categories are based on kernel names; fused normalization/activation/residual boundaries are not independently measurable here.",
            "CUDA event durations may overlap across streams; kernel-time sums are diagnostic work totals, not end-to-end latency.",
            "Peak memory includes the separate profiler pass.",
            "Historical annotation correction is derived from explicit name families because original raw data did not retain is_user_annotation.",
        ],
        "workloads": workloads, "per_round": per_round,
    }


def selfcheck():
    raw = {"cuda_kernels": [
        {"name": "stage/forward_loss", "count": 1, "cuda_us": 1000.0},
        {"name": "Optimizer.step#AdamW.step", "count": 1, "cuda_us": 200.0},
        {"name": "triton_per_fused_add_mean_mul_pow_rsqrt_2", "count": 3, "cuda_us": 12.0},
        {"name": "ampere_sgemm_32x32_tn", "count": 2, "cuda_us": 30.0},
        {"name": "Memcpy DtoD (Device -> Device)", "count": 1, "cuda_us": 4.0},
        {"name": "unknown_annotation_or_kernel", "count": 1, "cuda_us": 50.0},
    ]}
    result = filtered_profiler(raw)
    assert result["recognized_cuda_kernel_time_us"] == 42.0
    assert result["recognized_cuda_kernel_launches"] == 5
    assert result["memory_operation_time_us"] == 4.0
    assert result["excluded_annotation_time_us"] == 1200.0
    assert len(result["quarantined_unknown_events"]) == 1
    missing = decode_check({"case": {"mode": "infer", "seq": 128, "vocab": 512}})
    assert not missing["passed"] and not missing["exact_token_ids_equal"]
    return {"annotation_partition": "passed", "missing_decode_ids_rejected": "passed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path, nargs="?")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--expected-cases", type=int, default=10)
    parser.add_argument("--expected-rounds", type=int, default=2)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    checked = selfcheck()
    if args.raw is None:
        if not args.selfcheck:
            parser.error("raw input is required unless --selfcheck is used")
        print(json.dumps(checked))
        return
    content = args.raw.read_bytes()
    raw = json.loads(content)
    result = analyze(raw, args.raw, args.expected_cases, args.expected_rounds)
    result["raw_input_sha256"] = hashlib.sha256(content).hexdigest()
    result["selfcheck"] = checked
    output = args.out or args.raw.with_name("summary-" + args.raw.name)
    if output.resolve() == args.raw.resolve():
        parser.error("derived output must not replace raw input")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(output.resolve()), "workloads": result["workload_count"],
                      "rounds": result["round_count"],
                      "failed_rounds": result["failed_or_incomplete_verification_round_count"],
                      "all_complete_verified": result["all_expected_workloads_complete_and_verified"]}))


if __name__ == "__main__":
    main()
