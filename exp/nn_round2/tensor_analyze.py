"""Summarize tensor results without selecting winners on evaluation rounds."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics


def analyze(path):
    path = Path(path)
    record = json.loads(path.read_text())
    result = record.get("result", record)
    if record.get("status", "completed") != "completed":
        raise ValueError("only completed tensor records can be analyzed")
    rows = []
    all_checks, all_graph_checks = [], []
    for case in result["results"]:
        all_checks += case["checks"]
        all_graph_checks += case["graph_checks"]
        rounds = case["rounds"]
        holdout = rounds[1:]
        methods = case["methods"]
        comparisons = {}
        for name, baseline, candidate in (
            ("direct_vs_legacy", "oa_legacy", "oa_direct"),
            ("morton_vs_mfast", "oa_legacy", "oa_morton"),
            ("direct_morton_vs_direct", "oa_direct", "exp_direct_morton"),
            ("direct_morton_vs_morton", "oa_morton", "exp_direct_morton"),
            ("direct_fused_vs_legacy_fused", "oa_fused_relu", "exp_direct_fused_relu"),
            ("legacy_early_vs_legacy", "oa_legacy", "exp_legacy_early_advance"),
            ("direct_early_vs_direct", "oa_direct", "exp_direct_early_advance"),
        ):
            if all(baseline in item["median_ms"] and candidate in item["median_ms"] for item in holdout) and holdout:
                ratios = [item["median_ms"][baseline] / item["median_ms"][candidate] for item in holdout]
                comparisons[name] = {"baseline": baseline, "candidate": candidate,
                                     "holdout_round_ratios": ratios, "median_ratio": statistics.median(ratios),
                                     "greater_than_2pct_in_every_holdout_round": all(ratio > 1.02 for ratio in ratios)}
        selections = {}
        for epilogue in sorted({method["epilogue"] for method in methods.values()}):
            selectable = {kind: [name for name, method in methods.items() if method["kind"] == kind
                                  and method["epilogue"] == epilogue and name in rounds[0]["median_ms"]]
                          for kind in ("native_oa", "same_ptx_ptxas", "vendor_library")}
            selected = {kind: min(names, key=rounds[0]["median_ms"].get) for kind, names in selectable.items() if names}
            values = {kind: [item["median_ms"][name] for item in holdout] for kind, name in selected.items()}
            ratios = {}
            if "native_oa" in selected:
                for kind in ("same_ptx_ptxas", "vendor_library"):
                    if kind in selected:
                        ratios[kind + "_over_native"] = [left / right for left, right in zip(values[kind], values["native_oa"])]
            selections[epilogue] = {"selection_round": 0, "fixed_methods": selected, "holdout_ms": values,
                                    "holdout_round_ratios": ratios}
        rows.append({"name": case["name"], "shape": case["shape"], "dtype": case["dtype"],
                     "comparisons": comparisons, "selection": selections,
                     "all_full_checks_passed": all(check["passed"] for check in case["checks"]),
                     "all_graph_checks_passed": all(check["passed"] for check in case["graph_checks"])})
    return {"schema": 1, "source_record": str(path), "source_record_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "device": result["device"], "cases": len(rows), "full_checks": len(all_checks),
            "full_checks_passed": sum(check["passed"] for check in all_checks),
            "full_output_elements_checked": sum(check["elements"] for check in all_checks),
            "graph_checks": len(all_graph_checks), "graph_checks_passed": sum(check["passed"] for check in all_graph_checks),
            "graph_output_elements_checked": sum(check["elements"] for check in all_graph_checks),
            "worst_relative_l2_vs_fp64": max(check["relative_l2"] for check in all_checks),
            "worst_absolute_error_vs_fp64": max(check["max_abs_error"] for check in all_checks),
            "all_inputs_unchanged": all(check["inputs_unchanged"] for check in all_checks + all_graph_checks),
            "interpretation": "ratios>1 favor native/candidate; tune once on round0, evaluate fixed methods on later within-process rounds; no independent-process replication or exhaustive CUDA tuning implied",
            "results": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = analyze(args.record)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.write_text(rendered)
        print(args.out)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
