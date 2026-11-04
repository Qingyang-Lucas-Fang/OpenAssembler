"""Recompute the epilogue result summary from preserved hardware-run records."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE = HERE.parent
RUN = BASE / "runs" / "20261103T232923.792831Z"


def summarize(path: Path) -> dict:
    records = json.loads(path.read_text())
    if len(records) != 1 or records[0]["worker_rc"] != 0:
        raise ValueError(f"Expected one passing hardware result: {path}")
    result = records[0]
    dtype = result["dtype"]
    row_name = f"p3_epilogue_row_native_4096_4096_4096_{dtype}"
    times = result["milliseconds"]
    baseline, row = times["ours"], times[row_name]
    checks = [check for check in result["checks"]
              if check["method"] in ("ours", row_name)]
    provenance = result["input_provenance"]
    alternative = next(item for item in provenance["alternatives"]
                       if item["compilation"]["variant"] == "row_native")
    return {
        "artifact": str(path.relative_to(BASE)),
        "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "recorded_utc": result["recorded_utc"],
        "device": result["device"],
        "shape_mnk": result["shape_mnk"],
        "dtype": dtype,
        "output_dtype": result["output_dtype"],
        "accumulation": result["accumulation"],
        "baseline_ms": baseline,
        "pair_packed_ms": times[f"p3_epilogue_pair_packed_4096_4096_4096_{dtype}"],
        "pair_native_ms": times[f"p3_epilogue_pair_native_4096_4096_4096_{dtype}"],
        "row_native_ms": row,
        "row_speedup": baseline / row,
        "row_time_reduction_percent": 100 * (1 - row / baseline),
        "row_tflops": result["tflops"][row_name],
        "cublas_ms": times["cublas"],
        "row_percent_cublas_throughput": 100 * times["cublas"] / row,
        "samples_per_method": len(result["samples_ms"]["ours"]),
        "graph_nodes": result["graph_nodes"],
        "checks_for_baseline_and_row": len(checks),
        "full_output_elements_per_check": min(check["full_output_elements_checked"]
                                              for check in checks),
        "max_full_output_relative_l2_vs_cublas": max(
            check["full_output_relative_l2_vs_cublas"] for check in checks),
        "max_float64_sample_relative_l2": max(check["relative_l2"] for check in checks),
        "integer_and_zero_samples_exact": all(
            check["exact"] for check in checks if check["pattern"] in ("integer", "zero")),
        "baseline_build": provenance["build"],
        "baseline_cubin_sha256": provenance["ours_sha256"],
        "row_build": alternative["build"],
        "row_cubin_sha256": alternative["ours_sha256"],
    }


if __name__ == "__main__":
    rows = [summarize(RUN / f"p3_epilogue_large_{dtype}.json")
            for dtype in ("f16", "bf16")]
    target = HERE / "epilogue-benchmark-summary.json"
    target.write_text(json.dumps(rows, indent=2) + "\n")
    print(target)
