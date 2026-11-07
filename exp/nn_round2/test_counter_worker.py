"""CPU checks for counter transport/parsing and admitted-binary selection."""
import csv
import io
import json
import sys

from exp.nn_round2.counter_worker import (PROOF_PREFIX, bounded_process,
                                          metric_query_accepted, parse_ncu_csv, parse_proof)


def test_csv_parser_keeps_every_metric_and_metric_units():
    output = io.StringIO()
    output.write("==PROF== Connected\n")
    writer = csv.writer(output)
    writer.writerow(["ID", "Kernel Name", "Metric Name", "Metric Unit", "Metric Value"])
    writer.writerow(["0", "gemm", "smsp__inst_executed.sum", "inst", "1,234"])
    writer.writerow(["0", "gemm", "dram__bytes.sum", "byte", "9876"])
    writer.writerow(["0", "gemm", "lts__t_sector_hit_rate.pct", "%", "90.25"])
    output.write(PROOF_PREFIX+json.dumps({"passed": True, "outputs": [1, 2, 3]})+"\n")
    result = parse_ncu_csv(output.getvalue())
    assert len(result["metric_records"]) == 3
    assert result["metrics"]["smsp__inst_executed.sum"] == [
        {"value": 1234., "raw_value": "1,234", "unit": "inst", "kernel": "gemm", "id": "0"}
    ]
    assert result["metrics"]["lts__t_sector_hit_rate.pct"][0]["value"] == 90.25
    assert parse_proof(output.getvalue()) == {"passed": True, "outputs": [1, 2, 3]}


def test_profiler_failure_is_not_zero_metrics_or_successful_proof():
    result = parse_ncu_csv("==ERROR== ERR_NVGPUCTRPERM: permission denied\n")
    assert result["metric_records"] == [] and result["metrics"] == {}
    assert parse_proof("no child proof") is None
    assert parse_proof(PROOF_PREFIX+"invalid json") is None


def test_wide_raw_units_and_multiple_metrics_are_recovered():
    stream=io.StringIO()
    writer=csv.writer(stream)
    writer.writerow(["ID","Kernel Name","gpu__time_duration.sum","dram__bytes.sum"])
    writer.writerow(["","","nsecond","byte"])
    writer.writerow(["0","gemm","1250","1,024"])
    result=parse_ncu_csv(stream.getvalue())
    assert result["format"]=="wide"
    assert result["metrics"]["gpu__time_duration.sum"][0]["unit"]=="nsecond"
    assert result["metrics"]["dram__bytes.sum"][0]["value"]==1024


def test_zero_exit_invalid_metric_warning_is_rejected():
    accepted={"returncode":0,"timed_out":False,"stdout":"Device A100","stderr":""}
    assert metric_query_accepted(accepted)
    accepted["stdout"]+="\n==WARNING== The below metrics are either invalid or don't have suffixes."
    assert not metric_query_accepted(accepted)


def test_process_timeout_kills_worker_and_retains_error_output():
    result = bounded_process([sys.executable,"-c",
                              "import time; print('started',flush=True); time.sleep(60)"], .1)
    assert result["timed_out"] and result["returncode"] != 0
    assert "started" in result["stdout"]
    assert result["elapsed_seconds"] < 10


def test_full_counter_plan_uses_only_exact_previously_admitted_binaries():
    from exp.nn_round2.counter_plan import build_plan
    plan = build_plan()
    assert len(plan["profiles"]) == 12
    assert plan["profile_timeout_seconds"] <= 90
    ids = [profile["id"] for profile in plan["profiles"]]
    assert len(ids) == len(set(ids))
    assert {profile["kind"] for profile in plan["profiles"]} == {"operator", "tensor"}
    for profile in plan["profiles"]:
        assert profile["cubin_sha256"] and profile["admitted_evidence_sha256"]
        assert profile["threads"] == 128
        assert len(profile["grid"]) == 3
