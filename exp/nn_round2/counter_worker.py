"""Bounded, correctness-checked NCU collection for already admitted kernels.

The campaign calls ``run(payload)``. Each profile executes in a fresh process,
checks full outputs/canaries before profiling, warms up once, and profiles its
third matching launch by exact function name. NCU replay is diagnostic work,
not the separately collected performance timing. No production code is edited.
"""
from __future__ import annotations

import argparse
import base64
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time


METRICS = [
    "gpu__time_duration.sum",
    "sm__cycles_elapsed.avg.per_second",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "smsp__inst_executed.sum",
    "smsp__inst_executed_pipe_tensor.sum",
    "smsp__inst_executed_pipe_fma.sum",
    "smsp__inst_executed_pipe_sfu.sum",
    "smsp__inst_executed_pipe_xu.sum",
    "sm__pipe_xu_cycles_active.avg.pct_of_peak_sustained_active",
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_wait_per_warp_active.pct",
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
    "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
    "dram__bytes.sum",
    "lts__t_bytes.sum",
    "lts__t_sector_hit_rate.pct",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block_dynamic",
]
PROOF_PREFIX = "OA_COUNTER_CHECKS:"


def bounded_process(command, timeout, env=None):
    """Kill the complete NCU/worker process group on a bounded timeout."""
    started = time.monotonic()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, env=env, start_new_session=True)
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = process.communicate(timeout=10)
    return {"returncode": process.returncode, "stdout": stdout, "stderr": stderr,
            "timed_out": timed_out, "elapsed_seconds": time.monotonic()-started}


def parse_ncu_csv(stdout):
    """Read long metric rows or raw wide header/unit/data tables."""
    records = list(csv.reader(io.StringIO(stdout)))
    header_index = next((i for i, row in enumerate(records)
                         if "Metric Name" in row and "Metric Value" in row), None)
    if header_index is None:
        header_index = next((i for i,row in enumerate(records)
                             if row and row[0] == "ID" and "Kernel Name" in row
                             and any("__" in name for name in row)),None)
        if header_index is None:
            return {"format":"unrecognized","raw_rows": records, "metric_records": [], "metrics": {}}
        header=records[header_index]
        units=[""]*len(header)
        rows,metrics=[],{}
        for values in records[header_index+1:]:
            if len(values)!=len(header):
                continue
            if not values[0]:
                units=values
                continue
            if not values[0].isdigit():
                continue
            raw=dict(zip(header,values))
            for index,name in enumerate(header):
                if "__" not in name:
                    continue
                value=values[index]
                try:
                    number=float(value.replace(",",""))
                    number=number if math.isfinite(number) else None
                except ValueError:
                    number=None
                normalized={"ID":raw["ID"],"Kernel Name":raw.get("Kernel Name"),
                            "Metric Name":name,"Metric Unit":units[index],"Metric Value":value}
                rows.append(normalized)
                metrics.setdefault(name,[]).append({"value":number,"raw_value":value,
                                                    "unit":units[index],"kernel":raw.get("Kernel Name"),
                                                    "id":raw["ID"]})
        return {"format":"wide","raw_rows":records,"metric_records":rows,"metrics":metrics}
    header = records[header_index]
    rows, metrics = [], {}
    for values in records[header_index+1:]:
        if len(values) != len(header):
            continue
        row = dict(zip(header, values))
        if (not row.get("Metric Name") or row.get("Metric Name") == "Metric Name"
                or not row.get("ID", "").isdigit()):
            continue
        rows.append(row)
        name = row["Metric Name"]
        raw = row["Metric Value"]
        try:
            number = float(raw.replace(",", ""))
            number = number if math.isfinite(number) else None
        except ValueError:
            number = None
        metrics.setdefault(name, []).append({"value": number, "raw_value": raw,
                                            "unit": row.get("Metric Unit"),
                                            "kernel": row.get("Kernel Name"),
                                            "id": row.get("ID")})
    return {"format":"long","raw_rows": records, "metric_records": rows, "metrics": metrics}


def metric_query_accepted(query):
    """NCU 2025 can report invalid metric names with a zero exit status."""
    warning=(query.get("stdout","")+"\n"+query.get("stderr","")).lower()
    invalid=any(phrase in warning for phrase in ("invalid", "don't have suffixes", "not supported", "unknown metric"))
    return query.get("returncode")==0 and not query.get("timed_out") and not invalid


def parse_proof(stdout):
    for line in reversed(stdout.splitlines()):
        if line.startswith(PROOF_PREFIX):
            try:
                return json.loads(line[len(PROOF_PREFIX):])
            except json.JSONDecodeError:
                return None
    return None


def child(spec):
    """Three real matching launches: checked, warm-up, profiled/checked."""
    import ctypes as C
    import torch
    from openassembler.gpu.cuda_driver import Driver

    torch.set_num_threads(2)
    torch.cuda.init()
    anchor = torch.empty(1, device="cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    driver = Driver()
    if driver.info().cc != "8.0":
        raise RuntimeError("counter worker requires actual SM80")
    binary = base64.b64decode(spec["cubin"], validate=True)
    if hashlib.sha256(binary).hexdigest() != spec["cubin_sha256"]:
        raise RuntimeError("counter binary differs from admitted payload")
    module = driver.load(binary)
    fn = driver.function(module, spec["kernel"])
    if spec["smem"]:
        driver.ck(driver.cu.cuFuncSetAttribute(fn, 8, spec["smem"]), "shared memory opt-in")
    resources = driver.func_attrs(fn)
    stores, outputs = [], []

    def guarded(shape):
        count = math.prod(shape)
        storage = torch.full((count+512,), -12345., device="cuda", dtype=torch.float32)
        out = storage[256:-256].view(shape)
        out.fill_(float("nan"))
        stores.append(storage)
        outputs.append(out)
        return out

    if spec["kind"] == "operator":
        from exp.nn_round2.operator_worker import make_inputs, reference
        meta = spec["operator"]
        inputs = make_inputs(meta["op"], tuple(spec["shape"]), "normal", spec["seed"])
        expected = reference(meta["op"], inputs)
        for value in expected:
            guarded(tuple(value.shape))
        tensors = inputs + outputs + [outputs[0]]*(3-len(outputs))
        count = spec["shape"][0]
        values = [C.c_uint64(t.data_ptr()) for t in tensors] + [C.c_uint32(count)]
        rtol, atol = meta["rtol"], meta["atol"]
    elif spec["kind"] == "tensor":
        m,n,k = spec["shape"]
        torch.manual_seed(spec["seed"])
        dtype = torch.float16 if spec["dtype"] == "f16" else torch.bfloat16
        a = torch.randn((m,k), device="cuda", dtype=dtype)
        b = torch.randn((n,k), device="cuda", dtype=dtype)
        inputs = [a,b]
        out = guarded((m,n))
        expected = [a.double() @ b.double().T]
        values = [C.c_uint64(t.data_ptr()) for t in (a,b,out)]
        rtol, atol = 3e-4, 3e-3
    else:
        raise ValueError(spec["kind"])
    original = [value.clone().view(torch.uint8) for value in inputs]
    params = (C.c_void_p*len(values))(*[C.cast(C.pointer(value), C.c_void_p) for value in values])

    def launch():
        driver.ck(driver.cu.cuLaunchKernel(fn, *spec["grid"], spec["threads"],1,1,
                  spec["smem"], C.c_void_p(torch.cuda.current_stream().cuda_stream), params,None),
                  "profiled "+spec["id"])

    def validate():
        torch.cuda.synchronize()
        checks = []
        for actual, truth in zip(outputs, expected):
            difference = actual.double()-truth.double()
            relative = float(difference.norm()/truth.double().norm().clamp_min(1e-30))
            passed = bool(torch.isfinite(actual).all() and
                          torch.isclose(actual.double(), truth.double(), rtol=rtol, atol=atol).all())
            if spec["kind"] == "tensor":
                passed = passed and relative < 5e-5
            checks.append({"passed": passed, "elements": actual.numel(),
                           "max_abs": float(difference.abs().max()), "relative_l2": relative})
        canaries = all(bool((s[:256] == -12345.).all() and (s[-256:] == -12345.).all()) for s in stores)
        immutable = all(torch.equal(value.view(torch.uint8), saved) for value,saved in zip(inputs, original))
        return {"passed": canaries and immutable and all(check["passed"] for check in checks),
                "canaries": canaries, "inputs_unchanged": immutable, "outputs": checks}

    try:
        launch()
        before = validate()
        if not before["passed"]:
            raise RuntimeError("full-output preprofile correctness failed: "+json.dumps(before))
        launch()
        torch.cuda.synchronize()
        for out in outputs:
            out.fill_(float("nan"))
        # NCU --launch-skip 2 --launch-count 1 selects this exact function call.
        launch()
        after = validate()
        proof = {"passed": before["passed"] and after["passed"], "before": before,
                 "after": after, "resources": resources, "device": driver.info().__dict__,
                 "matching_launches": 3, "profiled_matching_launch_index": 2,
                 "cubin_sha256": spec["cubin_sha256"], "seed": spec["seed"]}
        print(PROOF_PREFIX+json.dumps(proof, allow_nan=False), flush=True)
        if not proof["passed"]:
            raise RuntimeError("profiled output correctness failed")
    finally:
        torch.cuda.synchronize()
        driver.unload(module)


def run(payload):
    """Return raw NCU CSV/errors/skips with bounded matched profiles."""
    from openassembler.gpu.cuda_driver import Driver

    profiles = payload.get("profiles", [])
    if not 1 <= len(profiles) <= 12:
        raise ValueError("counter payload must contain 1..12 profiles")
    if len({spec["id"] for spec in profiles}) != len(profiles):
        raise ValueError("counter profile IDs must be unique")
    started = time.monotonic()
    ncu = shutil.which("ncu") or shutil.which("nv-nsight-cu-cli")
    metadata = {"schema": 1, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "device": Driver().info().__dict__, "ncu": ncu,
                "plan": {key:value for key,value in payload.items() if key != "profiles"},
                "metrics_requested": METRICS, "profiles": [],
                "counter_worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "scope": "Diagnostic Nsight kernel replay, not benchmark latency. Exact kernel filter, two matching launches skipped; normal seeded inputs, complete output/guard/nonmutation checks. Cache-control none and clock-control none. Replay behavior differs from graph timing."}
    if not ncu:
        metadata.update(status="unavailable", reason="Nsight Compute is absent from the image")
        return metadata
    metadata["version"] = bounded_process([ncu,"--version"], 10)
    query_results, supported = {}, []
    for metric in METRICS:
        query = bounded_process([ncu,"--query-metrics","--metrics",metric], 5)
        query_results[metric] = query
        if metric_query_accepted(query):
            supported.append(metric)
    metadata["metric_queries"] = query_results
    metadata["metrics_supported_by_query"] = supported
    metadata["metrics_not_query_accepted"] = [metric for metric in METRICS if metric not in supported]
    metadata["metric_query_note"] = "Host launch attributes can appear automatically in raw output despite not being accepted as queried hardware metrics. Missing SFU/XU spellings are unavailable, not zero; XU, if present, is its named combined execution pipe, not an isolated SFU attribution."
    if not supported:
        metadata.update(status="unavailable", reason="No requested NCU metric accepted by the installed profiler")
        return metadata
    with tempfile.TemporaryDirectory(prefix="oa-round2-counters-") as temporary:
        directory = Path(temporary)
        for index,spec in enumerate(profiles):
            # Only the known literal source kernel identifier enters regex.
            if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", spec["kernel"]):
                raise ValueError("counter kernel must be a literal identifier")
            file = directory/f"profile-{index}.json"
            file.write_text(json.dumps(spec))
            command = [ncu,"--clock-control","none","--cache-control","none",
                       "--replay-mode","kernel","--target-processes","all",
                       "--kernel-name-base","function","--kernel-name","regex:^"+spec["kernel"]+"$",
                       "--launch-skip","2","--launch-count","1","--csv","--page","raw",
                       "--metrics",",".join(supported),sys.executable,str(Path(__file__).resolve()),
                       "--child",str(file)]
            environment = dict(os.environ)
            root = str(Path(__file__).resolve().parents[2])
            environment["PYTHONPATH"] = root + (":"+environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
            result = bounded_process(command, min(90, max(10, payload.get("profile_timeout_seconds",90))), environment)
            proof = parse_proof(result["stdout"])
            table = parse_ncu_csv(result["stdout"])
            collected = result["returncode"] == 0 and not result["timed_out"] and bool(table["metrics"]) and proof is not None and proof.get("passed")
            row = {key:value for key,value in spec.items() if key != "cubin"}
            row.update(status="collected" if collected else "timeout" if result["timed_out"] else "failed",
                       command=command, process=result, proof=proof, **table)
            metadata["profiles"].append(row)
            print(json.dumps({"counter":spec["id"],"status":row["status"],"metrics":len(table["metrics"])}),flush=True)
    metadata["elapsed_seconds"] = time.monotonic()-started
    metadata["status"] = "completed" if all(row["status"] == "collected" for row in metadata["profiles"]) else "completed_with_failures"
    return metadata


if __name__ == "__main__":
    root = str(Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", type=Path, required=True)
    args = parser.parse_args()
    child(json.loads(args.child.read_text()))
