"""Experiment: overlap next-tile pointer arithmetic with tensor computation.

Only the 16 independent 64-bit pointer advances are moved: from after the
compute-stage barrier to immediately after committing the next-tile copies.
Those copies still read their original pointers first. HMMA/LDSM/copy/barrier
instructions and all synchronization dependencies remain; the tensor scheduler
continues protecting asynchronous-copy address reads against these updates.

This is a hypothesis, not an admitted compiler optimization. In particular,
earlier pointer updates may wait for asynchronous address consumption and lose.
Use FP16 on a known favorable and unfavorable shape first; do not expand the
BF16 campaign unless the measured holdout results justify doing so.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

_SCRIPT = Path(__file__).resolve()
ROOT = _SCRIPT.parents[2] if len(_SCRIPT.parents) > 2 else Path.cwd()
HERE = _SCRIPT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def tensor_module():
    spec = importlib.util.spec_from_file_location("nn_round2_frozen_tensor", HERE / "tensor_suite.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def relocate_advances(source, dtype, k):
    from openassembler.kernels.gemm import _advance, _body, _copies, _compute
    for current, following, count in (("0", "32768", 2 if k > 128 else 1),
                                       ("32768", "0", 1 if k > 128 else 0)):
        old = "\n".join(_body(dtype, current, following, folded=True))
        new = "\n".join(["cp.async.wait_group 0; bar.sync 0;"] + _copies(following, folded=True)
                        + _advance() + _compute(dtype, current) + ["bar.sync 0;"])
        if source.count(old) != count:
            raise ValueError("unexpected stage-pair source; refuse imprecise relocation")
        source = source.replace(old, new)
    return source


def instruction_multiset(source):
    from openassembler.ptx.parse import parse_module
    kernel = parse_module(source).kernels["gemm"]
    return Counter((ins.op, ins.mods, str(ins.args), ins.guard) for ins in kernel.body)


def followup_definitions(dtypes=("f16",)):
    rows = []
    for name, shape, role in (("mlp_wide_forward", (1024, 4096, 1024), "favorable direct-pointer forward shape"),
                              ("tall", (16384, 4096, 4096), "unfavorable direct-pointer tall control")):
        for dtype in dtypes:
            if dtype not in ("f16", "bf16"):
                raise ValueError("follow-up supports f16/bf16")
            rows.append({"name": name + "_" + dtype, "shape": list(shape), "dtype": dtype, "role": role,
                         "candidates": [{"name": "exp_" + pointer + "_early_advance", "order": "m_fast",
                                         "pointer": pointer, "epilogue": "none", "transform": "advance_after_copy_commit"}
                                        for pointer in ("legacy", "direct")]})
    return rows


def build_payload(base_manifest, dtypes=("f16",)):
    base = tensor_module()
    original_source = base.source_and_config
    def candidate_source(case, candidate):
        source, config, plan = original_source(case, candidate)
        moved = relocate_advances(source, case["dtype"], case["shape"][2])
        if instruction_multiset(source) != instruction_multiset(moved):
            raise AssertionError("pointer relocation changed PTX operations/operands")
        return moved, config, plan
    # Only experiment helper dispatch is patched, never production methods.
    with patch.object(base, "definitions", lambda *args: followup_definitions(dtypes)), \
         patch.object(base, "source_and_config", candidate_source):
        payload = base.build_payload("full", dtypes, HERE / "tensor_followup_builds")
    controls = base.load_payload(base_manifest)
    controls_by_case = {case["name"]: case for case in controls["cases"]}
    wanted = set()
    for case in payload["cases"]:
        for method in case["methods"]:
            if method["kind"] == "same_ptx_ptxas":
                method["name"] = "ptxas_early_advance"
        control = controls_by_case[case["name"]]
        for method in control["methods"]:
            if method["name"] in ("oa_legacy", "oa_direct", "ptxas_m_fast"):
                case["methods"].append(method)
                case["sources"][method["source_id"]] = control["sources"][method["source_id"]]
                if method["kind"] == "native_oa":
                    wanted.add((case["name"], method["name"]))
    payload["builds"] += [record for record in controls["builds"]
                          if (record["case"], record["candidate"]["name"]) in wanted]
    payload["hypothesis"] = "move pointer updates after cp.async commit, before tensor work; retain all dependency checks"
    payload["same_ptx_source_instruction_multiset"] = True
    payload["frozen_control_manifest"] = str(Path(base_manifest).resolve().relative_to(ROOT))
    payload["followup_harness_sha256"] = base.digest(_SCRIPT.read_bytes())
    return payload


def load_payload(path):
    import base64
    base = tensor_module()
    payload = json.loads(Path(path).read_text())
    for filename, expected in payload["source_identities"].items():
        if base.digest((ROOT / filename).read_bytes()) != expected:
            raise RuntimeError(f"compiler source drifted: {filename}")
    for case in payload["cases"]:
        for method in case["methods"]:
            binary = base64.b64decode(method["binary_b64"])
            if method["kind"] == "same_ptx_ptxas":
                expected = case["sources"][method["source_id"]]["reference_sha256"]
            else:
                expected = next(row["native_sha256"] for row in payload["builds"]
                                if row["case"] == case["name"] and row["candidate"]["name"] == method["name"])
            if base.digest(binary) != expected:
                raise RuntimeError(f"follow-up binary identity mismatch: {case['name']}/{method['name']}")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--dtype", choices=("f16", "bf16", "both"), default="f16")
    args = parser.parse_args()
    dtypes = ("f16", "bf16") if args.dtype == "both" else (args.dtype,)
    payload = build_payload(args.base_manifest, dtypes)
    output = HERE / ("tensor_followup_payload_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + ".json")
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
