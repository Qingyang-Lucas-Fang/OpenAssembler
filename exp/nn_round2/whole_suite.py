"""Build a reproducible native-routed NN cohort, using only exp/ outputs.

PYTHONPATH=. .venv/bin/python exp/nn_round2/whole_suite.py build
PYTHONPATH=. modal run exp/nn_round2/campaign.py --phase whole
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
HERE = ROOT / "exp/nn_round2"
CASES = [
    dict(name="relu_mlp_small", family="mlp", tokens=128, width=128, hidden=512, layers=1, activation="relu"),
    dict(name="relu_mlp_wide", family="mlp", tokens=512, width=512, hidden=2048, layers=1, activation="relu"),
    dict(name="silu_gated_mlp", family="gated", tokens=256, width=256, hidden=1024, layers=1, activation="silu"),
    dict(name="transformer_single", family="transformer", batch=1, seq=128, tokens=128, width=128, hidden=512, heads=1, layers=1, activation="gelu_tanh"),
    dict(name="transformer_multihead", family="transformer", batch=2, seq=128, tokens=256, width=256, hidden=1024, heads=2, layers=2, activation="gelu_tanh"),
    dict(name="transformer_long", family="transformer", batch=1, seq=256, tokens=256, width=256, hidden=1024, heads=2, layers=1, activation="gelu_tanh"),
]
GATE = {"output": {"rtol": 1e-3, "atol": 1e-5},
        "gradient": {"rtol": 2e-3, "atol": 2e-5},
        "parameter": {"rtol": 1e-3, "atol": 1e-5},
        "optimizer": {"rtol": 2e-3, "atol": 2e-5}}


def shapes(cases=CASES):
    result = set()
    def linear(m, n, k):
        result.update(((m, n, k), (m, k, n), (n, k, m)))
    for c in cases:
        m, d, h = (c[k] for k in ("tokens", "width", "hidden"))
        linear(m, h, d); linear(m, d, h)
        if c["family"] == "transformer":
            linear(m, 3*d, d); linear(m, d, d); linear(m, 256, d)
            s, k = c["seq"], d // c["heads"]
            linear(s, s, k); linear(s, k, s)
    return sorted(result)


def build():
    from openassembler.gemm import compile_gemm
    from openassembler.kernels.gemm import GemmPlan
    root = HERE / "whole_build"
    root.mkdir(parents=True, exist_ok=True)
    payload = {"schema": 1, "cases": CASES, "gate": GATE, "gemms": {},
               "operator_payload": json.loads((HERE / "operator_build/payload.json").read_text())}
    for m, n, k in shapes():
        key = f"{m}_{n}_{k}"
        rec = {"shape": [m, n, k], "kernels": {}}
        for mode in ("legacy", "direct"):
            output = root / key / mode
            plan = GemmPlan(m, n, k, pointer_mode=mode)
            if compile_gemm(plan, output):
                raise RuntimeError(f"build failed: {key}/{mode}")
            manifest = json.loads((output / "manifest.json").read_text())
            entry = manifest["entries"][0]
            if any(x["compiled_by"] != "openassembler" for x in entry["kernels"]):
                raise RuntimeError("native fallback forbidden")
            for name, artifact in ((f"oa_{mode}", "cubin"), ("ptxas", "reference")):
                blob = (output / entry[artifact]).read_bytes()
                rec["kernels"][name] = base64.b64encode(blob).decode()
                rec[name + "_sha256"] = hashlib.sha256(blob).hexdigest()
            rec[mode + "_metadata"] = entry["kernels"]
        payload["gemms"][key] = rec
    payload["build_sources"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (ROOT / "openassembler").rglob("*.py")}
    path = root / "payload.json"
    path.write_text(json.dumps(payload) + "\n")
    (HERE / "whole_preregistration.json").write_text(json.dumps({
        "cases": CASES, "gates": GATE, "optimizer": "AdamW lr=1e-4, eps=1e-8, wd=.01, capturable foreach=False",
        "precision": "FP32 parameters and non-GEMM operations; every GEMM rounds both operands to FP16, accumulates/outputs FP32; explicit straight-through derivative",
        "methods": ["oa_legacy", "oa_direct", "oa_tensor_only", "ptxas", "ptxas_tensor_only", "cuda_library"],
        "reference_groups": {"native_allops": "oa_legacy/direct vs ptxas, all route eligible operators natively", "tensor_only": "oa_tensor_only and cuda_library vs ptxas_tensor_only; surrounding Torch operations identical"},
        "excluded_native_operator": "gelu_tanh_backward failed isolated predeclared gate in both compilers; use shared Torch derivative",
        "coverage": "Native GEMM forward/dX/dW and eligible native pointwise/RMSNorm/softmax; shared Torch embeddings, masks, loss, norm affine, optimizer, clipping, packing, casts",
        "goal": "faster after full output/gradient/state correctness, no guaranteed superiority",
        "timing": "paired shuffled event and host-wall spans; graph replay tested separately, full post-timing state gate"}, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("matrix", "build"))
    args = parser.parse_args()
    print(json.dumps({"cases": CASES, "shapes": shapes(), "gates": GATE}, indent=2)) if args.mode == "matrix" else build()
