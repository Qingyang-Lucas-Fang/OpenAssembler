"""Verify this research snapshot without CUDA, Modal, PyTorch or third-party deps."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).with_name("evidence_manifest.json")
REVIEWED = """
README.md REPRODUCING.md docs/FRONTENDS.md docs/STATUS.md docs/TENSOR_GEMM.md
docs/research/A100_MMA_COMPILER.md docs/research/A100_MMA_RESULTS.md
docs/research/README.md docs/research/REFERENCES.md docs/research/SCALAR_COMPILER_RESULTS.md
docs/research/WEAKNESS_REMEDIATION.md docs/research/WHOLE_STACK_RESULTS.md
openassembler/backend/compile.py openassembler/backend/ra.py openassembler/backend/sched.py
openassembler/backend/tensor.py openassembler/build.py openassembler/cli.py
openassembler/gpu/cuda_driver.py openassembler/kernels/gemm.py openassembler/kernels/reductions.py
openassembler/rowwise.py openassembler/runner.py openassembler/torchfront/worker.py
specs/sm_80/synth_templates.json specs/sm_80/synth_fieldmaps.json
tests/test_cli_gemm.py tests/test_cli_rowwise.py tests/test_reduction_codegen.py
tests/test_tensor_pointer_homes.py tests/test_torchfront.py tests/test_torchfront_numerics.py
""".split()


def record(path):
    content = path.read_bytes()
    return {"path": str(path.relative_to(ROOT)), "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="record the current explicitly reviewed snapshot")
    args = parser.parse_args()
    if args.refresh:
        experiment = subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "exp/weaknesses"],
            cwd=ROOT).decode().split("\0")
        paths = sorted({name for name in [*experiment, *REVIEWED]
                        if name and ROOT / name != MANIFEST and (ROOT / name).is_file()})
        result = {"schema": 1, "baseline_git_commit": "3507e6198a9323060241ab05c4558052d7bac2b3",
            "git_head_at_snapshot": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
            "scope": "Reviewed remediation code/docs and durable experiment evidence; generated builds excluded",
            "files": [record(ROOT / name) for name in paths]}
        MANIFEST.write_text(json.dumps(result, indent=2) + "\n")
    obj = json.loads(MANIFEST.read_text())
    errors = []
    for expected in obj["files"]:
        path = ROOT / expected["path"]
        if not path.resolve().is_relative_to(ROOT) or not path.is_file():
            errors.append(f"missing or outside repository: {expected['path']}")
        elif record(path) != expected:
            errors.append(f"snapshot differs: {expected['path']}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Verified {len(obj['files'])} source/document/evidence files against the recorded SHA-256 snapshot")


if __name__ == "__main__":
    main()
