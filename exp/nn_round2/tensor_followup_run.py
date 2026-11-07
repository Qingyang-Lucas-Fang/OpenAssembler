"""Explicit one-A100 launcher for the isolated early-pointer follow-up."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

import modal

_SCRIPT = Path(__file__).resolve()
ROOT = _SCRIPT.parents[2] if len(_SCRIPT.parents) > 2 else Path("/root")
HERE = ROOT / "exp/nn_round2"
app = modal.App("openassembler-nn-round2-tensor-followup")
image = (modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
         .add_local_dir(ROOT / "openassembler", "/root/openassembler", copy=True,
                        ignore=["**/__pycache__", "**/*.pyc"])
         .add_local_file(HERE / "tensor_suite.py", "/root/exp/nn_round2/tensor_suite.py", copy=True))


@app.function(image=image, gpu="A100-40GB", timeout=900, max_containers=1, retries=0)
def remote(data, options):
    import sys
    sys.path.insert(0, "/root")
    from exp.nn_round2.tensor_suite import execute
    return execute(data, **options)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--samples", type=int, default=11)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("nn_followup", HERE / "tensor_followup.py")
    followup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(followup)
    payload = followup.load_payload(args.payload)
    options = {"rounds": args.rounds, "samples": args.samples, "graph_replays": 5}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out = args.out or HERE / "runs" / ("tensor-followup-" + stamp + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    record = {"schema": 1, "phase": "tensor_followup", "status": "running",
              "started_utc": datetime.now(timezone.utc).isoformat(), "options": options,
              "payload_sha256": hashlib.sha256(args.payload.read_bytes()).hexdigest(),
              "harness_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in (Path(__file__), HERE / "tensor_suite.py", HERE / "tensor_followup.py")},
              "hypothesis": payload["hypothesis"], "builds": payload["builds"],
              "source_identities": payload["source_identities"]}
    out.write_text(json.dumps(record, indent=2) + "\n")
    try:
        with app.run():
            result = remote.remote(payload, options)
        record.update(status="completed", result=result)
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        out.write_text(json.dumps(record, indent=2) + "\n")
        print(out, flush=True)


if __name__ == "__main__":
    main()
