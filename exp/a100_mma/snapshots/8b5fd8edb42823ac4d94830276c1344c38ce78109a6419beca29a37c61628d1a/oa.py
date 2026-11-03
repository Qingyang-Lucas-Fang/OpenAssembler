"""Experimental CLI, retaining production dispatch/build/run and atomic outputs."""
from __future__ import annotations
import os
import sys
import argparse
import json
import time
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
from backend import HERE, install

if __name__ == "__main__":
    import backend
    from openassembler.cli import build_parser, dispatch
    experiment_parser = argparse.ArgumentParser(add_help=False)
    experiment_parser.add_argument("--ssa-locals", action="store_true")
    experiment_parser.add_argument("--licm", action="store_true")
    experiment_parser.add_argument("--interlocked-tensor", action="store_true")
    experiment_parser.add_argument("--mined-scalar", action="store_true")
    experiment_parser.add_argument("--tensor-issue", type=int, choices=(2, 4, 8, 12), default=4)
    experimental, remaining = experiment_parser.parse_known_args()
    backend.OPTIMIZE_LOCALS = experimental.ssa_locals
    backend.HOIST_LOOP_INVARIANTS = experimental.licm
    backend.RELAX_TENSOR = experimental.interlocked_tensor
    backend.MINED_SCALAR = experimental.mined_scalar
    backend.TENSOR_ISSUE_GAP = experimental.tensor_issue
    args = build_parser().parse_args(remaining)
    if args.cmd not in ("compile", "run", "cnr"):
        raise SystemExit("This isolated CLI permits compile/run/cnr only")
    output = Path(getattr(args, "out_dir", getattr(args, "outdir", HERE / "build"))).resolve()
    if not output.is_relative_to(HERE):
        raise SystemExit(f"Experimental outputs must stay under {HERE}; pass --out-dir explicitly")
    if args.cmd in ("compile", "cnr"):
        source = Path(args.src).resolve()
        if not source.is_relative_to(HERE):
            raise SystemExit("Keep experimental input sources beside this isolated CLI")
    install()
    tempfile.tempdir = str(HERE / "scratch")
    print("EXPERIMENTAL A100 MMA selector; conservative schedule; no core files changed", flush=True)
    start = time.perf_counter()
    rc = dispatch(args)
    if backend.STAGE_RECORDS and output.is_dir():
        record = {"compiler": "OpenAssembler experimental MMA selector (exp/a100_mma/backend.py)",
                  "reference_and_ELF_container": "NVIDIA ptxas", "fallback": False,
                  "cli_rc": rc, "cli_wall_ms": (time.perf_counter() - start) * 1000,
                  "kernels": backend.STAGE_RECORDS}
        (output / "experiment.json").write_text(json.dumps(record, indent=2) + "\n")
    raise SystemExit(rc)

