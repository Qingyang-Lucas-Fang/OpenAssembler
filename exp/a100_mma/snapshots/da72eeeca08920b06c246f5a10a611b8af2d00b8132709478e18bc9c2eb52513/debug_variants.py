"""Separate address motion, scalar timing and tensor issue spacing hypotheses."""
import json
from backend import HERE, install
import backend
from kernels import cta_gemm
from openassembler.cli import main

if __name__ == "__main__":
    install()
    variants = [
        ("asyncfix_tensor2", False, True, 2, False),
        ("asyncfix_tensor4", False, True, 4, False),
        ("asyncfix_scalar", False, False, 4, True),
        ("asyncfix_licm_safe", True, False, 4, False),
        ("asyncfix_licm_fast", True, True, 4, True),
    ]
    suite = []
    for name, licm, relax, issue, mined in variants:
        backend.OPTIMIZE_LOCALS = True
        backend.HOIST_LOOP_INVARIANTS = licm
        backend.RELAX_TENSOR = relax
        backend.TENSOR_ISSUE_GAP = issue
        backend.MINED_SCALAR = mined
        source = HERE / "ptx" / f"{name}.ptx"
        source.write_text(cta_gemm(128,192,256,"f16",True))
        rc = main(["compile",str(source),"--out-dir",str(HERE/"build"/name),"--opt-level","safe"])
        if rc: raise RuntimeError(name)
        suite.append(dict(build=name,m=128,n=192,k=256,dtype="f16",tile_m=64,tile_n=64,threads=128))
    (HERE / "debug-suite.json").write_text(json.dumps(suite,indent=2)+"\n")
