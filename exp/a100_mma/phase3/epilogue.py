"""Isolated FP32 GEMM epilogue alternatives; no production source edits.

Two adjacent output elements per lane share an aligned eight-byte address.
The native-group selector reuses the lower/upper half of a four-register HMMA
accumulator group, while declaring both physical source reads to the scheduler.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))
import backend
from pipeline import pipelined_gemm
from openassembler import build
from openassembler.backend import ra
from openassembler.cli import main
from openassembler.ptx.parse import Var, Vec


def vectorize_epilogue(source: str, *, n: int, wm: int = 64, wn: int = 64,
                       address_mode: str = "pair") -> str:
    """Replace only the known generated scalar epilogue, verifying its boundary.

    pair: calculate a new row base for each accumulator pair.
    row: calculate each output row's pointer once and encode column displacement
         in the store (no repeated IMAD.WIDE for each pair).
    """
    if address_mode not in ("pair", "row") or n % 2:
        raise ValueError("require an even output leading dimension and pair/row mode")
    marker = "shr.u32 %r50,%r3,2; add.u32 %r50,%r50,%r5; add.u32 %r50,%r50,%r1;"
    if source.count(marker) != 1 or "ld.param.u64 %d5,[Bias]" in source:
        raise ValueError("expected exactly one unfused generated epilogue")
    prefix, old = source.split(marker)
    ma, nb = wm // 16, wn // 8
    if old.count("st.global.f32") != ma * nb * 4 or not old.rstrip().endswith("}"):
        raise ValueError("unexpected scalar epilogue geometry")
    lines = [marker,
             "and.b32 %r51,%r3,3; mul.lo.u32 %r51,%r51,2; add.u32 %r51,%r51,%r6; add.u32 %r51,%r51,%r2;"]
    if address_mode == "pair":
        for mi in range(ma):
            for ni in range(nb):
                for j, dr in ((0, 0), (2, 8)):
                    c = (mi * nb + ni) * 4 + j
                    lines += [f"add.u32 %r52,%r50,{mi*16+dr}; mad.lo.u32 %r53,%r52,{n},%r51;",
                              f"add.u32 %r53,%r53,{ni*8}; mul.wide.u32 %d3,%r53,4; add.u64 %d4,%d2,%d3;",
                              f"st.global.v2.f32 [%d4],{{%c{c},%c{c+1}}};"]
    else:
        # Each row start is eight-byte aligned: N even, CTA/warp N starts even,
        # and the lane's column coordinate is 2*(lane%4).
        for mi in range(ma):
            for j, dr in ((0, 0), (2, 8)):
                lines += [f"add.u32 %r52,%r50,{mi*16+dr}; mad.lo.u32 %r53,%r52,{n},%r51;",
                          "mul.wide.u32 %d3,%r53,4; add.u64 %d4,%d2,%d3;"]
                for ni in range(nb):
                    c = (mi * nb + ni) * 4 + j
                    off = f"+{ni*32}" if ni else ""
                    lines += [f"st.global.v2.f32 [%d4{off}],{{%c{c},%c{c+1}}};"]
    return prefix + "\n".join(lines + ["ret;", "}", ""])


class NativePairSelector(backend.TensorSelector):
    """Reference an aligned half of an existing quad without copying its data."""
    native_pair_stores = 0

    def store_global(self, ins, t):
        if (ins.guard is None and ins.mods == ("global", "v2", "f32")
                and len(ins.args) == 2 and isinstance(ins.args[1], Vec)
                and len(ins.args[1].elems) == 2
                and all(isinstance(x, Var) for x in ins.args[1].elems)):
            vals = tuple(self.reg32(x) for x in ins.args[1].elems)
            quads = [group for op in self.ops for group in op.defs
                     if isinstance(group, tuple) and len(group) == 4]
            if any(vals in (group[:2], group[2:]) for group in quads):
                base, off = self._addr(ins.args[0])
                self.emit(f"STG.E.64 [{{s0}}.64{off}], {{s1}}", [],
                          [base, vals[0], vals[1]], side=True)
                self.native_pair_stores += 1
                return
        return super().store_global(ins, t)


@contextmanager
def native_pair_selector(enabled: bool):
    original = backend.TensorSelector
    if enabled:
        backend.TensorSelector = NativePairSelector
    try:
        yield
    finally:
        backend.TensorSelector = original


def configure() -> None:
    backend.OPTIMIZE_LOCALS = True
    backend.HOIST_LOOP_INVARIANTS = False
    backend.MINED_SCALAR = True
    backend.RELAX_TENSOR = True
    backend.TENSOR_ISSUE_GAP = 8
    backend.LDSM_READ_BARRIER = False
    backend.INLINE_POINTER_INCREMENTS = True
    backend.TENSOR_REUSE = True
    backend.LIST_SCHEDULE = False
    backend.EXPERIMENTAL_RECYCLE_WITHOUT_WAIT = False


def compile_variant(name: str, m: int, n: int, k: int, dtype: str, variant: str):
    configure()
    source = pipelined_gemm(m, n, k, dtype, 128, 128, 64, 64, 64,
                            compact=True, b_window=8)
    if variant != "scalar":
        source = vectorize_epilogue(source, n=n,
                                    address_mode="row" if variant == "row_native" else "pair")
    path = HERE / f"epilogue_{name}.ptx"
    path.write_text(source)
    output = BASE / "build" / f"p3_epilogue_{name}"
    backend.STAGE_RECORDS.clear()
    start = time.perf_counter()
    with native_pair_selector(variant in ("pair_native", "row_native")):
        rc = main(["compile", str(path), "--out-dir", str(output), "--opt-level", "safe"])
    if rc:
        raise RuntimeError(f"{name}: compilation failed, rc={rc}")
    metadata = {
        "compiler": "OpenAssembler experimental tensor selector plus isolated epilogue selector",
        "fallback": False, "variant": variant,
        "reference_and_ELF_container": "NVIDIA ptxas on matching PTX",
        "cli_wall_ms": (time.perf_counter() - start) * 1000,
        "geometry": {"m": m, "n": n, "k": k, "bm": 128, "bn": 128,
                     "bk": 64, "wm": 64, "wn": 64, "stages": 2},
        "epilogue_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_ptx_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "compiler_source_hashes": {
            str(path.relative_to(BASE.parents[1])): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), BASE / "backend.py", BASE / "pipeline.py",
                         BASE.parents[1] / "openassembler/backend/ra.py",
                         BASE.parents[1] / "openassembler/isa/encode.py")
        },
        "kernels": backend.STAGE_RECORDS.copy(),
    }
    (output / "experiment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    manifest = json.loads((output / "manifest.json").read_text())
    entry = manifest["entries"][0]
    sass = (output / entry["sass"]).read_text()
    from openassembler import oracle
    from collections import Counter
    counts = Counter(row.text.split()[0] for row in oracle.parse_listing(sass))
    metadata["sass_opcode_counts"] = dict(counts)
    (output / "experiment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return {"build": output.name, "name": output.name, "m": m, "n": n, "k": k,
            "dtype": dtype, "tile_m": 128, "tile_n": 128, "threads": 128,
            "smem": 65536}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--large", action="store_true")
    args = parser.parse_args()
    backend.install()
    suites = []
    shapes = ((4096, 4096, 4096),) if args.large else ((256, 384, 256),)
    for dtype in ("f16", "bf16"):
        for m, n, k in shapes:
            configs = []
            for variant in ("scalar", "pair_packed", "pair_native", "row_native"):
                name = f"{variant}_{m}_{n}_{k}_{dtype}"
                configs.append(compile_variant(name, m, n, k, dtype, variant))
            root = configs[0]
            root["name"] = f"p3_epilogue_{'large' if args.large else 'smoke'}_{dtype}"
            root["steady_warmup_rounds"] = 16
            root["alternatives"] = configs[1:]
            suites.append(root)
    (HERE / f"epilogue-{'large' if args.large else 'smoke'}-suite.json").write_text(
        json.dumps(suites, indent=2) + "\n")
