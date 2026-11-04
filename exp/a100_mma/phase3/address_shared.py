"""Fold copy-chunk shared addresses into proven LDGSTS immediates.

Global chunk pointers remain separate: their 128-KiB stride exceeds LDGSTS's
signed 12-bit source displacement. This experiment changes no shared module.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import sys
import time

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(HERE))
import epilogue
import search_combined
import search_sequences as search
from openassembler import build, oracle
from openassembler.cli import main
from openassembler.emu.operands import Mem
from openassembler.emu.program import parse_insn
from openassembler.isa import encode
from openassembler.isa.word import control, control_valid
from openassembler.ptx.parse import Addr, ImmI, Var


def fold_shared_addresses(source: str, *, bm=128, bn=128, bk=64,
                          wm=64, wn=64, swizzle_a=7, swizzle_b=7):
    threads = bm//wm * (bn//wn) * 32
    width = bk//8
    chunks_a, chunks_b = bm*width//threads, bn*width//threads
    if threads % width:
        raise ValueError("copy chunk rows must advance uniformly")
    row_delta = threads//width
    if row_delta & (swizzle_a | swizzle_b):
        raise ValueError("chunk rows must preserve the XOR-layout mask")
    counts = [0, 0]
    pattern = (r"add.u32 %r40,%r(\d+),%r(9|10);\n"
               r"cp.async.cg.shared.global \[%r40\],\[%d(\d+)\],16;")

    def replace(match):
        shared, stage, glob = map(int, match.groups())
        cid = shared-100
        which = 0 if cid < chunks_a else 1
        chunk = cid if which == 0 else cid-chunks_a
        chunks = chunks_a if which == 0 else chunks_b
        if not 0 <= chunk < chunks or glob != 10+cid:
            raise ValueError("unexpected generated copy operands")
        counts[which] += 1
        base = 100 if which == 0 else 100+chunks_a
        target = 80+which
        offset = chunk*row_delta*bk*2
        if not 0 <= offset < 524288 or offset % 16:
            raise ValueError("shared displacement exceeds the proven field")
        result = [f"add.u32 %r{target},%r{base},%r{stage};"] if chunk == 0 else []
        displacement = f"+{offset}" if offset else ""
        result += [f"cp.async.cg.shared.global [%r{target}{displacement}],[%d{glob}],16;"]
        return "\n".join(result)

    result = re.sub(pattern, replace, source)
    if counts != [2*chunks_a, 2*chunks_b]:
        raise ValueError(f"expected two complete copy groups, got {counts}")
    return result


class AddressSelector(search_combined.CombinedSelector):
    def async_copy(self, ins):
        if ins.mods != ("async", "cg", "shared", "global"):
            return super().async_copy(ins)
        if (ins.guard is not None or len(ins.args) != 3
                or not all(isinstance(arg, Addr) for arg in ins.args[:2])
                or not isinstance(ins.args[2], ImmI) or ins.args[2].value != 16):
            raise search.ra.NotSupported("expected unguarded 16-byte async copy")
        shared, source = ins.args[:2]
        if (not -524288 <= shared.off < 524288 or shared.off % 16
                or not -2048 <= source.off < 2048 or source.off % 16):
            raise search.ra.NotSupported("LDGSTS displacement range/alignment")
        sr = self.reg32(Var(shared.base))
        gr = self.wide(self.pair(source.base))
        fmt = lambda off: (f"+{off:#x}" if off >= 0 else f"-{abs(off):#x}") if off else ""
        self.emit(f"LDGSTS.E.BYPASS.128 [{{s0}}{fmt(shared.off)}], "
                  f"[{{s1}}.64{fmt(source.off)}]", [], [sr, gr], side=True)


class AddressEncoder:
    def __init__(self, base):
        self.base = base
        evidence = json.loads((HERE/"address-ldgsts-roundtrips.json").read_text())
        if len(evidence["roundtrips"]) != 89:
            raise RuntimeError("complete LDGSTS roundtrip evidence is required")
        self.seed = int(evidence["seed_word"], 16)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def encode(self, text, ctl=None):
        if not text.startswith("LDGSTS."):
            return self.base.encode(text, ctl)
        ins = parse_insn(0, text.replace("+-", "-"), 0)
        if (ins.guard is not None or ins.op != "LDGSTS"
                or ins.mods != ("E", "BYPASS", "128") or len(ins.ops) != 2):
            raise encode.EncodeError("unsupported address experiment instruction")
        dst, src = ins.ops
        if (not all(isinstance(arg, Mem) and arg.base is not None
                    and arg.base.file == "R" and arg.ur is None and arg.scale == 1
                    for arg in (dst, src))
                or dst.wide or not src.wide or src.base.idx % 2
                or not -524288 <= dst.off < 524288 or dst.off % 16
                or not -2048 <= src.off < 2048 or src.off % 16):
            raise encode.EncodeError("unsupported LDGSTS register/displacement")
        word = encode.set_bits(encode.set_bits(self.seed, 16, 8, dst.base.idx),
                               24, 8, src.base.idx)
        word = encode.set_bits(encode.set_bits(word, 32, 12, src.off & 4095),
                               44, 20, dst.off & 1048575)
        return self.base._ctl(word, ctl) if ctl is not None else word


@contextmanager
def address_backend():
    original_selector = search.backend.TensorSelector
    original_encoder = search.backend.encoder
    encoder = AddressEncoder(original_encoder())
    search.backend.TensorSelector = AddressSelector
    search.backend.encoder = lambda: encoder
    try:
        yield
    finally:
        search.backend.TensorSelector = original_selector
        search.backend.encoder = original_encoder


def make_source(m, n, k, dtype, **geometry):
    source = search_combined.source_for(m, n, k, dtype, **geometry)
    return fold_shared_addresses(source, **geometry)


def compile_case(m, n, k, dtype, tag="square", stage_pair=False):
    search.configure(gap=8, reuse=True)
    name = f"address_{'stagepair' if stage_pair else 'shared'}_{tag}_{dtype}"
    source = make_source(m, n, k, dtype)
    if stage_pair:
        from address_stage import specialize_two_stage
        source = specialize_two_stage(source, k=k)
    path = HERE/f"{name}.ptx"
    path.write_text(source)
    output = BASE/"build"/name
    captured = []
    old_generate = build.generate
    def capture(kernel, arch, opt_level):
        result = old_generate(kernel, arch, opt_level)
        captured.append(result)
        return result
    search.backend.STAGE_RECORDS.clear()
    started = time.perf_counter()
    build.generate = capture
    try:
        with address_backend():
            rc = main(["compile", str(path), "--out-dir", str(output), "--opt-level", "safe"])
    finally:
        build.generate = old_generate
    if rc or len(captured) != 1:
        raise RuntimeError(f"{name}: compilation failed: {rc}")
    manifest = json.loads((output/"manifest.json").read_text())
    records = oracle.parse_listing((output/manifest["entries"][0]["sass"]).read_text())
    code, labels, words, exits, regs = captured[0]
    assert [item.word for item in records[:len(words)]] == words
    normalize = lambda text: re.sub(r"\s+", "", text.replace(".reuse", "").replace("+-", "-"))
    for (want, ctl), actual in zip(code, records):
        assert ctl == control(actual.word) and control_valid(control(actual.word))
        if want.startswith(("HMMA", "LDSM", "LDGSTS", "LDGDEPBAR", "DEPBAR", "BAR", "STG")):
            assert normalize(want) == normalize(actual.text), (want, actual.text)
    sources = [Path(__file__), HERE/"address_stage.py", Path(search.__file__), Path(search_combined.__file__),
               Path(epilogue.__file__), BASE/"backend.py", BASE/"pipeline.py",
               BASE.parents[1]/"openassembler/backend/ra.py"]
    metadata = dict(compiler="OpenAssembler isolated LDGSTS shared-immediate experiment",
                    fallback=False, reference_and_ELF_container="NVIDIA ptxas on matching PTX",
                    candidate=dict(shared_chunk_immediates=True, global_chunk_immediates=False,
                                   stage_pair=stage_pair,
                                   paired_b=True, row_epilogue=True, gap=8, reuse=True),
                    geometry=dict(m=m, n=n, k=k, bm=128, bn=128, bk=64, wm=64, wn=64, stages=2),
                    source_ptx_sha256=hashlib.sha256(source.encode()).hexdigest(),
                    compiler_source_hashes={str(p.relative_to(BASE.parents[1])):
                                            hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                    cli_wall_ms=(time.perf_counter()-started)*1000,
                    kernels=search.backend.STAGE_RECORDS.copy(),
                    sass_opcode_counts=dict(Counter(item.text.split()[0] for item in records)),
                    exact_nvdisasm_tensor_memory_store_operands=True,
                    spliced_words_and_controls_exact=True,
                    address_field_evidence_sha256=hashlib.sha256(
                        (HERE/"address-ldgsts-roundtrips.json").read_bytes()).hexdigest())
    (output/"experiment.json").write_text(json.dumps(metadata, indent=2)+"\n")
    return dict(build=name, name=name, m=m, n=n, k=k, dtype=dtype,
                tile_m=128, tile_n=128, threads=128, smem=65536)


if __name__ == "__main__":
    search.backend.install()
    suite = []
    for dtype in ("f16", "bf16"):
        candidate = compile_case(4096, 4096, 4096, dtype)
        root = dict(candidate, build=f"search_combined_square_row_bpair_{dtype}",
                    name=f"address_shared_square_{dtype}_paired",
                    alternatives=[candidate], steady_warmup_rounds=16)
        suite.append(root)
    target = HERE/"address-shared-suite.json"
    target.write_text(json.dumps(suite, indent=2)+"\n")
    print(target)
