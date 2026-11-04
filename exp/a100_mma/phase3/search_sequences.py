"""Isolated SM80 sequence/register search, using OpenAssembler-produced code.

All compiler monkeypatches are process-local and restored after each build. The
suite preserves cp.async waits, CTA barriers and per-output accumulation order.
No benchmark result is inferred from instruction count.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import sys
import time

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))
import backend
from pipeline import pipelined_gemm
from openassembler.backend import ra
from openassembler.cli import main
from openassembler.ptx.parse import Var, Vec


def compute_sequence(dtype="f16", *, bk=64, wm=64, wn=64, paired_b=False,
                     overlap=False, column=True, paired_preaddress=False):
    """Known full-tile fragment map; overlap never crosses a shared-stage edge.

    x4 B loads concatenate two x2 fragments. Lanes 16..31 provide addresses
    for the second N tile; their low-four-bit mapping is unchanged. x4's two
    aligned halves are passed directly to HMMA by HalfBSelector below.
    """
    ma, nb = wm // 16, wn // 8
    if paired_b and nb % 2:
        raise ValueError("paired B requires an even number of N fragments")
    lines = []
    if paired_b and not paired_preaddress:
        lines += ["shr.u32 %r47,%r3,4;", f"mul.lo.u32 %r47,%r47,{16*bk};"]

    def addresses(ik, ar=41, br=42):
        result = [f"add.u32 %r{ar},%r{200+ik*ma},%r9;",
                  f"add.u32 %r{br},%r{300+ik*nb},%r9;"]
        if paired_b and not paired_preaddress:
            result += [f"add.u32 %r{br},%r{br},%r47;"]
        return result

    def load_a(mi, ar=41):
        aa = ",".join(f"%r{400+mi*4+j}" for j in range(4))
        return [f"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{aa}}},[%r{ar}+{mi*32*bk}];"]

    def load_b(base, br=42):
        width = 4 if paired_b else 2
        return [f"ldmatrix.sync.aligned.m8n8.x{width}.shared.b16 "
                "{" + ",".join(f"%r{base+ni*2+j}" for j in range(width)) +
                f"}},[%r{br}+{ni*16*bk}];" for ni in range(0, nb, width//2)]

    def mma(mi, ni, base):
        aa = ",".join(f"%r{400+mi*4+j}" for j in range(4))
        bb = ",".join(f"%r{base+ni*2+j}" for j in range(2))
        cc = ",".join(f"%c{(mi*nb+ni)*4+j}" for j in range(4))
        return (f"mma.sync.aligned.m16n8k16.row.col.f32.{dtype}.{dtype}.f32 "
                f"{{{cc}}},{{{aa}}},{{{bb}}},{{{cc}}};")

    for ik in range(bk // 16):
        base = 500 + (100 * (ik % 2) if overlap else 0)
        if not overlap or ik == 0:
            lines += addresses(ik)
            for mi in range(ma):
                lines += load_a(mi)
            lines += load_b(base)
        if overlap and ik + 1 < bk // 16:
            # Both fragment addresses are independent of current tensor inputs.
            lines += addresses(ik+1, 45, 46)
        if overlap:
            for mi in range(ma):
                lines += [mma(mi, ni, base) for ni in range(nb)]
                if ik + 1 < bk // 16:
                    # This A row has no remaining readers in the current slice.
                    lines += load_a(mi, 45)
                    if mi == 0:
                        lines += load_b(600 if base == 500 else 500, 46)
        elif column:
            lines += [mma(mi, ni, base) for ni in range(nb) for mi in range(ma)]
        else:
            lines += [mma(mi, ni, base) for mi in range(ma) for ni in range(nb)]
    return "\n".join(lines)


def make_source(m, n, k, dtype, *, paired_b=False, overlap=False,
                column=True, swizzle_a=7, swizzle_b=7,
                bm=128, bn=128, bk=64, wm=64, wn=64,
                paired_preaddress=False):
    source = pipelined_gemm(m, n, k, dtype, bm, bn, bk, wm, wn,
                            compact=True, b_window=0,
                            swizzle_a=swizzle_a, swizzle_b=swizzle_b)
    pattern = r"add.u32 %r41,%r200,%r9;\n.*?(?=\nbar.sync 0;)"
    blocks = re.findall(pattern, source, flags=re.S)
    if (len(blocks) != 2 or blocks[0] != blocks[1]
            or blocks[0].count("mma.sync") != (bk//16)*(wm//16)*(wn//8)):
        raise ValueError("unexpected generated compute-region boundaries")
    replacement = compute_sequence(dtype, bk=bk, wm=wm, wn=wn,
                                   paired_b=paired_b, overlap=overlap, column=column,
                                   paired_preaddress=paired_preaddress)
    source = re.sub(pattern, lambda _: replacement, source, flags=re.S)
    if paired_preaddress:
        if not paired_b:
            raise ValueError("paired_preaddress requires paired B loads")
        # Adjust the already loop-invariant shared addresses once. Every lane
        # supplies the same low-16-lane mapping; the upper half supplies N+8.
        lines = ["shr.u32 %r47,%r3,4;", f"mul.lo.u32 %r47,%r47,{16*bk};"]
        lines += [f"add.u32 %r{300+ik*(wn//8)},%r{300+ik*(wn//8)},%r47;"
                  for ik in range(bk//16)]
        marker = "add.u32 %r40,%r100,%r9;"
        if source.count(marker) != 1:
            raise ValueError("expected unique prologue async-copy boundary")
        source = source.replace(marker,"\n".join(lines)+"\n"+marker,1)
    return source


class HalfBSelector(backend.TensorSelector):
    """Use each aligned half of an LDSM x4 result without packing MOVs.

    Register allocation still sees the enclosing quad; scheduler sees both
    scalar B reads, not just the printed implicit-group base register.
    """
    def mma(self, ins):
        if (ins.guard is not None or len(ins.args) != 4
                or any(not isinstance(x, Vec) for x in ins.args)
                or tuple(len(x.elems) for x in ins.args) != (4, 4, 2, 4)
                or ins.mods[:6] != ("sync", "aligned", "m16n8k16", "row", "col", "f32")
                or len(ins.mods) != 9 or ins.mods[6] not in ("f16", "bf16")
                or ins.mods[7:] != (ins.mods[6], "f32")):
            return super().mma(ins)
        d, a, b, c = ins.args
        if any(not isinstance(x, Var) for group in ins.args for x in group.elems):
            return super().mma(ins)
        bv = tuple(self.reg32(v) for v in b.elems)
        quads = [g for op in self.ops for g in op.defs
                 if isinstance(g, tuple) and len(g) == 4]
        if not any(bv == g[:2] or bv == g[2:] for g in quads):
            return super().mma(ins)
        aa = self.wide(tuple(self.reg32(v) for v in a.elems))
        names = tuple(v.name for v in d.elems)
        if self.has_cfg:
            dd = tuple(self.homes[name] for name in names)
            cc0 = tuple(self.reg32(v) for v in c.elems)
            cc = cc0 if cc0 == dd else self.wide(cc0)
        else:
            dd = tuple(self.new() for _ in range(4))
            cc = self.wide(tuple(self.reg32(v) for v in c.elems))
        suffix = ".BF16" if ins.mods[6] == "bf16" else ""
        self.emit(f"HMMA.16816.F32{suffix} {{d0}}, {{s0}}, {{s1}}, {{s2}}",
                  [dd], [aa, bv[0], cc, bv[1]])
        if not backend.RELAX_TENSOR:
            self.emit("NOP", side=True)
        for name, value in zip(names, dd):
            if self.has_cfg:
                self.env[name] = value
            else:
                self.bind(name, value)


def assert_allocation(ops, r1, phys):
    """Validate R1 ABI, implicit-group alignment and every liveness overlap."""
    assert phys[r1] == 1
    groups = ra._groups(ops)
    start, end = ra._intervals(ops)
    for group in set(groups.values()):
        base = phys[group[0]]
        assert base % len(group) == 0
        assert [phys[v] for v in group] == list(range(base, base+len(group)))
    by_reg = {}
    for v, reg in phys.items():
        assert v == r1 or reg != 1
        if v not in start:
            continue
        for other in by_reg.setdefault(reg, []):
            # End == start is safe read-before-write, as in the core allocator.
            assert end[other] <= start[v] or end[v] <= start[other], (v, other, reg)
        by_reg[reg].append(v)


def biased_alloc(ops, r1, hints, max_reg, *, role_bias=0):
    """Core allocation invariants with optional tensor operand base preference.

    Residue classes are empirical search dimensions, not a claim that a
    particular register-bank model is proven. Copy affinities retain priority.
    """
    grp = ra._groups(ops)
    start, end = ra._intervals(ops)
    src = {op.defs[0]: op.uses[0] for op in ops
           if op.copy and isinstance(op.uses[0], ra.V)}
    roles = {}
    for op in ops:
        if op.fmt.startswith("HMMA"):
            for role, value in (("a", op.uses[0]), ("b", op.uses[1]), ("c", op.defs[0])):
                values = value if isinstance(value, tuple) else (value,)
                for v in values:
                    if isinstance(v, ra.V):
                        roles[grp[v]] = role
    groups = sorted(set(grp.values()), key=lambda g: (min(start[v] for v in g), -len(g)))
    phys, last_end = {r1: 1}, {1: 10**9}
    for group in groups:
        if group == (r1,):
            continue
        width = len(group)
        prefs = []
        for i, v in enumerate(group):
            if v in hints:
                prefs.append(hints[v]-i)
            if v in src and src[v] in phys:
                prefs.append(phys[src[v]]-i)
        bases = list(range(0, max_reg-width+1, width))
        role = roles.get(group)
        if role and role_bias:
            residue = {"a": 0, "b": 4 if role_bias == 1 else 8,
                       "c": 8 if role_bias == 1 else 4}[role]
            # Prefer nearby alternate quads; never force a new occupancy tier.
            bases.sort(key=lambda b: (b//16, (b-residue) % 16))
        for base in [x for x in prefs if x >= 0 and x % width == 0] + bases:
            if base + width > max_reg:
                continue
            if all(last_end.get(base+i, -1) <= start[v] for i, v in enumerate(group)):
                for i, v in enumerate(group):
                    phys[v], last_end[base+i] = base+i, end[v]
                break
        else:
            raise ra.NotSupported("role-aware search out of registers")
    assert_allocation(ops, r1, phys)
    return phys


@contextmanager
def compiler_variant(*, paired_b=False, role_bias=0, permute=0):
    old_selector, old_alloc, old_allocate = backend.TensorSelector, ra._alloc, ra.allocate
    if paired_b:
        backend.TensorSelector = HalfBSelector
    if role_bias:
        ra._alloc = lambda ops, r1, hints, max_reg: biased_alloc(
            ops, r1, hints, max_reg, role_bias=role_bias)
    if permute:
        def allocate(ops, r1, max_reg=250):
            phys = old_allocate(ops, r1, max_reg)
            peak = max(phys.values()) + 1
            quads = (peak+3)//4
            # Keep quad zero (and ABI R1) fixed. Permute only whole quads;
            # all implicit pair/quad alignment survives without new registers.
            span = quads-1
            shift = permute % span
            mapped = {v: (reg if reg < 4 else
                          4*(1+(reg//4-1+shift) % span)+(reg % 4))
                      for v, reg in phys.items()}
            assert_allocation(ops, r1, mapped)
            return mapped
        ra.allocate = allocate
    try:
        yield
    finally:
        backend.TensorSelector, ra._alloc, ra.allocate = old_selector, old_alloc, old_allocate


def configure(gap=8, reuse=True):
    backend.OPTIMIZE_LOCALS = True
    backend.HOIST_LOOP_INVARIANTS = False
    backend.MINED_SCALAR = True
    backend.RELAX_TENSOR = True
    backend.TENSOR_ISSUE_GAP = gap
    backend.LDSM_READ_BARRIER = False
    backend.INLINE_POINTER_INCREMENTS = True
    backend.TENSOR_REUSE = reuse
    backend.LIST_SCHEDULE = False
    backend.EXPERIMENTAL_RECYCLE_WITHOUT_WAIT = False


def compile_variant(name, m, n, k, dtype, *, paired_b=False, overlap=False,
                    column=True, role_bias=0, permute=0, gap=8, reuse=True,
                    swizzle_a=7, swizzle_b=7, paired_preaddress=False):
    configure(gap, reuse)
    source = make_source(m, n, k, dtype, paired_b=paired_b, overlap=overlap,
                         column=column, swizzle_a=swizzle_a, swizzle_b=swizzle_b,
                         paired_preaddress=paired_preaddress)
    ptx = HERE / f"search_{name}.ptx"
    ptx.write_text(source)
    output = BASE / "build" / f"search_{name}"
    backend.STAGE_RECORDS.clear()
    started = time.perf_counter()
    from openassembler import build
    original_generate = build.generate
    captured = []
    def capture(kernel,arch,opt_level):
        result = original_generate(kernel,arch,opt_level)
        captured.append(result)
        return result
    build.generate = capture
    try:
        with compiler_variant(paired_b=paired_b, role_bias=role_bias, permute=permute):
            rc = main(["compile", str(ptx), "--out-dir", str(output), "--opt-level", "safe"])
    finally:
        build.generate = original_generate
    if rc:
        raise RuntimeError(f"{name}: compile rc={rc}")
    manifest = json.loads((output / "manifest.json").read_text())
    entry = manifest["entries"][0]
    from collections import Counter
    from openassembler import oracle
    records = oracle.parse_listing((output / entry["sass"]).read_text())
    if not records or any("???" in r.text for r in records):
        raise AssertionError("nvdisasm rejected encoded code")
    # The disassembled cubin must contain the literal OA words, including all
    # waits, yield, reuse, predicates and branch offsets. Compare tensor/memory
    # operand text independently (nvdisasm appends .reuse annotations).
    from openassembler.isa.word import control, control_valid
    if len(captured) != 1:
        raise AssertionError("expected one OpenAssembler-generated kernel")
    code,labels,words,exits,regs = captured[0]
    if [r.word for r in records[:len(words)]] != words:
        raise AssertionError("spliced cubin words differ from OA encoder output")
    def normalized(text):
        return re.sub(r"\s+", "", text.replace(".reuse", ""))
    for (want,ctl),got in zip(code,records):
        if not control_valid(control(got.word)) or ctl != control(got.word):
            raise AssertionError("invalid or changed instruction control word")
        if want.startswith(("HMMA", "LDSM", "LDGSTS", "LDGDEPBAR", "DEPBAR", "BAR")):
            if normalized(want) != normalized(got.text):
                raise AssertionError(f"tensor/memory roundtrip mismatch: {want} -> {got.text}")
    metadata = {"compiler": "OpenAssembler isolated sequence/register search", "fallback": False,
                "reference_and_ELF_container": "NVIDIA ptxas on matching PTX",
                "candidate": dict(paired_b=paired_b, overlap=overlap, column=column,
                                  role_bias=role_bias, permute=permute, gap=gap,
                                  reuse=reuse, swizzle_a=swizzle_a, swizzle_b=swizzle_b,
                                  paired_preaddress=paired_preaddress),
                "geometry": dict(m=m,n=n,k=k,bm=128,bn=128,bk=64,wm=64,wn=64,stages=2),
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "experiment_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "kernels": backend.STAGE_RECORDS.copy(),
                "sass_opcode_counts": dict(Counter(r.text.split()[0] for r in records)),
                "nvdisasm_valid": True, "nvdisasm_exact_tensor_memory_operands": True,
                "spliced_words_and_control_exact": True,
                "cli_wall_ms": (time.perf_counter()-started)*1000}
    (output / "experiment.json").write_text(json.dumps(metadata, indent=2)+"\n")
    return dict(build=output.name, name=output.name, m=m,n=n,k=k,dtype=dtype,
                tile_m=128,tile_n=128,threads=128,smem=65536)


VARIANTS = [
    ("baseline", {}),
    ("bpair", dict(paired_b=True)),
    ("bpair_preaddress", dict(paired_b=True, paired_preaddress=True)),
    ("bpair_row", dict(paired_b=True, column=False)),
    ("overlap", dict(overlap=True, column=False)),
    ("bpair_overlap", dict(paired_b=True, overlap=True, column=False)),
    ("role1", dict(role_bias=1)),
    ("role2", dict(role_bias=2)),
    ("quadshift1", dict(permute=1)),
    ("quadshift3", dict(permute=3)),
    ("bpair_role1", dict(paired_b=True, role_bias=1)),
    ("bpair_swizzle3_7", dict(paired_b=True, swizzle_a=3)),
    ("bpair_swizzle7_3", dict(paired_b=True, swizzle_b=3)),
    ("bpair_gap4", dict(paired_b=True, gap=4)),
    ("bpair_no_reuse", dict(paired_b=True, reuse=False)),
]


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--large", action="store_true")
    parser.add_argument("--only", default="")
    parser.add_argument("--dtype", choices=("f16", "bf16", "both"), default="both")
    args = parser.parse_args()
    backend.install()
    suite, failures = [], []
    m,n,k = (4096,4096,4096) if args.large else (256,384,256)
    tag = "large" if args.large else "smoke"
    for dtype in (("f16","bf16") if args.dtype == "both" else (args.dtype,)):
        configs = []
        for name, options in VARIANTS:
            if args.only and name not in args.only.split(","):
                continue
            try:
                configs.append(compile_variant(f"{tag}_{dtype}_{name}",m,n,k,dtype,**options))
            except Exception as exc:
                failures.append(dict(dtype=dtype,candidate=name,error=repr(exc)))
                print(f"REJECTED {name} {dtype}: {exc}", flush=True)
        if configs:
            root = configs[0]
            root["name"] = f"search_{tag}_{dtype}"
            root["steady_warmup_rounds"] = 16
            root["alternatives"] = configs[1:]
            suite.append(root)
    (HERE / f"search_{tag}_suite.json").write_text(json.dumps(suite,indent=2)+"\n")
    (HERE / f"search_{tag}_failures.json").write_text(json.dumps(failures,indent=2)+"\n")
    print(json.dumps({"suite":str(HERE/f"search_{tag}_suite.json"),"failures":failures},indent=2))


if __name__ == "__main__":
    cli()
