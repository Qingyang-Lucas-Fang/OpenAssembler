"""Opt-in tensor selector using the existing OpenAssembler IR/allocator/encoder.

This module never edits the imported production modules on disk. The safe
schedule is padded; optional schedules are supported by the retained A100
clocked probes, completion tests and full-output GEMM comparisons.
"""
from __future__ import annotations

import dataclasses
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from openassembler.backend import ra
from openassembler.ptx.parse import Addr, ImmI, Var, Vec

OPTIMIZE_LOCALS = False
HOIST_LOOP_INVARIANTS = False
RELAX_TENSOR = False
TENSOR_ISSUE_GAP = 4
MINED_SCALAR = False
LDSM_READ_BARRIER = True
LIST_SCHEDULE = False
TENSOR_REUSE = False
EXPERIMENTAL_RECYCLE_WITHOUT_WAIT = False
INLINE_POINTER_INCREMENTS = False
STAGE_RECORDS = []


def tensor_schedule(body, labels):
    """Source-order scheduler with SB0 reserved for cp.async completion groups.

    ptxas's LDGDEPBAR has W0; DEPBAR.LE SB0 observes this async-copy counter.
    Ordinary memory scoreboards use 1..5, and live IDs retire before recycling.
    """
    from openassembler.backend.compile import (
        MEM_ASYNC, _opname, _pdefs, _puses, _mined_stalls, _branch_pred_latency)
    from openassembler.isa.word import Control
    pending_w, pending_r = {}, {}
    out = []
    joins = set(labels.values())
    next_sb = 1
    async_group = 0
    committed_group_seen = False
    active = set()

    for i, op in enumerate(body):
        name = _opname(op.text)
        p_read = [("p", p) for p in _puses(op.text)]
        p_write = [("p", p) for p in _pdefs(op.text)]
        wait = 0
        if i in joins or name in ("BRA", "EXIT", "BAR"):
            for b in active:
                wait |= 1 << b
        for reg in op.reads + op.writes + p_read + p_write:
            if reg in pending_w:
                wait |= 1 << pending_w[reg]
        for reg in op.writes:
            if reg in pending_r:
                for b in pending_r[reg]:
                    wait |= 1 << b

        def retire(mask):
            for reg in [r for r,b in pending_w.items() if mask & (1<<b)]:
                pending_w.pop(reg)
            for reg, bars in list(pending_r.items()):
                bars.difference_update(b for b in tuple(bars) if mask & (1<<b))
                if not bars:pending_r.pop(reg)
            active.difference_update(b for b in tuple(active) if mask & (1<<b))

        retire(wait)
        def take():
            nonlocal next_sb, wait
            b = next_sb
            next_sb = 1 + next_sb % 5
            if b in active and not EXPERIMENTAL_RECYCLE_WITHOUT_WAIT:
                wait |= 1 << b
                retire(1 << b)
            active.add(b)
            return b

        wbar = rbar = 7
        if op.var_lat:
            wbar = take()
            for reg in op.writes + p_write:
                pending_w[reg] = wbar
        if name in MEM_ASYNC and op.reads and (name != "LDSM" or LDSM_READ_BARRIER):
            rbar = take()
            for reg in op.reads:
                pending_r.setdefault(reg,set()).add(rbar)
        elif name == "LDSM" and not LDSM_READ_BARRIER:
            # A completed LDSM result implies its address has been consumed.
            # Reuse its write-completion scoreboard to protect address WARs.
            # This omits the hardware read barrier without dropping the
            # dependency, including when several loads share one address reg.
            if wbar == 7:raise AssertionError("LDSM needs a completion scoreboard")
            for reg in op.reads:
                pending_r.setdefault(reg,set()).add(wbar)
        if name == "LDGDEPBAR":
            wbar = 0
            async_group += 1
            committed_group_seen = True
        if name == "DEPBAR":
            if not committed_group_seen:
                raise ra.NotSupported("DEPBAR without an experimental copy-group commit")
            threshold=int(op.text.rsplit(' ',1)[1],16)
            async_group=min(async_group,threshold)
        stall = 2 if op.var_lat else 6
        if name in ("STG", "BRA", "EXIT"):
            stall = 5
        out.append((op.text, Control(stall=stall, yield_=False, wbar=wbar, rbar=rbar, wait=wait, reuse=0)))
    if MINED_SCALAR:
        out = _mined_stalls(body, out, joins)
    # The scalar cost model treats synchronization as an ordinary no-result
    # instruction and otherwise shrinks these to S1. ptxas uses S6 after
    # deferred CTA barriers and S4 after DEPBAR in this exact SM80 family.
    # Preserve those issue gaps while testing them as separate semantics.
    for i, (text, ctl) in enumerate(out):
        minimum = {"BAR": 6, "DEPBAR": 4, "LDSM": 4 if not LDSM_READ_BARRIER else 2}.get(_opname(text), 1)
        if ctl.stall < minimum:
            out[i] = (text, dataclasses.replace(ctl, stall=minimum))
    return _branch_pred_latency(out)


def loop_carried_registers(kernel):
    """Exact liveness for our single-latch-loop PTX subset, including vector defs.

    This is intentionally an experiment in a restricted CFG, not an unvalidated
    replacement for the production backend's general guarded-definition rules.
    """
    from openassembler.ptx.parse import PairDst
    body = kernel.body
    branches = [(i, ins) for i, ins in enumerate(body) if ins.op == "bra"]
    if len(branches) != 1:
        raise ra.NotSupported("local-home optimization requires exactly one loop branch")
    latch, branch = branches[0]
    head = kernel.labels.get(branch.args[0].name, latch + 1)
    if head >= latch or branch.guard is None or any(
            ins.guard is not None and i != latch for i, ins in enumerate(body)):
        raise ra.NotSupported("local-home optimization requires a single unguarded loop body")

    def regs(operand):
        if isinstance(operand, Var):
            return {operand.name} if operand.name.startswith("%") else set()
        if isinstance(operand, Addr):
            return {operand.base} if operand.base.startswith("%") else set()
        if isinstance(operand, Vec):
            return set().union(*(regs(e) for e in operand.elems))
        if isinstance(operand, PairDst):
            return regs(operand.a) | regs(operand.b)
        return set()
    uses, defs = [], []
    for ins in body:
        writes = ins.op not in ("st", "bra", "ret", "bar", "cp", "red", "membar")
        dst = regs(ins.args[0]) if writes and ins.args else set()
        srcs = ins.args[1:] if writes else ins.args
        read = set().union(*(regs(a) for a in srcs))
        if ins.guard:
            read.add(ins.guard[0])
        uses.append(read); defs.append(dst)
    live = [set() for _ in body]
    changed = True
    while changed:
        changed = False
        for i in range(len(body) - 1, -1, -1):
            successors = [] if body[i].op == "ret" else ([i + 1] if i + 1 < len(body) else [])
            if i == latch:
                successors.append(head)
            after = set().union(*(live[j] for j in successors))
            before = uses[i] | (after - defs[i])
            if before != live[i]:
                live[i] = before
                changed = True
    return live[head] & set().union(*defs[head:latch + 1])


def hoist_loop_invariants(ops):
    """Hoist pure SSA integer address expressions out of one unconditional body.

    Fixed homes with more than one definition, predicates, memory operations,
    guards and barriers remain in place. Works on virtual IR before allocation.
    """
    import collections
    labels = {op.label: i for i, op in enumerate(ops) if op.label}
    backedges = [(i, labels[op.target]) for i, op in enumerate(ops)
                 if op.target in labels and labels[op.target] < i]
    if len(backedges) != 1:
        raise ra.NotSupported("experimental LICM requires one loop backedge")
    latch, head = backedges[0]
    counts = collections.Counter(v for op in ops for v in ra._defs(op))
    loop_defs = set().union(*(ra._defs(op) for op in ops[head:latch + 1]))
    available = {v for op in ops[:head] for v in ra._defs(op)} - loop_defs
    moved, keep = [], []
    pure = {"MOV", "IMAD", "IADD3", "LOP3", "SHF", "LEA"}
    for op in ops[head + 1:latch]:
        defs = ra._defs(op)
        uses = ra._all_uses(op)
        base = op.fmt.split(" ", 1)[0].split(".", 1)[0]
        if (base in pure and defs and not op.side and not op.var_lat and op.guard is None
                and not op.target and not op.label and not any(isinstance(v, ra.PV) for v in defs | uses)
                and all(counts[v] == 1 for v in defs) and uses <= available):
            moved.append(op)
            available |= defs
        else:
            keep.append(op)
    return ops[:head] + moved + [ops[head]] + keep + ops[latch:], len(moved)


class TensorSelector(ra.Selector):
    def _init_tensor_homes(self):
        if getattr(self, "_tensor_homes_ready", False):
            return
        self._tensor_homes_ready = True
        self.tensor_groups = {}
        if OPTIMIZE_LOCALS:
            carried = loop_carried_registers(self.k)
            self.homes = {name: value for name, value in self.homes.items() if name in carried}
        if not self.has_cfg:
            return
        for ins in self.k.body:
            if ins.op != "mma":
                continue
            if not ins.args or not isinstance(ins.args[0], Vec):
                raise ra.NotSupported("MMA destination must be a register vector")
            names = tuple(v.name for v in ins.args[0].elems if isinstance(v, Var))
            if len(names) != 4 or len(set(names)) != 4:
                raise ra.NotSupported("MMA needs four distinct accumulator destinations")
            for name in names:
                old = self.tensor_groups.setdefault(name, names)
                if old != names:
                    raise ra.NotSupported("overlapping MMA accumulator groups")
                if self.homes.get(name) is None:
                    self.homes[name] = self.new()

    def select(self, ins):
        self._init_tensor_homes()
        if (INLINE_POINTER_INCREMENTS and ins.guard is None and ins.op=='add' and ins.mods in (('u64',),('s64',))
                and len(ins.args)==3 and isinstance(ins.args[0],Var) and ins.args[0]==ins.args[1]
                and isinstance(ins.args[2],ImmI) and 0<=ins.args[2].value<2**31):
            name=ins.args[0].name
            pair=self.pair(name)
            if self.homes.get(name)==pair and all(isinstance(v,ra.V) for v in pair):
                lo,hi=pair;carry=self.newp()
                self.emit('IADD3 {d0}, {p0}, {s0}, {s1}, RZ',[lo,carry],[lo,ra.I(ins.args[2].value)])
                self.emit('IADD3.X {d0}, {s0}, RZ, RZ, {q0}, !PT',[hi],[hi,(carry,False)])
                self.env[name]=pair
                return
        if ins.op == "mma":
            return self.mma(ins)
        if ins.op == "ldmatrix":
            return self.ldmatrix(ins)
        if ins.op == "cp":
            return self.async_copy(ins)
        return super().select(ins)

    def async_copy(self, ins):
        if ins.guard is not None:
            raise ra.NotSupported("experimental cp.async must be unguarded")
        if ins.mods == ("async", "commit_group") and not ins.args:
            self.emit("LDGDEPBAR", side=True)
            return
        if (ins.mods == ("async", "wait_group") and len(ins.args) == 1
                and isinstance(ins.args[0], ImmI) and ins.args[0].value in (0,1,2)):
            self.emit(f"DEPBAR.LE SB0, {ins.args[0].value:#x}", side=True)
            return
        if (ins.mods != ("async", "cg", "shared", "global") or len(ins.args) != 3
                or not all(isinstance(x, Addr) and x.off == 0 for x in ins.args[:2])
                or not isinstance(ins.args[2], ImmI) or ins.args[2].value != 16):
            raise ra.NotSupported("only cp.async.cg.shared.global of 16 bytes and wait_group 0/1/2 are implemented")
        shared = self.reg32(Var(ins.args[0].base))
        source = self.wide(self.pair(ins.args[1].base))
        self.emit("LDGSTS.E.BYPASS.128 [{s0}], [{s1}.64]", [], [shared, source], side=True)

    def mma(self, ins):
        # Reject every unimplemented shape, type and layout, rather than silently
        # discarding a modifier with arithmetic or synchronization semantics.
        mods = ins.mods
        prefix = ("sync", "aligned", "m16n8k16", "row", "col", "f32")
        if (mods[:6] != prefix or len(mods) != 9 or mods[6] not in ("f16", "bf16")
                or mods[7] != mods[6] or mods[8] != "f32" or ins.guard is not None):
            raise ra.NotSupported(f"experimental MMA modifiers/guard: {mods}")
        if len(ins.args) != 4 or any(not isinstance(x, Vec) for x in ins.args):
            raise ra.NotSupported("MMA requires D/A/B/C register vectors")
        d, a, b, c = ins.args
        if tuple(len(x.elems) for x in (d, a, b, c)) != (4, 4, 2, 4):
            raise ra.NotSupported("m16n8k16 fragment widths must be 4/4/2/4")
        if any(not isinstance(x, Var) for x in d.elems):
            raise ra.NotSupported("MMA destination must contain registers")
        aa = self.wide(tuple(self.reg32(v) for v in a.elems))
        bb = self.wide(tuple(self.reg32(v) for v in b.elems))
        names = tuple(v.name for v in d.elems)
        if self.has_cfg:
            dd = tuple(self.homes[n] for n in names)
            cc0 = tuple(self.reg32(v) for v in c.elems)
            cc = cc0 if cc0 == dd else self.wide(cc0)
        else:
            dd = tuple(self.new() for _ in range(4))
            cc = self.wide(tuple(self.reg32(v) for v in c.elems))
        suffix = ".BF16" if mods[6] == "bf16" else ""
        self.emit(f"HMMA.16816.F32{suffix} {{d0}}, {{s0}}, {{s1}}, {{s2}}",
                  [dd], [aa, bb, cc])
        # A real instruction, with a real PC: labels are resolved after padding.
        if not RELAX_TENSOR:
            self.emit("NOP", side=True)
        for name, value in zip(names, dd):
            if self.has_cfg:
                self.env[name] = value
            else:
                self.bind(name, value)

    def ldmatrix(self, ins):
        allowed = {
            ("sync", "aligned", "m8n8", "x4", "shared", "b16"): (4, "M88"),
            ("sync", "aligned", "m8n8", "x2", "shared", "b16"): (2, "M88"),
            ("sync", "aligned", "m8n8", "x2", "trans", "shared", "b16"): (2, "MT88"),
        }
        if ins.mods not in allowed or ins.guard is not None:
            raise ra.NotSupported(f"experimental LDSM modifiers/guard: {ins.mods}")
        width, layout = allowed[ins.mods]
        if (len(ins.args) != 2 or not isinstance(ins.args[0], Vec)
                or len(ins.args[0].elems) != width or not isinstance(ins.args[1], Addr)):
            raise ra.NotSupported("ldmatrix fragment/address")
        ad = ins.args[1]
        if ad.off < 0 or ad.off % 16 or ad.off >= 16384:
            raise ra.NotSupported("experimental ldmatrix offset must be aligned and below 16 KiB")
        address = self.reg32(Var(ad.base))
        group = tuple(self.new() for _ in range(width))
        offset = f"+{ad.off:#x}" if ad.off else ""
        self.emit(f"LDSM.16.{layout}.{width} {{d0}}, [{{s0}}{offset}]", [group], [address], var_lat=True)
        for dst, value in zip(ins.args[0].elems, group):
            if not isinstance(dst, Var):
                raise ra.NotSupported("ldmatrix destinations must be registers")
            self.bind(dst.name, value)


def prepare():
    """Redirect *all* discovery writes before instantiating the base encoder."""
    from openassembler import oracle
    from openassembler.isa import encode
    HERE.joinpath("scratch").mkdir(exist_ok=True)
    oracle.SCRATCH = HERE / "scratch"
    cache = HERE / "fieldmaps.json"
    if not cache.exists():
        shutil.copyfile(encode.FIELDMAP_CACHE, cache)
    encode.FIELDMAP_CACHE = cache
    encode.SYNTH_CACHE = HERE / "templates.json"


_ENCODER = None


class CopyEncoder:
    """The base encoder cannot encode two independently varying memory operands.

    The narrow LDGSTS form below uses register fields recovered and checked by
    nvdisasm bit probes. Offsets, predicates, scopes and widths are not guessed.
    """
    def __init__(self, base):
        from openassembler import oracle
        from openassembler.isa import encode
        from openassembler.emu.program import parse_insn
        self.base = base
        self.seed = base.templates[encode.key(parse_insn(0, "LDGSTS.E.BYPASS.128 [R7], [R2.64]", 0))].word
        fields = encode.field_probe(self.seed)
        if (any(fields.get(b) != "op0" for b in range(16, 24))
                or any(fields.get(b) != "op1" for b in range(24, 32))):
            raise encode.EncodeError("LDGSTS register field probes disagree with the experiment")
        variants = [(0, 8), (9, 12), (31, 24)]
        words = [encode.set_bits(encode.set_bits(self.seed, 16, 8, d), 24, 8, s) for d, s in variants]
        expected = [f"LDGSTS.E.BYPASS.128 [R{d}], [R{s}.64]" for d, s in variants]
        actual = oracle.disasm_words(words)
        if actual != expected:
            raise encode.EncodeError(f"LDGSTS round trip failed: {actual}")
        (HERE / "ldgsts-encoding-evidence.json").write_text(json.dumps({
            "seed": f"{self.seed:032x}", "bit_to_operand": fields,
            "roundtrips": [{"word": f"{w:032x}", "text": text} for w, text in zip(words, actual)]
        }, indent=2) + "\n")

    def __getattr__(self, name):
        return getattr(self.base, name)

    def encode(self, text, ctl=None):
        if text in ('DEPBAR.LE SB0, 0x1','DEPBAR.LE SB0, 0x2'):
            from openassembler.isa.word import set_bits
            # phase2/depbar-fields.json: bits 38..43 independently encode the
            # counter threshold; each variant is also round-tripped at setup.
            seed=self.base.encode('DEPBAR.LE SB0, 0x0')
            word=set_bits(seed,38,6,int(text.rsplit(' ',1)[1],16))
            return self.base._ctl(word,ctl) if ctl is not None else word
        if not text.startswith("LDGSTS."):
            return self.base.encode(text, ctl)
        from openassembler.isa import encode
        from openassembler.emu.program import parse_insn
        from openassembler.emu.operands import Mem
        ins = parse_insn(0, text, 0)
        if (ins.guard is not None or ins.op != "LDGSTS" or ins.mods != ("E", "BYPASS", "128")
                or len(ins.ops) != 2):
            raise encode.EncodeError("unsupported experimental async-copy encoding")
        dst, src = ins.ops
        if (not all(isinstance(v, Mem) and v.base is not None and v.base.file == "R"
                    and v.ur is None and v.off == 0 and v.scale == 1 for v in (dst, src))
                or dst.wide or not src.wide):
            raise encode.EncodeError("LDGSTS requires simple [shared R], [global R.64] operands")
        word = encode.set_bits(encode.set_bits(self.seed, 16, 8, dst.base.idx), 24, 8, src.base.idx)
        return self.base._ctl(word, ctl) if ctl is not None else word


def encoder():
    global _ENCODER
    if _ENCODER is not None:
        return _ENCODER
    prepare()
    from openassembler.isa import encode
    from openassembler.emu.program import parse_insn
    enc = encode.Encoder("sm_80")
    if encode.SYNTH_CACHE.exists():
        for row in json.loads(encode.SYNTH_CACHE.read_text()):
            ins = parse_insn(0, row["text"], int(row["word"], 16))
            enc.templates[encode.key(ins)] = encode.Encoder._learn([ins], canonical=True)
    wanted = ["LDSM.16.M88.2 R8, [R0]"]
    encode.synthesize(enc, wanted)
    _ENCODER = CopyEncoder(enc)
    from openassembler import oracle
    for threshold in (1,2):
        text=f'DEPBAR.LE SB0, {threshold:#x}'
        if oracle.disasm_words([_ENCODER.encode(text)])[0]!=text:
            raise encode.EncodeError('DEPBAR threshold roundtrip failed')
    return _ENCODER


def generate(kernel, arch, opt_level):
    from openassembler.backend.compile import Out, schedule
    from openassembler.container.cubin import required_regcount
    from openassembler.isa.encode import EncodeError
    if arch != "sm_80":
        raise ra.NotSupported("the isolated tensor compiler only targets sm_80")
    times = {}
    stamp = time.perf_counter()
    def lap(name):
        nonlocal stamp
        now = time.perf_counter()
        times[name] = (now - stamp) * 1000
        stamp = now
    s = TensorSelector(kernel)
    raw_ops = s.run()
    lap("select_ms")
    hoisted = 0
    if HOIST_LOOP_INVARIANTS:
        raw_ops, hoisted = hoist_loop_invariants(raw_ops)
    lap("licm_ms")
    ops = ra.coalesce(ra.dce(raw_ops))
    lap("dce_coalesce_ms")
    phys = ra.allocate(ops, s.r1)
    lap("allocate_ms")
    lowered = ra.lower(ops, phys)
    if LIST_SCHEDULE:
        from openassembler.backend import sched
        old_ordered = sched.ORDERED
        try:
            sched.ORDERED = old_ordered + ("HMMA", "LDGDEPBAR")
            lowered = sched.list_schedule(lowered)
        finally:
            sched.ORDERED = old_ordered
    lap("lower_ms")
    if RELAX_TENSOR:
        drained = []
        for i, item in enumerate(lowered):
            drained.append(item)
            if item[0].startswith("HMMA") and (i + 1 == len(lowered)
                                               or not lowered[i + 1][0].startswith("HMMA")):
                # Chain interlocks do not protect ordinary observers: the
                # result-to-STG.128 sweep failed at encoded gaps 2..14 and
                # passed at 15+. Drain each tensor epoch for at least 17
                # encoded cycles (HMMA S2 + NOP S15) before any scalar op,
                # register reuse, memory op, label or branch can follow.
                drained.append(("NOP", False, [], []))
        lowered = drained
    # Keep source order for this initial experiment. Do not run the scalar list
    # scheduler, which has no tensor-pipe or async completion cost model yet.
    labels, body = {}, []
    for text, variable, reads, writes in lowered:
        if text.startswith("<label "):
            labels[text[7:-1]] = len(body)
        else:
            body.append(Out(text, variable, reads, writes))
    code = tensor_schedule(body, labels)
    for i, (text, ctl) in enumerate(code):
        if text.startswith("HMMA"):
            if RELAX_TENSOR:
                # The accumulator-chain probe checks encoded gap 2. Tensor
                # epochs are drained separately for ordinary observers.
                # This is not a two-cycle HMMA execution-latency claim.
                code[i] = (text, dataclasses.replace(ctl, stall=TENSOR_ISSUE_GAP, yield_=TENSOR_ISSUE_GAP > 11))
                continue
            code[i] = (text, dataclasses.replace(ctl, stall=15, yield_=True))
            if i + 1 >= len(code) or code[i + 1][0] != "NOP":
                raise AssertionError("lost HMMA timing padding")
            nt, nc = code[i + 1]
            code[i + 1] = (nt, dataclasses.replace(nc, stall=15, yield_=True))
        elif RELAX_TENSOR and text == "NOP":
            code[i] = (text, dataclasses.replace(ctl, stall=15, yield_=True))
    if TENSOR_REUSE:
        from openassembler.emu.program import parse_insn
        for i in range(len(code)-1):
            current = parse_insn(0,code[i][0],0)
            following = parse_insn(0,code[i+1][0],0)
            if current.op != "HMMA" or following.op != "HMMA":
                continue
            reuse = 0
            written = set(body[i].writes)
            for operand, width in ((1,4),(2,2)):
                old,new = current.ops[operand],following.ops[operand]
                if old == new and not written.intersection(range(old.idx,old.idx+width)):
                    reuse |= 1 << (operand-1)
            text,ctl=code[i]
            code[i]=(text,dataclasses.replace(ctl,reuse=reuse))
    lap("schedule_ms")
    enc = encoder()
    lap("encoder_setup_ms")
    words = ra.encode_code(enc, code, labels)
    words.append(enc.encode_branch("BRA `(.L_x_0)", 16 * len(words), 16 * len(words)))
    while len(words) % 8:
        words.append(enc.encode("NOP"))
    lap("encode_ms")
    STAGE_RECORDS.append({"kernel": kernel.name, "requested_opt_level": opt_level,
                          "applied_schedule": (f"source-order-HMMA{TENSOR_ISSUE_GAP}-epoch-drain15" if RELAX_TENSOR
                                               else "source-order-safe-HMMA30"),
                          "scalar_schedule": "mined" if MINED_SCALAR else "safe",
                          "protected_scoreboard_recycling": True,
                          "ldsm_read_barrier": LDSM_READ_BARRIER,
                          "ldsm_address_war_uses_write_completion": not LDSM_READ_BARRIER,
                          "list_schedule_with_tensor_ordering": LIST_SCHEDULE,
                          "tensor_reuse_hints": TENSOR_REUSE,
                          "experimental_recycle_without_wait": EXPERIMENTAL_RECYCLE_WITHOUT_WAIT,
                          "inline_64bit_pointer_increments": INLINE_POINTER_INCREMENTS,
                          "optimize_local_homes": OPTIMIZE_LOCALS,
                          "licm_ops_hoisted": hoisted,
                          "ptx_instructions": len(kernel.body), "sass_instructions": len(code),
                          "registers": required_regcount([t for t, _ in code]), "stages": times})
    exits = [16 * i for i, (text, _) in enumerate(code) if text.endswith("EXIT")]
    return code, labels, words, exits, required_regcount([t for t, _ in code])


def install():
    """Opt in only for this process; no package or specification file is rewritten."""
    from openassembler import build
    prepare()
    build.generate = generate

