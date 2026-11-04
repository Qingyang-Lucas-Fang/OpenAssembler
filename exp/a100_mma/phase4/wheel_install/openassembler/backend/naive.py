"""M3 v0 (kept for A/B comparison): naive PTX -> SASS compiler, one register per PTX vreg.

Pipeline: parse PTX -> select SASS per PTX instruction -> assign one physical
register per PTX virtual register (no reuse; the probes are tiny) -> conservative
scheduling (control bits from the mined model, scoreboards for every variable-
latency op) -> template encoder -> splice into a ptxas-built cubin of the same
kernel (we reuse its ELF/.nv.info skeleton and fix REGCOUNT and EXIT offsets).

Coverage is deliberately small. Unsupported PTX raises NotSupported.
Includes our own strength reduction: div/rem by an immediate uses the exhaustively
proved multiply-high sequences from openassembler.opt.divconst.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from openassembler.isa.word import Control
from openassembler.opt import divconst
from openassembler.ptx.parse import Addr, ImmI, Instr, Kernel, Var, Vec

PARAM_BASE = 0x160

# PTX special register -> (S2R name | c[0x0] offset). Offsets from docs/research/02.
SREG = {"%tid.x": ("s2r", "SR_TID.X"), "%tid.y": ("s2r", "SR_TID.Y"), "%tid.z": ("s2r", "SR_TID.Z"),
        "%ctaid.x": ("s2r", "SR_CTAID.X"), "%ctaid.y": ("s2r", "SR_CTAID.Y"), "%ctaid.z": ("s2r", "SR_CTAID.Z"),
        "%laneid": ("s2r", "SR_LANEID"), "%lanemask_eq": ("s2r", "SR_EQMASK"),
        "%lanemask_lt": ("s2r", "SR_LTMASK"), "%lanemask_ge": ("s2r", "SR_GEMASK"),
        "%ntid.x": ("c0", 0x0), "%ntid.y": ("c0", 0x4), "%ntid.z": ("c0", 0x8),
        "%nctaid.x": ("c0", 0xC), "%nctaid.y": ("c0", 0x10), "%nctaid.z": ("c0", 0x14)}


class NotSupported(Exception):
    pass


@dataclass
class Out:
    text: str
    var_lat: bool = False   # variable latency: needs a write scoreboard
    reads: list = field(default_factory=list)   # physical regs read
    writes: list = field(default_factory=list)  # physical regs written


class Compiler:
    def __init__(self, k: Kernel):
        self.k = k
        self.reg: dict[str, int] = {}
        self.next = 2           # R0/R1 reserved (R1 = stack pointer by convention)
        self.code: list[Out] = []
        self.param_off = {}
        off = 0
        for ty, name in k.params:
            sz = 8 if ty.endswith("64") else 4
            off = (off + sz - 1) // sz * sz
            self.param_off[name] = PARAM_BASE + off
            off += sz
        self.ptr_regs: set[str] = set()

    # ---- registers ---------------------------------------------------------------

    def r(self, name: str, wide: bool = False) -> int:
        if name not in self.reg:
            if wide and self.next % 2:
                self.next += 1
            self.reg[name] = self.next
            self.next += 2 if wide else 1
            if self.next > 250:
                raise NotSupported("out of registers")
        return self.reg[name]

    def opnd(self, op) -> str:
        if isinstance(op, ImmI):
            return f"{op.value & 0xFFFFFFFF:#x}"
        if isinstance(op, Var):
            if op.name.startswith("%") and op.name not in self.reg and op.name.split(".")[0] in (
                    "%tid", "%ntid", "%ctaid", "%nctaid", "%laneid", "%warpid", "%smid", "%clock", "%clock64"):
                raise NotSupported(f"special register operand {op.name}")
            if op.name not in self.reg and not self._defined_later(op.name):
                raise NotSupported(f"read of undefined register {op.name}")
            return f"R{self.r(op.name)}"
        raise NotSupported(f"operand {op}")

    def _defined_later(self, name: str) -> bool:
        return False

    def emit(self, text, var_lat=False, reads=(), writes=()):
        self.code.append(Out(text, var_lat, list(reads), list(writes)))

    # ---- instruction selection --------------------------------------------------------

    def compile(self) -> list[Out]:
        self.emit("MOV R1, c[0x0][0x28]", writes=[1])
        self.emit("ULDC.64 UR4, c[0x0][0x118]")  # global memory descriptor
        for ins in self.k.body:
            if ins.guard is not None:
                raise NotSupported("predicated PTX")
            self.select(ins)
        return self.code

    def select(self, ins: Instr) -> None:
        op, m, a = ins.op, ins.mods, ins.args
        t = next((x for x in m if x[:1] in ("s", "u", "b", "f") and x[1:].isdigit()), None)
        if op == "ld" and "param" in m:
            d = a[0].name
            off = self.param_off[a[1].base] + a[1].off
            if t in ("u64", "b64", "s64"):
                rd = self.r(d, wide=True)
                self.emit(f"MOV R{rd}, c[0x0][{off:#x}]", writes=[rd])
                self.emit(f"MOV R{rd + 1}, c[0x0][{off + 4:#x}]", writes=[rd + 1])
            else:
                rd = self.r(d)
                self.emit(f"MOV R{rd}, c[0x0][{off:#x}]", writes=[rd])
            return
        if op == "cvta":
            d, s = a[0].name, a[1].name
            self.reg[d] = self.r(s, wide=True)  # global generic == global: alias
            return
        if op == "mov" and isinstance(a[1], Var) and a[1].name.startswith("%") and a[1].name in SREG:
            rd = self.r(a[0].name)
            how, what = SREG[a[1].name]
            if how == "s2r":
                self.emit(f"S2R R{rd}, {what}", var_lat=True, writes=[rd])
            else:  # launch constant in c[0x0]
                self.emit(f"MOV R{rd}, c[0x0][{what:#x}]", writes=[rd])
            return
        if op == "mov" and isinstance(a[1], Var) and a[1].name.startswith("%") and a[1].name not in self.reg:
            raise NotSupported(f"special register {a[1].name}")
        if op == "mov" and isinstance(a[1], Vec):  # pack {lo, hi} into 64-bit
            rd = self.r(a[0].name, wide=True)
            lo, hi = (self.r(e.name) for e in a[1].elems)
            self.emit(f"MOV R{rd}, R{lo}", reads=[lo], writes=[rd])
            self.emit(f"MOV R{rd + 1}, R{hi}", reads=[hi], writes=[rd + 1])
            return
        if op == "mov" and t in ("b64", "u64", "f64") and isinstance(a[0], Var) and isinstance(a[1], Var):  # alias
            # Only alias a register that already holds a value. A symbol (`mov.u64 %rd, smem`)
            # or never-written name would alias garbage: found on A100 (cvta.shared probe).
            if a[1].name not in self.reg:
                raise NotSupported(f"64-bit mov from symbol/undefined {a[1].name}")
            self.reg[a[0].name] = self.reg[a[1].name]
            return
        if op == "mov" and isinstance(a[0], Vec):  # unpack 64-bit into {lo, hi}
            rs = self.r(a[1].name, wide=True)
            lo, hi = (self.r(e.name) for e in a[0].elems)
            self.emit(f"MOV R{lo}, R{rs}", reads=[rs], writes=[lo])
            self.emit(f"MOV R{hi}, R{rs + 1}", reads=[rs + 1], writes=[hi])
            return
        if op == "mov" and t in ("u32", "b32", "s32", "f32"):
            rd = self.r(a[0].name)
            src = self.opnd(a[1])
            self.emit(f"MOV R{rd}, {src}", reads=self._regs(a[1]), writes=[rd])
            return
        if op == "mul" and "wide" in m:
            rd = self.r(a[0].name, wide=True)
            sfx = ".U32" if t == "u32" else ""
            self.emit(f"IMAD.WIDE{sfx} R{rd}, {self.opnd(a[1])}, {self.opnd(a[2])}, RZ",
                      reads=self._regs(a[1], a[2]), writes=[rd, rd + 1])
            return
        if op == "add" and t in ("s64", "u64"):
            rd = self.r(a[0].name, wide=True)
            x, y = self.r(a[1].name, wide=True), self.r(a[2].name, wide=True)
            self.emit(f"IADD3 R{rd}, P0, R{x}, R{y}, RZ", reads=[x, y], writes=[rd])
            self.emit(f"IADD3.X R{rd + 1}, R{x + 1}, R{y + 1}, RZ, P0, !PT", reads=[x + 1, y + 1], writes=[rd + 1])
            return
        if op == "ld" and "global" in m:
            ad = a[1]
            base = self.r(ad.base, wide=True)
            off = f"+{ad.off:#x}" if ad.off else ""
            if "v4" in m:
                regs = [self.r(e.name) for e in a[0].elems]
                if regs != list(range(regs[0], regs[0] + 4)) or regs[0] % 4:
                    regs = self._fresh_quad(a[0].elems)
                self.emit(f"LDG.E.128 R{regs[0]}, [R{base}.64{off}]", var_lat=True,
                          reads=[base, base + 1], writes=regs)
                return
            if "v2" in m or "v8" in m:
                raise NotSupported("ld.global.v2")
            sfx = {"u8": ".U8", "s8": ".S8", "u16": ".U16", "s16": ".S16", "b32": "", "u32": "", "s32": "",
                   "f32": "", "b64": ".64", "u64": ".64", "s64": ".64", "f64": ".64"}.get(t)
            if sfx is None:
                raise NotSupported(f"ld.global.{t}")
            caching = [x for x in m if x in ("nc", "cg", "cs", "lu", "ca", "cv", "volatile")]
            if caching:
                raise NotSupported(f"cache operator .{caching[0]}")  # needs its own encoding (CONSTANT/EF/LU...)
            rd = self.r(a[0].name, wide=sfx == ".64")
            self.emit(f"LDG.E{sfx} R{rd}, [R{base}.64{off}]", var_lat=True, reads=[base, base + 1],
                      writes=[rd, rd + 1] if sfx == ".64" else [rd])
            return
        if op == "st" and "global" in m:
            ad, src = a[0], a[1]
            base = self.r(ad.base, wide=True)
            off = f"+{ad.off:#x}" if ad.off else ""
            if "v2" in m:
                lo, hi = (self.r(e.name) for e in src.elems)
                pair = self.next + (self.next % 2)
                self.next = pair + 2
                self.emit(f"MOV R{pair}, R{lo}", reads=[lo], writes=[pair])
                self.emit(f"MOV R{pair + 1}, R{hi}", reads=[hi], writes=[pair + 1])
                self.emit(f"STG.E.64 [R{base}.64{off}], R{pair}", reads=[base, base + 1, pair, pair + 1])
                return
            if t not in ("b32", "u32", "s32", "f32"):
                raise NotSupported(f"st.global.{t}")
            self.emit(f"STG.E [R{base}.64{off}], {self.opnd(src)}", reads=[base, base + 1] + self._regs(src))
            return
        if op == "ret":
            self.emit("EXIT")
            return
        if t in ("u32", "s32", "b32"):
            return self.alu32(ins, t)
        raise NotSupported(f"{op}.{'.'.join(m)}")

    def _fresh_quad(self, elems):
        base = (self.next + 3) // 4 * 4
        self.next = base + 4
        for k, e in enumerate(elems):
            self.reg[e.name] = base + k
        return [base + k for k in range(4)]

    def _regs(self, *ops):
        return [self.r(o.name) for o in ops if isinstance(o, Var)]

    def alu32(self, ins, t):
        op, a = ins.op, ins.args
        d = self.r(a[0].name)
        srcs = a[1:]
        x = self.opnd(srcs[0]) if srcs else None
        y = self.opnd(srcs[1]) if len(srcs) > 1 else None
        rr = self._regs(*srcs)
        signed = t == "s32"
        if op in ("div", "rem") and isinstance(srcs[1], ImmI):
            return self.divconst(ins, d, srcs[0], srcs[1].value, signed)
        imm_neg = lambda o: f"{(-o.value) & 0xFFFFFFFF:#x}"  # noqa: E731
        table = {
            "add": lambda: f"IADD3 R{d}, {x}, {y}, RZ",
            "sub": lambda: (f"IADD3 R{d}, {x}, {imm_neg(srcs[1])}, RZ" if isinstance(srcs[1], ImmI)
                            else f"IADD3 R{d}, {x}, -{y}, RZ"),
            "and": lambda: f"LOP3.LUT R{d}, {x}, {y}, RZ, 0xc0, !PT",
            "or": lambda: f"LOP3.LUT R{d}, {x}, {y}, RZ, 0xfc, !PT",
            "xor": lambda: f"LOP3.LUT R{d}, {x}, {y}, RZ, 0x3c, !PT",
            "not": lambda: f"LOP3.LUT R{d}, RZ, {x}, RZ, 0x33, !PT",
            "neg": lambda: f"IADD3 R{d}, RZ, -{x}, RZ",
            "min": lambda: f"IMNMX{'' if signed else '.U32'} R{d}, {x}, {y}, PT",
            "max": lambda: f"IMNMX{'' if signed else '.U32'} R{d}, {x}, {y}, !PT",
        }
        if op == "mul" and "lo" in ins.mods:
            self.emit(f"IMAD R{d}, {x}, {y}, RZ", reads=rr, writes=[d])
            return
        if op == "mul" and "hi" in ins.mods:
            self.emit(f"IMAD.HI{'' if signed else '.U32'} R{d}, {x}, {y}, RZ", reads=rr, writes=[d])
            return
        if op == "mad" and "lo" in ins.mods:
            self.emit(f"IMAD R{d}, {x}, {y}, {self.opnd(srcs[2])}", reads=rr, writes=[d])
            return
        if op in ("shl", "shr") and isinstance(srcs[1], ImmI) and srcs[1].value >= 32:
            raise NotSupported("shift >= 32")
        if op == "shl" and isinstance(srcs[1], ImmI) and srcs[1].value < 32:
            self.emit(f"SHF.L.U32 R{d}, {x}, {srcs[1].value:#x}, RZ", reads=rr, writes=[d])
            return
        if op == "shr" and isinstance(srcs[1], ImmI) and srcs[1].value < 32:
            kind = "S32" if signed else "U32"
            self.emit(f"SHF.R.{kind}.HI R{d}, RZ, {srcs[1].value:#x}, {x}", reads=rr, writes=[d])
            return
        if op in table:
            self.emit(table[op](), reads=rr, writes=[d])
            return
        raise NotSupported(f"{op}.{t}")

    def divconst(self, ins, d, x_op, divisor, signed):
        if divisor == 0 or (signed and not 2 <= divisor < (1 << 31)) or divisor < 2:
            raise NotSupported("div by 0/1/negative constant")
        seq = divconst.lower(ins.op, signed, divisor)
        xr = self.r(x_op.name)
        names = {"x": f"R{xr}"}
        for o in seq.ops:
            k, dst = o[0], o[1]
            if dst not in names:
                names[dst] = f"R{self.next}"
                self.next += 1
        v = lambda o: f"{o & 0xFFFFFFFF:#x}" if isinstance(o, int) else names[o]  # noqa: E731
        tmp = f"R{self.next}"
        self.next += 1
        for o in seq.ops:
            k, dst = o[0], names[o[1]]
            s = o[2:]
            if k == "lea_hi_sign":
                # q = t + (t >>> 31) as SHF + IADD3 (same result as LEA.HI, proven by the
                # same exhaustive check since openassembler.opt.divconst.run defines the op).
                self.emit(f"SHF.R.U32.HI {tmp}, RZ, 0x1f, {v(s[1])}", reads=[int(v(s[1])[1:])], writes=[int(tmp[1:])])
                self.emit(f"IADD3 {dst}, {v(s[0])}, {tmp}, RZ", reads=[int(v(s[0])[1:]), int(tmp[1:])],
                          writes=[int(dst[1:])])
                continue
            text = {
                "mov": lambda: f"MOV {dst}, {v(s[0])}",
                "mulhi_u": lambda: f"IMAD.HI.U32 {dst}, {v(s[0])}, {v(s[1])}, RZ",
                "mulhi_s": lambda: f"IMAD.HI {dst}, {v(s[0])}, {v(s[1])}, RZ",
                "shr_u": lambda: f"SHF.R.U32.HI {dst}, RZ, {s[1]:#x}, {v(s[0])}",
                "shr_s": lambda: f"SHF.R.S32.HI {dst}, RZ, {s[1]:#x}, {v(s[0])}",
                "add": lambda: f"IADD3 {dst}, {v(s[0])}, {v(s[1])}, RZ",
                "sub": lambda: f"IADD3 {dst}, {v(s[0])}, -{v(s[1])}, RZ",
                "lea_hi_sign": lambda: f"LEA.HI {dst}, {v(s[1])}, {v(s[0])}, RZ, 0x1",
                "mad": lambda: f"IMAD {dst}, {v(s[0])}, {v(s[1])}, {v(s[2])}",
            }[k]()
            regs_read = [int(v(x)[1:]) for x in s if isinstance(x, str)]
            self.emit(text, reads=regs_read, writes=[int(dst[1:])])
        self.emit(f"MOV R{d}, {names[seq.result]}", reads=[int(names[seq.result][1:])], writes=[d])


# ---- scheduling: conservative control bits ---------------------------------------------

def schedule(code: list[Out]) -> list[tuple[str, Control]]:
    """In-order, no reordering. Fixed-latency ALU results: stall so the next reader
    is >= 6 cycles later (above every fixed latency mined from ptxas, max 5-6 for
    IMAD.WIDE/LEA). Variable-latency ops get a write scoreboard; readers wait on it."""
    out = []
    pending_sb: dict[int, int] = {}  # reg -> scoreboard
    next_sb = 0
    for k, o in enumerate(code):
        wait = 0
        for r in o.reads + o.writes:
            if r in pending_sb:
                wait |= 1 << pending_sb.pop(r)
        # clear all regs tied to waited scoreboards
        for r in [r for r, b in pending_sb.items() if wait >> b & 1]:
            pending_sb.pop(r)
        wbar = 7
        if o.var_lat:
            wbar = next_sb
            next_sb = (next_sb + 1) % 6
            for r in o.writes:
                pending_sb[r] = wbar
        stall = 6 if not o.var_lat else 2
        if o.text.startswith(("STG", "EXIT")):
            stall = 5
        out.append((o.text, Control(stall=stall, yield_=False, wbar=wbar, rbar=7, wait=wait, reuse=0)))
    return out
