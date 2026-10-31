"""Reference PTX interpreter (one CTA, lock-step warps).

Semantics come from the PTX ISA spec, not from SASS. Values are stored as raw
bits per register; each instruction's type suffix says how to read them.
Undefined-by-spec results (div/rem by zero, shift >= width handled per spec)
follow the spec where it defines them and raise `Undefined` where it doesn't,
so the harness can skip those lanes instead of encoding ptxas's choice as truth.

Execution model: a warp runs in lock-step with an active mask. Divergent
branches are handled by a reconvergence stack (immediate post-dominator =
the lowest pending pc, valid for the reducible, forward-structured code in our
probes); collectives see the active lanes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction

from openassembler.emu import fp
from openassembler.emu.machine import Memory
from openassembler.ptx.parse import Addr, ImmF, ImmI, Instr, Kernel, Module, PairDst, Var, Vec


class Undefined(Exception):
    """Result is undefined by the PTX spec (e.g. integer division by zero)."""


class NotModeled(Exception):
    pass


M = lambda bits: (1 << bits) - 1  # noqa: E731
_TBITS = {"b8": 8, "u8": 8, "s8": 8, "b16": 16, "u16": 16, "s16": 16, "f16": 16, "bf16": 16,
          "b32": 32, "u32": 32, "s32": 32, "f32": 32, "b64": 64, "u64": 64, "s64": 64, "f64": 64,
          "f16x2": 32, "bf16x2": 32, "pred": 1}
_TYPES = set(_TBITS)


def sext(v: int, bits: int) -> int:
    v &= M(bits)
    return v - (1 << bits) if v >> (bits - 1) else v


def types_of(ins: Instr) -> list[str]:
    return [m for m in ins.mods if m in _TYPES]


@dataclass
class Lane:
    tid: tuple[int, int, int]
    lane: int
    regs: dict[str, int] = field(default_factory=dict)
    local: Memory = field(default_factory=Memory)
    done: bool = False
    undefined: bool = False   # some result on this lane was spec-undefined


@dataclass
class Launch:
    block: tuple[int, int, int] = (32, 1, 1)
    grid: tuple[int, int, int] = (1, 1, 1)
    ctaid: tuple[int, int, int] = (0, 0, 0)
    params: dict[str, int] = field(default_factory=dict)


class Interp:
    def __init__(self, mod: Module, kernel: str, launch: Launch, gmem: Memory | None = None):
        self.mod, self.k, self.L = mod, mod.kernels[kernel], launch
        self.gmem = gmem or Memory()
        self.smem = Memory()
        self.cmem = Memory()
        # Symbol addresses: shared from 0, const in its own space, globals in a high window.
        self.sym: dict[str, tuple[str, int]] = {}
        offs = {"shared": 0, "const": 0, "global": 0x7E00_0000_0000, "local": 0}
        for d in self.mod.decls + self.k.decls:
            base = (offs[d.space] + d.align - 1) // d.align * d.align
            self.sym[d.name] = (d.space, base)
            offs[d.space] = base + d.size
        # Kernel-scope .local declarations may appear after first use in our PTX parser's view.
        self._local_decls = {d.name: self.sym[d.name][1] for d in self.k.decls if d.space == "local"}
        bx, by, bz = launch.block
        tids = [(x, y, z) for z in range(bz) for y in range(by) for x in range(bx)]
        self.warps = [[Lane(t, i) for i, t in enumerate(tids[w:w + 32])] for w in range(0, len(tids), 32)]
        self.steps = 0

    # ---- values ------------------------------------------------------------------

    def get(self, ln: Lane, op, bits: int) -> int:
        if isinstance(op, ImmI):
            return op.value & M(bits)
        if isinstance(op, ImmF):
            if op.width == bits:
                return op.bits
            # 0f literal used with a 64-bit type: convert value exactly
            return fp.encode(fp.decode(op.bits, fp.F32), fp.F64)
        if isinstance(op, Var):
            n = op.name
            if n in self.sym:
                return self.sym[n][1]
            if n in self._local_decls:
                return self._local_decls[n]
            if n.startswith("%") and n not in ln.regs and n.split(".")[0] in SREGS:
                return self.sreg(ln, n) & M(bits)
            if n not in ln.regs:
                if ln.undefined:
                    return 0  # value is already poisoned for this lane; result is skipped
                raise NotModeled(f"read of undefined register {n}")
            return ln.regs[n] & M(bits)
        raise NotModeled(f"operand {op}")

    def put(self, ln: Lane, op, value: int, bits: int) -> None:
        if isinstance(op, Var) and op.name != "_":
            ln.regs[op.name] = value & M(bits)

    def pred(self, ln: Lane, name: str) -> bool:
        neg = name.startswith("!")
        return bool(ln.regs.get(name.lstrip("!"), 0)) != neg

    def sreg(self, ln: Lane, name: str) -> int:
        L, (x, y, z) = self.L, ln.tid
        base, _, comp = name.partition(".")
        idx = "xyz".index(comp) if comp else 0
        table = {"%tid": ln.tid[idx], "%ntid": L.block[idx], "%ctaid": L.ctaid[idx], "%nctaid": L.grid[idx],
                 "%laneid": ln.lane, "%lanemask_eq": 1 << ln.lane, "%lanemask_lt": (1 << ln.lane) - 1,
                 "%lanemask_le": (1 << (ln.lane + 1)) - 1, "%lanemask_ge": M(32) & ~((1 << ln.lane) - 1),
                 "%lanemask_gt": M(32) & ~((1 << (ln.lane + 1)) - 1)}
        if base not in table:
            raise NotModeled(f"special register {name}")
        return table[base]

    def addr(self, ln: Lane, a: Addr) -> int:
        if a.base in self.sym:
            return self.sym[a.base][1] + a.off
        if a.base in [p for _, p in self.k.params]:
            raise NotModeled("address of param")
        return (ln.regs[a.base] + a.off) & M(64)

    def space_mem(self, ins: Instr, ln: Lane):
        for s, mem in (("shared", self.smem), ("local", ln.local), ("const", self.cmem), ("global", self.gmem)):
            if s in ins.mods:
                return mem
        if "param" in ins.mods:
            return None
        return self.gmem  # generic addressing: probes only use global pointers generically

    # ---- execution -------------------------------------------------------------------

    def run(self, max_steps: int = 2_000_000) -> None:
        from openassembler.ptx import semantics as S

        for w in self.warps:
            # Each entry: (pc, lanes). Always run the entry with the lowest pc;
            # entries reaching the same pc merge (reconvergence).
            groups: dict[int, list[Lane]] = {0: list(w)}
            while groups:
                pc = min(groups)
                lanes = [ln for ln in groups.pop(pc) if not ln.done]
                if not lanes:
                    continue
                if pc >= len(self.k.body):
                    raise NotModeled("fell off the end of the kernel")
                ins = self.k.body[pc]
                nxt = S.execute(self, ins, lanes, pc)
                for target, tl in nxt.items():
                    groups.setdefault(target, []).extend(tl)
                self.steps += 1
                if self.steps > max_steps:
                    raise NotModeled("step limit exceeded")

    def guard(self, ln: Lane, ins: Instr) -> bool:
        if ins.guard is None:
            return True
        name, neg = ins.guard
        return bool(ln.regs.get(name, 0)) != neg


SREGS = {"%tid", "%ntid", "%ctaid", "%nctaid", "%laneid", "%lanemask_eq", "%lanemask_lt",
         "%lanemask_le", "%lanemask_ge", "%lanemask_gt"}


def run_kernel(src_mod: Module, kernel: str, launch: Launch, gmem: Memory) -> Interp:
    it = Interp(src_mod, kernel, launch, gmem)
    it.run()
    return it


__all__ = ["Interp", "Launch", "Lane", "Undefined", "NotModeled", "run_kernel", "sext", "M",
           "Fraction", "math", "PairDst", "Vec"]
