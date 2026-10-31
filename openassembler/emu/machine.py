"""Machine state and execution loop for one CTA.

Model: every thread has its own PC (Volta+ independent thread scheduling).
Each warp repeatedly picks the lowest PC among its live threads and executes
that instruction for all threads sitting at it. For structured code this
reconverges exactly where BSSY/BSYNC would, so convergence barriers are no-ops.
Warp collectives (SHFL, VOTE, ...) see the lanes executing together.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

from openassembler.emu.operands import CBank, Imm, Mem, Pred, Raw, Reg
from openassembler.emu.program import Insn, Program

M32 = 0xFFFFFFFF


class EmuError(Exception):
    pass


class Unsupported(EmuError):
    pass


@dataclass
class Memory:
    """Sparse byte-addressed little-endian memory."""

    data: dict[int, int] = field(default_factory=dict)

    def read(self, addr: int, n: int) -> int:
        return int.from_bytes(bytes(self.data.get(addr + i, 0) for i in range(n)), "little")

    def write(self, addr: int, n: int, value: int) -> None:
        for i, b in enumerate((value & ((1 << 8 * n) - 1)).to_bytes(n, "little")):
            self.data[addr + i] = b

    def write_bytes(self, addr: int, data: bytes) -> None:
        for i, b in enumerate(data):
            self.data[addr + i] = b


@dataclass
class Thread:
    tid: tuple[int, int, int]
    lane: int
    warp: int
    r: list[int] = field(default_factory=lambda: [0] * 256)
    p: list[bool] = field(default_factory=lambda: [False] * 8)
    pc: int = 0
    done: bool = False
    local: Memory = field(default_factory=Memory)
    stack: list[int] = field(default_factory=list)  # CALL return addresses


@dataclass
class Warp:
    lanes: list[Thread]
    ur: list[int] = field(default_factory=lambda: [0] * 64)
    up: list[bool] = field(default_factory=lambda: [False] * 8)


@dataclass
class Launch:
    grid: tuple[int, int, int] = (1, 1, 1)
    block: tuple[int, int, int] = (32, 1, 1)
    ctaid: tuple[int, int, int] = (0, 0, 0)
    params: bytes = b""
    param_base: int = 0x160           # sm_80; 0x210 on sm_90, 0x380 on sm_100
    stack_top: int = 0xFFFC           # value ptxas loads into R1 from c[0x0][0x28]
    smem_size: int = 0x10000
    smem_static: int = 0              # bytes of static .shared the kernel declares
    smem_reserved: int = 0x400        # c[0x0][0x114] on sm_80 (system-reserved smem)
    poison: int | None = None         # if set, registers start as this pattern (not 0): exposes
                                      # reads of never-written registers, which real GPUs leave as garbage


class Machine:
    def __init__(self, prog: Program, launch: Launch, gmem: Memory | None = None):
        self.prog = prog
        self.launch = launch
        self.gmem = gmem or Memory()
        self.smem = Memory()
        self.cbank: dict[int, Memory] = {0: Memory(), 2: Memory(), 3: Memory()}
        self._init_cbank0()
        bx, by, bz = launch.block
        threads = [(x, y, z) for z in range(bz) for y in range(by) for x in range(bx)]
        self.warps: list[Warp] = []
        for w in range(0, len(threads), 32):
            lanes = [Thread(t, i, w // 32) for i, t in enumerate(threads[w:w + 32])]
            for th in lanes:
                if launch.poison is not None:
                    th.r = [(launch.poison ^ (i * 0x9E3779B1)) & M32 for i in range(256)]
                th.r[255] = 0
                th.p[7] = True
            wp = Warp(lanes)
            wp.up[7] = True
            self.warps.append(wp)
        self.steps = 0

    def _init_cbank0(self) -> None:
        c0, L = self.cbank[0], self.launch
        for i, v in enumerate(L.block):
            c0.write(0x0 + 4 * i, 4, v)           # ntid.{x,y,z}
        for i, v in enumerate(L.grid):
            c0.write(0xC + 4 * i, 4, v)           # nctaid.{x,y,z}
        c0.write(0x28, 4, L.stack_top)
        c0.write(0x114, 4, L.smem_reserved)
        c0.write_bytes(L.param_base, L.params)

    # ---- operand access ------------------------------------------------------

    def rd(self, th: Thread, wp: Warp, op, *, width: int = 32) -> int:
        """Read an integer source operand (no neg/not applied)."""
        if isinstance(op, Reg):
            # width/32 consecutive registers, low word first. 128-bit (STS.128/STG.E.128)
            # used to read only one word: found by the A100 ldmatrix probe.
            regs, zero = (wp.ur, 63) if op.file == "UR" else (th.r, 255)
            if op.idx == zero:
                return 0
            v = 0
            for k in range(max(1, width // 32)):
                v |= regs[op.idx + k] << (32 * k)
            return v
        if isinstance(op, Imm):
            return op.int32()
        if isinstance(op, CBank):
            bank = op.bank if op.ur is None else wp.ur[op.ur]
            off = op.off + (self.rd(th, wp, op.reg) if op.reg else 0)
            return self.cbank.setdefault(bank, Memory()).read(off, width // 8)
        raise Unsupported(f"read of {op!r}")

    def src(self, th: Thread, wp: Warp, op) -> int:
        """32-bit integer source with '-' (2's complement) and '~' applied."""
        v = self.rd(th, wp, op)
        if getattr(op, "bnot", False):
            v = ~v & M32
        if getattr(op, "neg", False):
            v = -v & M32
        return v

    def wr(self, th: Thread, wp: Warp, op: Reg, value: int, *, width: int = 32) -> None:
        regs, zero = (wp.ur, 63) if op.file == "UR" else (th.r, 255)
        for k in range(width // 32):
            if op.idx + k != zero:
                regs[op.idx + k] = (value >> (32 * k)) & M32

    def prd(self, th: Thread, wp: Warp, op: Pred) -> bool:
        v = (wp.up if op.file == "UP" else th.p)[op.idx]
        return v != op.neg

    def pwr(self, th: Thread, wp: Warp, op: Pred, value: bool) -> None:
        if op.idx != 7:
            (wp.up if op.file == "UP" else th.p)[op.idx] = value

    def addr(self, th: Thread, wp: Warp, m: Mem, wide: bool | None = None) -> int:
        """`wide` overrides the operand's .64 marker: nvdisasm prints ATOMG.E's address
        as [R2] although .E (extended) means a 64-bit register pair."""
        if wide is not None:
            m = Mem(m.base, wide, m.off, m.ur, m.scale)
        a = self.rd(th, wp, m.base, width=64 if m.wide else 32) * m.scale if m.base else 0
        if m.ur is not None and m.base is not None and not m.wide:
            a += wp.ur[m.ur]
        return (a + m.off) & ((1 << 64) - 1)

    # ---- execution -----------------------------------------------------------

    def run(self, max_steps: int = 1_000_000) -> None:
        """Warps run one at a time until they finish or reach a CTA barrier. With several
        warps, BAR.SYNC is cooperative: a warp stops at the barrier, and when every live
        warp is waiting, all of them pass it together."""
        from openassembler.emu import ops  # late import: ops registers handlers on Machine

        for wp in self.warps:
            for th in wp.lanes:
                th.pc = self.prog.entry
        multi = len(self.warps) > 1
        waiting: set[int] = set()
        while True:
            for w, wp in enumerate(self.warps):
                if w in waiting:
                    continue
                while True:
                    live = [t for t in wp.lanes if not t.done]
                    if not live:
                        break
                    pc = min(t.pc for t in live)
                    group = [t for t in live if t.pc == pc]
                    insn = self.prog.insns.get(pc)
                    if insn is None:
                        raise EmuError(f"no instruction at {pc:#x}")
                    if multi and insn.op == "BAR" and insn.has("SYNC") and not insn.has("RED") \
                            and any(self.guard(t, wp, insn) for t in group):
                        waiting.add(w)
                        break
                    ops.execute(self, wp, group, insn)
                    self.steps += 1
                    if self.steps > max_steps:
                        raise EmuError("step limit exceeded")
            live_w = {w for w, wp in enumerate(self.warps) if any(not t.done for t in wp.lanes)}
            if not live_w:
                return
            if live_w - waiting:
                continue          # cannot happen: every non-waiting live warp ran to a stop
            # every live warp is at a barrier: release all of them
            for w in waiting:
                wp = self.warps[w]
                live = [t for t in wp.lanes if not t.done]
                pc = min(t.pc for t in live)
                for t in live:
                    if t.pc == pc:
                        t.pc += 16
            waiting.clear()

    def guard(self, th: Thread, wp: Warp, insn: Insn) -> bool:
        return insn.guard is None or self.prd(th, wp, insn.guard)


def f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & M32))[0]


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def special_reg(m: Machine, th: Thread, wp: Warp, name: str) -> int:
    L = m.launch
    table = {
        "SR_TID.X": th.tid[0], "SR_TID.Y": th.tid[1], "SR_TID.Z": th.tid[2],
        "SR_CTAID.X": L.ctaid[0], "SR_CTAID.Y": L.ctaid[1], "SR_CTAID.Z": L.ctaid[2],
        "SR_LANEID": th.lane,
        "SR_LEMASK": 1 << th.lane, "SR_LTMASK": (1 << th.lane) - 1,
        "SR_LEMASK_LE": (1 << (th.lane + 1)) - 1, "SR_GEMASK": (M32 << th.lane) & M32,
        "SR_GTMASK": (M32 << (th.lane + 1)) & M32,
        "SR_EQMASK": 1 << th.lane,
        "SR_VIRTUALSMID": 0, "SR_SMID": 0, "SR_VIRTID": th.warp << 8,
        # total shared allocation incl. the 1 KB system reserve that ptxas subtracts
        # via c[0x0][0x114] (A100: total_smem_size = 0 for a kernel with no smem)
        "SR_SMEMSZ": m.cbank[0].read(0x114, 4) + L.smem_static,
    }
    if name not in table:
        raise Unsupported(f"special register {name}")
    return table[name] & M32


__all__ = ["Machine", "Launch", "Memory", "Thread", "Warp", "EmuError", "Unsupported",
           "M32", "f32", "f32_bits", "special_reg", "Raw"]
