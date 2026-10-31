"""Load an `nvdisasm -hex -c` listing into executable instructions."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from openassembler.emu.operands import Label, Operand, Pred, parse, split_operands

_INSN = re.compile(
    r"/\*(?P<addr>[0-9a-f]{4,})\*/\s+(?P<text>.*?)\s*;\s*/\*\s*0x(?P<lo>[0-9a-f]{16})\s*\*/"
)
_HI = re.compile(r"^\s*/\*\s*0x(?P<hi>[0-9a-f]{16})\s*\*/\s*$")
_LABEL = re.compile(r"^(?P<name>[.$\w]+):\s*$")
_GUARD = re.compile(r"^@(!?)(U?P)(T|\d)\s+")
_TRAILER = re.compile(r"\s*\(\*.*\*\)\s*$")  # e.g. BRX's (*"BRANCH_TARGETS ..."*)


@dataclass
class Insn:
    addr: int
    text: str
    word: int
    op: str                       # base mnemonic, e.g. "ISETP"
    mods: tuple[str, ...]         # e.g. ("GE", "U32", "AND")
    ops: list[Operand]
    guard: Pred | None = None     # None = unguarded

    def has(self, *mods: str) -> bool:
        return all(m in self.mods for m in mods)


@dataclass
class Program:
    insns: dict[int, Insn] = field(default_factory=dict)   # addr -> insn
    labels: dict[str, int] = field(default_factory=dict)   # label -> addr
    entry: int = 0

    def target(self, op: Operand) -> int:
        assert isinstance(op, Label), op
        return self.labels[op.name]


def parse_insn(addr: int, text: str, word: int) -> Insn:
    guard = None
    t = _TRAILER.sub("", text)
    if m := _GUARD.match(t):
        guard = Pred(m[2], 7 if m[3] == "T" else int(m[3]), m[1] == "!")
        t = t[m.end():]
    head, _, rest = t.partition(" ")
    op, *mods = head.split(".")
    return Insn(addr, text, word, op, tuple(mods), [parse(o) for o in split_operands(rest)], guard)


def load_listing(text: str, kernel: str | None = None) -> Program:
    """Parse one kernel's code section. Labels (incl. internal subroutines in the
    same section) map to the address of the next instruction."""
    prog = Program()
    lines = text.splitlines()
    pending: list[str] = []
    in_kernel = kernel is None
    for i, line in enumerate(lines):
        if line.lstrip().startswith(".section") and not line.lstrip().startswith(".sectioninfo"):
            sec = line.split()[1].split(",")[0]
            in_kernel = kernel is None or sec == f".text.{kernel}"
        if not in_kernel:
            continue
        if m := _LABEL.match(line.strip()):
            pending.append(m["name"])
            continue
        m = _INSN.search(line)
        if not m or i + 1 >= len(lines) or not (h := _HI.match(lines[i + 1])):
            continue
        addr = int(m["addr"], 16)
        word = int(m["lo"], 16) | (int(h["hi"], 16) << 64)
        prog.insns[addr] = parse_insn(addr, m["text"].strip(), word)
        for name in pending:
            prog.labels[name] = addr
        pending.clear()
    return prog


def load_sass_file(path: Path) -> Program:
    return load_listing(Path(path).read_text())
