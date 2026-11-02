"""Minimal PTX parser: one module -> kernels of flat instruction lists.

Covers the syntax our probes and typical ptxas input use: directives, .reg /
.shared / .local / .const / .global declarations (incl. `%r<N>` ranges),
labels, guards, vector operands `{a, b}`, addresses `[reg+off]`, `a|b`
predicate pairs, and immediates (hex, decimal, 0f/0d float bits).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Var:
    name: str          # %r1, smem, a param name


@dataclass(frozen=True)
class ImmI:
    value: int


@dataclass(frozen=True)
class ImmF:
    bits: int          # raw IEEE bits
    width: int         # 32 or 64


@dataclass(frozen=True)
class Addr:
    base: str          # register or symbol name
    off: int = 0


@dataclass(frozen=True)
class Vec:
    elems: tuple       # tuple of Var | ImmI | None ("_" sink)


@dataclass(frozen=True)
class PairDst:
    a: Var             # setp %p1|%p2
    b: Var


Operand = Var | ImmI | ImmF | Addr | Vec | PairDst


@dataclass
class Instr:
    op: str                         # "add"
    mods: tuple[str, ...]           # ("s32",) or ("sync", "bfly", "b32")
    args: list[Operand]
    guard: tuple[str, bool] | None  # (pred, negated)
    line: int = 0

    def has(self, *ms: str) -> bool:
        return all(m in self.mods for m in ms)


@dataclass
class VarDecl:
    space: str        # shared, local, const, global
    name: str
    size: int
    align: int


@dataclass
class Kernel:
    name: str
    params: list[tuple[str, str]]          # (type, name)
    body: list[Instr]
    labels: dict[str, int]                 # label -> index into body
    decls: list[VarDecl] = field(default_factory=list)
    branchtargets: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class Module:
    kernels: dict[str, Kernel]
    decls: list[VarDecl]


_TYPE_SIZE = {"b8": 1, "u8": 1, "s8": 1, "b16": 2, "u16": 2, "s16": 2, "f16": 2,
              "b32": 4, "u32": 4, "s32": 4, "f32": 4, "b64": 8, "u64": 8, "s64": 8, "f64": 8}
_VAR_DECL = re.compile(r"\.(shared|local|const|global)\s+(?:\.align\s+(\d+)\s+)?\.(\w+)\s+(\w+)(?:\[(\d+)\])?")
_PARAM = re.compile(r"\.param\s+\.(\w+)\s+(\w+)")
_ENTRY = re.compile(r"\.(?:visible\s+)?\.entry\s+(\w+)\s*\((.*?)\)\s*\{", re.S)


def _strip_comments(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return re.sub(r"//[^\n]*", "", src)


def _match_brace(src: str, start: int) -> int:
    depth = 0
    for i in range(start, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    raise SyntaxError("unbalanced braces")


def _split_top(s: str, sep: str = ",") -> list[str]:
    out, depth, cur = [], 0, []
    for ch in s:
        if ch in "{[(":
            depth += 1
        elif ch in "}])":
            depth -= 1
        if ch == sep and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        out.append("".join(cur).strip())
    return out


def parse_operand(t: str) -> Operand:
    t = t.strip().replace("+-", "-")
    if t.startswith("{"):
        return Vec(tuple(None if e == "_" else parse_operand(e) for e in _split_top(t[1:-1])))
    if t.startswith("["):
        inner = t[1:-1].replace(" ", "")
        m = re.fullmatch(r"([%\w$.]+)(?:([+-])(0x[0-9a-fA-F]+|\d+))?", inner)
        if not m:
            raise SyntaxError(f"address {t}")
        off = int(m[3], 0) if m[3] else 0
        return Addr(m[1], -off if m[2] == "-" else off)
    if "|" in t:
        a, b = t.split("|")
        return PairDst(Var(a.strip()), Var(b.strip()))
    if re.fullmatch(r"0[fF][0-9a-fA-F]{8}", t):
        return ImmF(int(t[2:], 16), 32)
    if re.fullmatch(r"0[dD][0-9a-fA-F]{16}", t):
        return ImmF(int(t[2:], 16), 64)
    if re.fullmatch(r"-?(0[xX][0-9a-fA-F]+|\d+)U?", t):
        return ImmI(int(t.rstrip("U"), 0))
    return Var(t)


def parse_instr(stmt: str, line: int) -> Instr:
    guard = None
    if stmt.startswith("@"):
        g, stmt = stmt.split(None, 1)
        neg = g.startswith("@!")
        guard = (g[2:] if neg else g[1:], neg)
    head, _, rest = stmt.partition(" ")
    if "\t" in head:
        head, _, r2 = head.partition("\t")
        rest = r2 + " " + rest
    op, *mods = head.split(".")
    args = [parse_operand(a) for a in _split_top(rest)] if rest.strip() else []
    return Instr(op, tuple(mods), args, guard, line)


def _decl(m: re.Match) -> VarDecl:
    space, align, ty, name, count = m.groups()
    size = _TYPE_SIZE[ty] * (int(count) if count else 1)
    return VarDecl(space, name, size, int(align) if align else _TYPE_SIZE[ty])


def parse_module(src: str) -> Module:
    src = _strip_comments(src)
    kernels: dict[str, Kernel] = {}
    module_decls: list[VarDecl] = []
    pos = 0
    for m in _ENTRY.finditer(src):
        for dm in _VAR_DECL.finditer(src[pos:m.start()]):
            module_decls.append(_decl(dm))
        params = [(pm[1], pm[2]) for pm in _PARAM.finditer(m[2])]
        end = _match_brace(src, m.end() - 1)
        kernels[m[1]] = _parse_body(m[1], params, src[m.end():end])
        pos = end + 1
    return Module(kernels, module_decls)


def _parse_body(name: str, params, body: str) -> Kernel:
    k = Kernel(name, params, [], {})
    for n, raw in enumerate(body.split(";")):
        stmt = " ".join(raw.split())
        # Peel labels ("L1: add ...", possibly several).
        while (lm := re.match(r"^([A-Za-z_$][\w$]*):\s*(.*)$", stmt)):
            label, stmt = lm[1], lm[2]
            if stmt.startswith(".branchtargets"):
                k.branchtargets[label] = [t.strip() for t in stmt.split(None, 1)[1].split(",")]
                stmt = ""
                break
            k.labels[label] = len(k.body)
        if not stmt or stmt in ("{", "}"):
            continue
        directive = stmt.split(None, 1)[0]
        if directive in (".reg", ".pragma", ".loc", ".file") or (stmt.startswith("{") and stmt.endswith("}")):
            continue
        if dm := _VAR_DECL.match(stmt):
            k.decls.append(_decl(dm))
            continue
        if stmt.startswith("."):
            continue
        k.body.append(parse_instr(stmt, n))
    return k
