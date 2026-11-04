"""PTX-level unrolling of counted do-while loops (the shape ptxas also unrolls 4x).

Recognized loop (no other labels/branches inside, %c not written by B, %n loop-invariant):

    L:  B;  add.u32 %c, %c, 1;  setp.lt.u32 %p, %c, %n;  @%p bra L;

Rewritten (B runs exactly as often as before; %p is false at the exit as before):

        B; inc                                    first iteration (do-while)
        setp.lt.u32 %p, %c, %n;  @!%p bra DONE
        sub.u32 %t, %n, %c;  setp.lt.u32 %q, %t, U;  @%q bra TAIL     # %t = iterations left
    MAIN: (B; inc) x U;  sub.u32 %t, %n, %c;  setp.ge.u32 %q, %t, U;  @%q bra MAIN
          (if B does not read %c:  add %c, U; setp.lt %q, %c, %n-(U-1); B x U; @%q bra MAIN)
        setp.lt.u32 %p, %c, %n;  @!%p bra DONE
    TAIL: B; inc;  setp.lt.u32 %p, %c, %n;  @%p bra TAIL
    DONE:

Correctness is checked per kernel by the caller (emulator vs PTX interpreter on the
original kernel).
"""

from __future__ import annotations

import copy

from openassembler.ptx.parse import ImmI, Instr, Kernel, Var


def _writes(ins: Instr) -> set:
    if ins.op in ("st", "bar", "red", "bra", "ret") or not ins.args:
        return set()
    a = ins.args[0]
    return {a.name} if isinstance(a, Var) else set()


def _reads(ins: Instr) -> set:
    out = set()
    for a in ins.args[1:] if _writes(ins) else ins.args:
        if isinstance(a, Var):
            out.add(a.name)
        elif hasattr(a, "base"):
            out.add(a.base)
        elif hasattr(a, "elems"):
            out |= {e.name for e in a.elems if isinstance(e, Var)}
    return out


def _match(k: Kernel, label: str):
    s = k.labels[label]
    e = next((i for i in range(s, len(k.body)) if k.body[i].op == "bra"), None)
    if e is None or e - s < 3:
        return None
    br, setp, inc = k.body[e], k.body[e - 1], k.body[e - 2]
    if not (br.guard and not br.guard[1] and isinstance(br.args[0], Var) and br.args[0].name == label):
        return None
    if not (setp.op == "setp" and setp.mods[:2] == ("lt", "u32") and setp.guard is None):
        return None
    p, c, n = setp.args
    if not (isinstance(p, Var) and p.name == br.guard[0] and isinstance(c, Var) and isinstance(n, Var)):
        return None
    if not (inc.op == "add" and inc.guard is None and inc.args[0] == c and inc.args[1] == c
            and inc.args[2] == ImmI(1)):
        return None
    body = k.body[s:e - 2]
    if any(i.op in ("bra", "ret", "exit") or i.guard is not None for i in body):
        return None
    if any(lab != label and s < idx < e + 1 for lab, idx in k.labels.items()):
        return None
    if any(c.name in _writes(i) or n.name in _writes(i) for i in body):
        return None
    return s, e, body, inc, setp, p, c, n


def unroll_loops(k: Kernel, factor: int = 4) -> Kernel:
    k = copy.deepcopy(k)
    for label in sorted(k.labels, key=k.labels.get, reverse=True):
        m = _match(k, label)
        if m is None:
            continue
        s, e, body, inc, setp, p, c, n = m
        t, q = Var(f"%ut_{label}"), Var(f"%uq_{label}")
        lm, lt, ld = f"{label}_U", f"{label}_T", f"{label}_D"

        def I(op, mods, args, guard=None):  # noqa: E741
            return Instr(op, tuple(mods), list(args), guard)

        it = list(body) + [inc]
        new, labels = [], {}
        new += it
        new += [setp, I("bra", [], [Var(ld)], (p.name, True))]
        new += [I("sub", ["u32"], [t, n, c]), I("setp", ["lt", "u32"], [q, t, ImmI(factor)]),
                I("bra", [], [Var(lt)], (q.name, False))]
        if any(c.name in _reads(i) for i in body):
            labels[lm] = len(new)
            new += it * factor
            new += [I("sub", ["u32"], [t, n, c]), I("setp", ["ge", "u32"], [q, t, ImmI(factor)]),
                    I("bra", [], [Var(lm)], (q.name, False))]
        else:
            # The body does not read the counter: one "add c, U" per trip, and the exit test
            # c < n - (U-1) (no underflow: we only get here with n - c >= U) is independent
            # of the body, so the scheduler can hide the 13-cycle predicate latency.
            lim = Var(f"%ul_{label}")
            new += [I("sub", ["u32"], [lim, n, ImmI(factor - 1)])]
            labels[lm] = len(new)
            new += [I("add", ["u32"], [c, c, ImmI(factor)]), I("setp", ["lt", "u32"], [q, c, lim])]
            new += list(body) * factor
            new += [I("bra", [], [Var(lm)], (q.name, False))]
        new += [setp, I("bra", [], [Var(ld)], (p.name, True))]
        labels[lt] = len(new)
        new += it + [setp, I("bra", [], [Var(lt)], (p.name, False))]
        labels[ld] = len(new)
        delta = len(new) - (e + 1 - s)
        k.body[s:e + 1] = new
        nl = {}
        for name, idx in k.labels.items():
            if name == label:
                continue          # the original header is gone (only its own branch used it)
            nl[name] = idx + delta if idx > e else idx
        for name, off in labels.items():
            nl[name] = s + off
        k.labels = nl
    return k
