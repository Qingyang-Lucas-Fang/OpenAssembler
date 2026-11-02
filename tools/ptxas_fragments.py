"""Freeze ptxas's SASS for single PTX ops into openassembler/backend/ptxas_fragments.json.

Usage: uv run --no-project python tools/ptxas_fragments.py [--only KEY]   (needs ptxas + nvdisasm)

For each op (div.rn.f32, rcp.approx.ftz.f32, div.s32, cvt.rmi.f32.f32, ...): build the
standard probe kernel (tools/ptx_probes.py ABI: inputs from a 16-byte global slot per
thread, result stored as 8 bytes), compile with ptxas -arch=sm_80, disassemble, and cut
out the op's code:
  * drop the probe scaffolding: R1 setup, S2R/IMAD.WIDE.U32 address arithmetic, ULDC,
    LDG/STG, EXIT, BSSY/BSYNC, padding;
  * inputs = the registers the LDG wrote (by word), output = the register the STG stored;
  * CALL.REL.NOINC sub / RET.REL.NODEC Rx become plain branches (sub entered by BRA,
    left by BRA to a label after the call site) -- the return-address MOVs then die and
    are removed by a liveness DCE over the fragment's CFG, like every other dead def;
  * branch threading (BRA to BRA, @P BRA a; BRA b; a: -> @!P BRA b) on the result.
Every instruction kept is ptxas's text verbatim (registers are renamed per use site by
the backend). The whole ptxas listing is stored too, so tests can run ptxas's own code
on the emulator offline and compare it with ours.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import ptx_probes  # noqa: E402
from openassembler import oracle  # noqa: E402
from openassembler.backend import fragments as F  # noqa: E402
from openassembler.isa.word import control  # noqa: E402

OUT = ROOT / "openassembler" / "backend" / "ptxas_fragments.json"

# key -> (number of sources, source type, destination type). Sources are the probe's
# %f1/%f2 (f32) or %r1/%r2 (32-bit int) = input words 0 and 1.
F1, F2 = ("f32",) * 2
OPS: dict[str, tuple[int, str, str]] = {}
for k in ("div.rn.f32", "div.approx.f32", "div.approx.ftz.f32", "div.full.f32",
          "div.full.ftz.f32", "div.rz.f32", "div.rm.f32", "div.rp.f32"):
    OPS[k] = (2, "f32", "f32")
for k in ("rcp.rn.f32", "rcp.rn.ftz.f32", "rcp.approx.f32", "rcp.approx.ftz.f32", "sqrt.rn.f32",
          "sqrt.rn.ftz.f32", "sqrt.approx.f32", "sqrt.approx.ftz.f32", "rsqrt.approx.f32",
          "rsqrt.approx.ftz.f32", "ex2.approx.f32", "ex2.approx.ftz.f32", "lg2.approx.f32",
          "lg2.approx.ftz.f32", "sin.approx.f32", "sin.approx.ftz.f32", "cos.approx.f32",
          "cos.approx.ftz.f32", "tanh.approx.f32"):
    OPS[k] = (1, "f32", "f32")
for k in ("div.s32", "div.u32", "rem.s32", "rem.u32"):
    OPS[k] = (2, k[-3:], k[-3:])
for t in ("s32", "u32"):
    OPS[f"cvt.rn.f32.{t}"] = (1, t, "f32")
    OPS[f"cvt.rz.f32.{t}"] = (1, t, "f32")
    OPS[f"cvt.rm.f32.{t}"] = (1, t, "f32")
    OPS[f"cvt.rp.f32.{t}"] = (1, t, "f32")
    for r in ("rzi", "rni", "rmi", "rpi"):
        OPS[f"cvt.{r}.{t}.f32"] = (1, "f32", t)
        OPS[f"cvt.{r}.ftz.{t}.f32"] = (1, "f32", t)
for r in ("rzi", "rni", "rmi", "rpi"):
    OPS[f"cvt.{r}.f32.f32"] = (1, "f32", "f32")
    OPS[f"cvt.{r}.ftz.f32.f32"] = (1, "f32", "f32")
OPS["cvt.sat.f32.f32"] = (1, "f32", "f32")
OPS["cvt.ftz.sat.f32.f32"] = (1, "f32", "f32")


def probe_body(key: str) -> str:
    n, st, dt = OPS[key]
    src = ["%f1", "%f2"] if st == "f32" else ["%r1", "%r2"]
    if dt == "f32":
        return f" {key} %f10, {', '.join(src[:n])}; mov.b32 %r10, %f10;"
    return f" {key} %r10, {', '.join(src[:n])};"


def probe_ptx(key: str) -> tuple[str, str]:
    kname = "q_" + re.sub(r"\W", "_", key)
    src = ptx_probes.HEADER.format(target="sm_80") + ptx_probes.KERNEL.format(
        name=kname, shared="", body=probe_body(key))
    return kname, src


_LABEL = re.compile(r"^\s*(\.L_x_\d+|\$[\w$]+):\s*$")


def listing(sass: str):
    """nvdisasm -hex output -> [("label", name) | ("insn", text, word)] (labels kept)."""
    lines, out = sass.splitlines(), []
    for i, line in enumerate(lines):
        m = _LABEL.match(line)
        if m:
            out.append(("label", m[1]))
            continue
        m = oracle._INSN.search(line)
        if m and i + 1 < len(lines) and (h := oracle._HI.match(lines[i + 1])):
            out.append(("insn", int(m["addr"], 16), m["text"].strip(),
                        int(m["lo"], 16) | (int(h["hi"], 16) << 64)))
    return out


def _normalize_calls(rows, prog):
    """Inline each helper called by the probe body, replacing CALL/RET with BRA.

    The common out-of-line helpers are leaf routines. Refuse nested calls rather
    than emit a fragment with behavior we did not model.
    """
    from openassembler.emu.operands import Label

    main = list(rows)
    clones = []
    call_no = 0
    i = 0
    while i < len(main):
        tag = main[i][0]
        if tag != "insn":
            i += 1
            continue
        ins = prog.insns[main[i][1]]
        if ins.op != "CALL":
            i += 1
            continue
        target = next((o for o in ins.ops if isinstance(o, Label)), None)
        if target is None or target.name not in prog.labels:
            raise RuntimeError(f"cannot resolve ptxas helper call {ins.text}")
        start = prog.labels[target.name]
        addrs = sorted(a for a in prog.insns if a >= start)
        if not addrs:
            raise RuntimeError(f"empty ptxas helper {target.name}")
        helper = []
        helper_labels = {n: a for n, a in prog.labels.items() if a >= start}
        end = None
        for addr in addrs:
            h = prog.insns[addr]
            if h.op == "CALL":
                raise RuntimeError(f"nested call inside ptxas helper {target.name} is not supported")
            helper.append(("insn", addr, h.text, h.word))
            if h.op == "RET":
                end = addr
                break
        if end is None:
            raise RuntimeError(f"ptxas helper {target.name} has no RET")
        helper_labels = {n: a for n, a in helper_labels.items() if a <= end}
        call_no += 1
        clone_entry = f"F_call_{call_no}"
        cont = f"F_cont_{call_no}"
        main[i] = ("insn", main[i][1], f"BRA `({clone_entry})", ins.word)
        main.insert(i + 1, ("label", cont))
        rename = {n: f"{n}_f{call_no}" for n in helper_labels}
        clones.append(("label", clone_entry))
        clone_index = 0
        for item in helper:
            _, addr, text, word = item
            for old, old_addr in helper_labels.items():
                if old_addr == addr:
                    clones.append(("label", rename[old]))
            h = prog.insns[addr]
            if h.op == "RET":
                text = f"BRA `({cont})"
            else:
                for old, new in rename.items():
                    text = text.replace(f"`({old})", f"`({new})")
            clones.append(("insn", -1 - clone_index, text, word))
            clone_index += 1
        i += 2
    return main + clones


def _slice_rows(rows, input_load_addrs: list[int], output_store_addr: int, prog):
    """Backward dataflow slice for the probe result, retaining branches that can
    reach result-producing code. The input vector load is a boundary: its live
    register words become the fragment's arguments. Output pointer/address uses
    are deliberately excluded from the store root.
    """
    from openassembler.emu.operands import Label, Reg
    from openassembler.emu.program import parse_insn

    # Slice only through the probe's entry body; helper clones are included by the
    # call normalizer and may follow the output store in address order.
    labels, insns, source_rows = {}, [], []
    for row in rows:
        if row[0] == "label":
            labels[row[1]] = len(insns)
            source_rows.append(row)
        else:
            _, addr, text, word = row
            insns.append(parse_insn(addr, text, word))
            source_rows.append(row)
    label_at_row = {}
    ix = 0
    for ri, row in enumerate(source_rows):
        if row[0] == "label":
            label_at_row[ri] = row[1]
        else:
            ix += 1
    row_to_i, i_to_row = [], []
    for ri, row in enumerate(source_rows):
        if row[0] == "insn":
            row_to_i.append((ri, len(i_to_row)))
            i_to_row.append(ri)
    addr_to_i = {source_rows[ri][1]: ii for ri, ii in row_to_i if source_rows[ri][1] >= 0}
    labels_i = {}
    for ri, name in label_at_row.items():
        nxt = next((ii for rj, ii in row_to_i if rj > ri), len(i_to_row))
        labels_i[name] = nxt

    succ = [[] for _ in insns]
    uses, defs = [], []
    from mine_latency import defs_uses
    for i, ins in enumerate(insns):
        if ins.op in F._WHITELIST and not (F._WIDE_MODS & set(ins.mods) and ins.op != "SHF"):
            rr, rw, pr, pw = F.reads_writes(ins.text)
            d = {("R", r) for r in rw} | {("P", p) for p in pw}
            u = {("R", r) for r in rr} | {("P", p) for p in pr}
        else:
            d, u = defs_uses(ins)
        defs.append(set(d))
        uses.append(set(u))
        nxt = i + 1
        if ins.op == "BRA":
            label = next((o.name for o in ins.ops if isinstance(o, Label)), None)
            if label not in labels_i:
                raise RuntimeError(f"branch target {label!r} is outside the fragment")
            succ[i].append(labels_i[label])
            if ins.guard is not None and nxt < len(insns):
                succ[i].append(nxt)
        elif ins.op == "EXIT":
            pass
        elif i == addr_to_i.get(output_store_addr):
            pass
        elif nxt < len(insns):
            succ[i].append(nxt)

    root_i = addr_to_i[output_store_addr]
    input_is = {addr_to_i[a] for a in input_load_addrs}
    # The output store is a synthetic sink that reads only the stored value, not
    # the probe's unrelated output pointer.
    store = insns[root_i]
    src = next((operand for operand in reversed(store.ops) if isinstance(operand, Reg)), None)
    if src is None:
        raise RuntimeError("probe result store has no register source")
    root_use = {("R", src.idx)}
    in_sets = [set() for _ in insns]
    out_sets = [set() for _ in insns]
    changed = True
    while changed:
        changed = False
        for i in range(len(insns) - 1, -1, -1):
            out = root_use if i == root_i else set().union(*(in_sets[j] for j in succ[i])) if succ[i] else set()
            if i == root_i or i in input_is:
                inn = out - defs[i]
            else:
                needed = bool(out & defs[i]) or insns[i].op == "BRA"
                kill = defs[i] if insns[i].guard is None else set()
                inn = ((out - kill) | uses[i]) if needed else out
            if i == root_i:
                inn = root_use
            if out != out_sets[i] or inn != in_sets[i]:
                out_sets[i], in_sets[i] = out, inn
                changed = True

    reach = [False] * len(insns)
    reach[root_i] = True
    changed = True
    while changed:
        changed = False
        for i in range(len(insns) - 1, -1, -1):
            v = reach[i] or any(reach[j] for j in succ[i])
            if v != reach[i]:
                reach[i], changed = v, True

    keep = set()
    for i, ins in enumerate(insns):
        if i == root_i:
            keep.add(i)
        elif ins.op == "BRA" and any(reach[j] for j in succ[i]):
            keep.add(i)
        elif defs[i] & out_sets[i]:
            keep.add(i)
        if i in input_is:
            keep.discard(i)

    # A live input word is identified by the byte offset of the scalar/vector
    # global load relative to the probe's input base.
    from mine_latency import width_of
    input_map = {}
    for addr in input_load_addrs:
        i = addr_to_i[addr]
        ld = prog.insns[addr]
        dst = next((o for o in ld.ops if isinstance(o, Reg)), None)
        mem = next((o for o in ld.ops if o.__class__.__name__ == "Mem"), None)
        if dst is None or mem is None:
            continue
        start = max(0, mem.off // 4)
        for k in range(width_of(ld, dst, True)):
            r = dst.idx + k
            input_map[r] = start + k

    keep_rows = []
    live_labels = {o.name for i in keep for o in insns[i].ops if hasattr(o, "name") and o.__class__.__name__ == "Label"}
    # Any retained label is harmless; keep it to preserve relative control flow.
    for ri, row in enumerate(source_rows):
        if row[0] == "label":
            keep_rows.append(row)
            continue
        ii = next(ii for rj, ii in row_to_i if rj == ri)
        if ii < min(input_is):
            continue
        if ii not in keep:
            continue
        ins = insns[ii]
        if ins.op in ("BSSY", "BSYNC", "BREAK", "NOP"):
            continue
        if ii == root_i:
            keep_rows.append(("insn", -10000, "BRA `(F_END)", 0))
            continue
        if ins.op in ("STG", "ST", "EXIT"):
            continue
        if ins.op not in ("BRA",) and ins.op not in F._WHITELIST:
            raise RuntimeError(f"needed SASS opcode is not supported in a fragment: {ins.text}; "
                               f"defs={defs[ii]}, live_out={out_sets[ii]}, input_is={input_is}")
        keep_rows.append(row)
    if not input_map:
        raise RuntimeError("the result slice has no live input words")
    keep_rows.append(("label", "F_END"))
    return keep_rows, input_map, src.idx


def build_fragment(key: str, record: dict | None = None) -> dict:
    kname, ptx = probe_ptx(key)
    if record is not None:
        if record["kernel"] != kname or record["ptx"] != ptx:
            raise RuntimeError(f"cached probe differs for {key}")
        sass = record["sass"]
    else:
        src = oracle._scratch_file(ptx.encode(), ".ptx")
        cub = src.with_suffix(".cubin")
        try:
            oracle.ptxas(src, cub, arch="sm_80")
            sass = oracle.disasm_text(cub)
        finally:
            src.unlink(missing_ok=True)
            cub.unlink(missing_ok=True)
    from openassembler.emu.program import load_listing

    prog = load_listing(sass, kernel=kname)
    ordered = sorted(prog.insns.values(), key=lambda i: i.addr)
    lds = [i for i in ordered if i.op == "LDG"]
    ld = lds[0] if lds else None
    st = next((i for i in ordered if i.op == "STG" and
               any(x.op == "EXIT" for x in ordered if x.addr > i.addr)), None)
    if ld is None or st is None:
        raise RuntimeError(f"{key}: couldn't identify probe input/output vector memory operations")
    first_exit = next(i.addr for i in ordered if i.op == "EXIT")
    rows = listing(sass)
    main = [r for r in rows if (r[0] == "label" and prog.labels.get(r[1], st.addr + 1) <= st.addr)
            or (r[0] == "insn" and r[1] <= st.addr)]
    # Keep from the beginning: liveness removes all ABI/address setup but retains
    # any immediate materialization the selected arithmetic actually needs.
    rows = _normalize_calls(main, prog)
    # The synthetic result store serves as the root. Strip its address path later.
    rows = [r for r in rows if not (r[0] == "insn" and r[1] > st.addr)]
    input_addrs = [i.addr for i in lds if i.addr < st.addr]
    rows, inputs, output = _slice_rows(rows, input_addrs, st.addr, prog)
    inputs = {r: i for r, i in inputs.items() if i < OPS[key][0]}
    lines, vl = [], []
    for row in rows:
        if row[0] == "label":
            lines.append({"label": row[1]})
        else:
            _, _addr, text, word = row
            ins = __import__("openassembler.emu.program", fromlist=["parse_insn"]).parse_insn(0, text, word)
            if ins.op in ("BSSY", "BSYNC", "BREAK", "NOP", "STG", "EXIT"):
                continue
            # ptxas uses a packed-half multiply-add as a constant materializer.
            # Preserve its raw payload without depending on half-immediate encoding.
            if ins.op == "HFMA2" and text.split(",")[1:3] == [" -RZ", " RZ"]:
                dst = text.split()[1].rstrip(",")
                text = f"MOV {dst}, {((word >> 32) & 0xffffffff):#x}"
            lines.append({"text": text, "var_lat": ins.op != "BRA" and (ins.op in ("MUFU", "I2F", "F2I", "FRND") or control(word).wbar != 7)})
    if len(lines) >= 2 and lines[-1] == {"label": "F_END"} and lines[-2].get("text") == "BRA `(F_END)":
        lines = lines[:-2]
    # Drop labels that are not targets after pruning, but preserve fallthrough labels
    # because the caller may branch to them from an adjacent retained block.
    return {"inputs": {str(k): v for k, v in sorted(inputs.items())}, "output": output,
            "lines": lines, "sass": sass, "ptx": ptx, "kernel": kname}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated operation keys or substring")
    ap.add_argument("--output", type=Path, default=OUT)
    ap.add_argument("--from-library", type=Path, help="re-slice recorded ptxas listings without invoking the toolkit")
    args = ap.parse_args()
    recorded = json.loads(args.from_library.read_text())["ops"] if args.from_library else {}
    if not oracle.available("ptxas") or not oracle.available("nvdisasm"):
        raise SystemExit(oracle.NEEDS_TOOLKIT)
    keys = [k for k in OPS if not args.only or args.only in k or k in args.only.split(",")]
    ops, errors = {}, []
    for i, key in enumerate(keys, 1):
        try:
            ops[key] = build_fragment(key, recorded.get(key))
            print(f"[{i}/{len(keys)}] {key}: {len(ops[key]['lines'])} SASS lines")
        except Exception as e:
            errors.append(key)
            print(f"FAILED {key}: {type(e).__name__}: {e}", file=sys.stderr)
    if errors or not ops:
        raise SystemExit(f"fragment generation failed; previous output preserved: {errors}")
    data = {"schema": 1, "arch": "sm_80", "toolkit": "12.8", "ops": ops}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    tmp.replace(args.output)
    print(f"{len(ops)} fragments -> {args.output}")


if __name__ == "__main__":
    main()
