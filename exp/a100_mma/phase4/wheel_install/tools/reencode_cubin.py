"""End-to-end check of encoder + cubin writer: text -> words -> cubin -> nvdisasm.

Usage: uv run --no-project python tools/reencode_cubin.py [sm_80]

For every corpus cubin: re-encode each instruction of each .text section from its
*disassembly text* (plus control bits), splice the words into a copy of the cubin
with our ELF writer, then disassemble the copy with nvdisasm and compare the text.
Kernels with any refused instruction keep the original word for that slot and are
counted as partial.
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openassembler import oracle  # noqa: E402
from openassembler.container.elf import Elf  # noqa: E402
from openassembler.emu.operands import Label  # noqa: E402
from openassembler.emu.program import load_listing  # noqa: E402
from openassembler.isa.encode import EncodeError, Encoder  # noqa: E402
from openassembler.isa.word import control  # noqa: E402
from openassembler.oracle import words_to_bytes  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main(arch: str = "sm_80") -> None:
    corpus = ROOT / "corpus" / arch
    out = corpus / "reenc"
    out.mkdir(exist_ok=True)
    for f in out.iterdir():
        f.unlink()
    enc = Encoder(arch)
    n_insn = n_ours = full = 0
    names = []
    for sass in sorted((corpus / "sass").glob("*.sass")):
        kname = sass.stem
        cubin = corpus / "cubin" / f"{kname}.cubin"
        prog = load_listing(sass.read_text(), kernel=kname)
        words, ours = [], 0
        for addr in sorted(prog.insns):
            ins = prog.insns[addr]
            c = control(ins.word)
            try:
                lbl = next((o for o in ins.ops if isinstance(o, Label)), None)
                if lbl is not None and ins.op in ("BRA", "BSSY", "CALL"):
                    w = enc.encode_branch(ins.text, addr, prog.labels[lbl.name], c)
                else:
                    w = enc.encode(ins.text, c)
                ours += 1
            except (EncodeError, KeyError):
                w = ins.word  # fall back to ptxas's word for this slot
            words.append(w)
        n_insn += len(words)
        n_ours += ours
        full += ours == len(words)
        e = Elf.parse(cubin.read_bytes())
        sec = e.section(f".text.{kname}")
        new = words_to_bytes(words)
        if len(new) != sec.size:
            continue  # listing didn't cover the whole section (multiple functions); skip
        e.replace_data(sec.name, new)
        (out / f"{kname}.cubin").write_bytes(e.serialize())
        names.append(kname)

    # Disassemble all rewritten cubins in the VM and compare against the originals.
    script = (f"cd {shlex.quote(str(out))} && for f in *.cubin; do "
              f"nvdisasm -c $f > ${{f%.cubin}}.sass 2>${{f%.cubin}}.err; "
              f"cuobjdump -elf $f > /dev/null 2>>${{f%.cubin}}.err; done; true")
    oracle.run_tool("bash", "-c", script)
    same = 0
    diffs = []
    for k in names:
        a = [i.text for i in oracle.parse_listing((corpus / "sass" / f"{k}.sass").read_text())]
        raw = (out / f"{k}.sass").read_text()
        b = [ln.split("*/", 1)[1].split(";")[0].strip() for ln in raw.splitlines()
             if ln.strip().startswith("/*") and "*/" in ln and ln.split("*/", 1)[1].strip()]
        err = (out / f"{k}.err").read_text().strip()
        if a == b and not err:
            same += 1
        elif len(diffs) < 5:
            diffs.append((k, err[:120] or next(((x, y) for x, y in zip(a, b) if x != y), "length")))
    print(f"{arch}: {n_ours}/{n_insn} instructions encoded by us ({100 * n_ours / n_insn:.1f}%), "
          f"{full}/{len(names)} kernels fully ours")
    print(f"rewritten cubins accepted by nvdisasm+cuobjdump with identical disassembly: {same}/{len(names)}")
    for d in diffs:
        print("  diff:", d)


if __name__ == "__main__":
    main(*sys.argv[1:])
