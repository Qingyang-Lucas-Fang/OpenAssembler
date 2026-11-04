"""Build the cubin variants used to find why our cubins faulted on the A100.

    PYTHONPATH=. uv run --no-project python tools/gpu_bisect.py elf      # -> corpus/sm_80/bisect
    PYTHONPATH=. uv run --no-project python tools/gpu_bisect.py regs     # -> corpus/sm_80/bisect5
    modal run tools/gpu_modal.py --which dir:corpus/sm_80/bisect  --trials 2 --budget 120
    modal run tools/gpu_modal.py --which dir:corpus/sm_80/bisect5 --trials 1 --budget 200

elf:  ptxas cubins rewritten by our ELF writer (same bytes / grown .text / raised
      register count / re-encoded words). All matched on A100: the writer is sound.
regs: ptxas's add.s32.r kernel plus one inserted `MOV Rd, R6`, for many Rd and
      declared register counts n. Result (gpu_a100_40gb_bisect5.json): faults with
      CUDA_ERROR_ILLEGAL_INSTRUCTION iff n < Rd + 3  ->  openassembler.container.cubin.required_regcount.
Each file is "<variant>@<kernel>.cubin"; gpu_common checks it against the emulator
running ptxas's SASS for <kernel>.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openassembler import oracle  # noqa: E402
from openassembler.container import nvinfo  # noqa: E402
from openassembler.container.cubin import regcount, splice_text, text_words  # noqa: E402
from openassembler.container.elf import Elf  # noqa: E402
from openassembler.emu.operands import Label  # noqa: E402
from openassembler.emu.program import load_listing  # noqa: E402
from openassembler.isa.encode import EncodeError, Encoder  # noqa: E402
from openassembler.isa.word import Control, control  # noqa: E402

C = ROOT / "corpus" / "sm_80"


def fresh(name: str) -> Path:
    d = C / name
    d.mkdir(exist_ok=True)
    for f in d.iterdir():
        f.unlink()
    return d


def exits_of(cubin: bytes, k: str) -> list[int]:
    recs = nvinfo.parse(Elf.parse(cubin).section(f".nv.info.{k}").data)
    return next(nvinfo.u32s(r) for r in recs if r.name == "EXIT_INSTR_OFFSETS")


def build_elf(enc: Encoder) -> None:
    out = fresh("bisect")
    nop = enc.encode("NOP")
    for k in ("p_add_s32_r", "p_xor_b32_c", "p_mul_lo_s32_i"):
        orig = (C / "cubin" / f"{k}.cubin").read_bytes()
        w, n, ex = text_words(orig, k), regcount(orig, k), exits_of(orig, k)
        (out / f"orig@{k}.cubin").write_bytes(orig)
        (out / f"resplice@{k}.cubin").write_bytes(splice_text(orig, k, w, n, ex))
        (out / f"grow@{k}.cubin").write_bytes(splice_text(orig, k, w + [nop] * 8, n, ex))
        (out / f"regs@{k}.cubin").write_bytes(splice_text(orig, k, w, 40, ex))
        prog = load_listing((C / "sass" / f"{k}.sass").read_text(), kernel=k)
        ww = []
        for a in sorted(prog.insns):
            i = prog.insns[a]
            lbl = next((o for o in i.ops if isinstance(o, Label)), None)
            try:
                ww.append(enc.encode_branch(i.text, a, prog.labels[lbl.name], control(i.word)) if lbl
                          else enc.encode(i.text, control(i.word)))
            except (EncodeError, KeyError):
                ww.append(i.word)
        (out / f"reenc@{k}.cubin").write_bytes(splice_text(orig, k, ww + w[len(ww):], n, ex))
    print(f"{len(list(out.iterdir()))} variants -> {out.relative_to(ROOT)}")


def build_regs(enc: Encoder) -> None:
    out = fresh("bisect5")
    k = "p_add_s32_r"
    orig = (C / "cubin" / f"{k}.cubin").read_bytes()
    w = text_words(orig, k)
    texts = [i.text for i in oracle.parse_listing((C / "sass" / f"{k}.sass").read_text())]
    ex = texts.index("EXIT")
    ctl = Control(stall=6, yield_=False, wbar=7, rbar=7, wait=0, reuse=0)
    for d in (11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 29, 30, 31, 47, 63, 100):
        for n in range(max(12, d - 1), d + 4):
            ws = w[:ex] + [enc.encode(f"MOV R{d}, R6", ctl)] + w[ex:-1]
            (out / f"d{d:03d}n{n:03d}@{k}.cubin").write_bytes(splice_text(orig, k, ws, n, [16 * (ex + 1)]))
    print(f"{len(list(out.iterdir()))} variants -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    enc = Encoder("sm_80")
    {"elf": build_elf, "regs": build_regs}[sys.argv[1] if len(sys.argv) > 1 else "elf"](enc)
