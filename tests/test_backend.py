"""Encoder, ELF/.nv.info containers, and div-by-constant sequences (offline)."""

import json
import random
from pathlib import Path

import pytest

from openassembler.container import nvinfo
from openassembler.container.elf import Elf
from openassembler.isa.encode import EncodeError, Encoder
from openassembler.isa.word import control
from openassembler.opt.divconst import lower, run

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus" / "sm_80"
needs_corpus = pytest.mark.skipif(not (CORPUS / "pairs.jsonl").exists(), reason="build the corpus first")


@needs_corpus
def test_elf_roundtrip_all_cubins():
    files = sorted((CORPUS / "cubin").glob("*.cubin"))
    assert files
    for f in files:
        b = f.read_bytes()
        assert Elf.parse(b).serialize() == b, f.name


@needs_corpus
def test_nvinfo_roundtrip_and_values():
    e = Elf.parse((CORPUS / "smoke.cubin").read_bytes())
    recs = nvinfo.parse(e.section(".nv.info.saxpy").data)
    assert nvinfo.serialize(recs) == e.section(".nv.info.saxpy").data
    by = {r.name: r for r in recs}
    assert nvinfo.value(by["CBANK_PARAM_SIZE"]) == 0x18
    assert nvinfo.u32s(by["EXIT_INSTR_OFFSETS"]) == [0x50, 0xE0]


@needs_corpus
def test_encoder_never_emits_wrong_word_on_corpus():
    """Every word the encoder produces from a corpus text equals ptxas's word
    (refusals are allowed; wrong answers are not)."""
    enc = Encoder("sm_80")
    seen = {}
    for line in (CORPUS / "pairs.jsonl").open():
        r = json.loads(line)
        seen.setdefault(int(r["word"], 16), r["text"])
    produced = 0
    for w, t in seen.items():
        if "`" in t:
            continue  # label operands go through encode_branch
        try:
            assert enc.encode(t, control(w)) == w, t
            produced += 1
        except EncodeError:
            pass
    assert produced > 1000


@pytest.mark.parametrize("op,signed", [("div", False), ("div", True), ("rem", False), ("rem", True)])
def test_divconst_sequences_random(op, signed):
    """Fast random check; the full proof is tools/verify_divconst_exhaustive.py (all 2^32 x)."""
    rng = random.Random(0)
    for d in [2, 3, 7, 10, 641, 1000, 0x1234, 1000000007] + [rng.randrange(2, 1 << 31) for _ in range(20)]:
        seq = lower(op, signed, d)
        for _ in range(500):
            x = rng.getrandbits(32)
            if signed:
                sx = x - (1 << 32) if x >> 31 else x
                q = abs(sx) // d * (1 if sx >= 0 else -1)
                want = (q if op == "div" else sx - q * d) & 0xFFFFFFFF
            else:
                want = x // d if op == "div" else x % d
            assert run(seq, x) == want, (op, signed, d, x)
