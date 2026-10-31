"""Generic field decoding, checked against real sm_80 words (corpus/sm_80/smoke.cubin)."""

import json
import re
from pathlib import Path

import pytest

from openassembler.isa.word import Control, bits, control, control_valid, fields, set_bits

W = lambda lo, hi: lo | (hi << 64)  # noqa: E731

MOV = W(0x00000A0000017A02, 0x000FE40000000F00)      # MOV R1, c[0x0][0x28]
S2R = W(0x0000000000047919, 0x000E280000002500)      # S2R R4, SR_CTAID.X
IMAD = W(0x0000000004047A24, 0x001FCA00078E0203)     # IMAD R4, R4, c[0x0][0x0], R3
ISETP = W(0x0000580004007A0C, 0x000FDA0003F06270)    # ISETP.GE.AND P0, PT, R4, c[0x0][0x160], PT
EXIT_P0 = W(0x000000000000094D, 0x000FEA0003800000)  # @P0 EXIT

CORPUS = Path(__file__).resolve().parent.parent / "corpus" / "sm_80" / "pairs.jsonl"


def test_bits_roundtrip():
    w = set_bits(0, 105, 4, 0xB)
    assert bits(w, 105, 4) == 0xB and w == 0xB << 105


def test_mov_cbank():
    f = fields(MOV)
    assert (f.opcode, f.form, f.pred, f.pred_neg, f.rd) == (0x002, 5, 7, False, 1)
    assert (f.cb_bank, f.cb_off) == (0, 0x28)


def test_imad_operands():
    f = fields(IMAD)
    assert (f.opcode, f.rd, f.ra, f.cb_off, f.rc) == (0x024, 4, 4, 0x0, 3)


def test_isetp_param_offset():
    assert fields(ISETP).cb_off == 0x160  # first kernel param on sm_80


def test_guard():
    f = fields(EXIT_P0)
    assert (f.pred, f.pred_neg) == (0, False)


def test_scoreboard_dependency():
    # S2R writes through scoreboard 0; the IMAD consuming R4 waits on it.
    assert control(S2R).wbar == 0
    assert control(IMAD).wait & 1
    assert control(MOV) == Control(stall=2, yield_=False, wbar=7, rbar=7, wait=0, reuse=0)


def test_yield_polarity():
    # sm_120 words where nvdisasm prints the yield state itself:
    # bit 109 = 1 -> "?transN" (no yield), bit 109 = 0 -> "?WAITn_END_GROUP" (yield).
    trans = set_bits(set_bits(0, 105, 4, 1), 109, 1, 1)
    end_group = set_bits(0, 105, 4, 5)
    assert not control(trans).yield_ and control(end_group).yield_


@pytest.mark.skipif(not CORPUS.exists(), reason="run tools/build_corpus.py sm_80 first")
def test_corpus_control_valid():
    """Every word ptxas emitted satisfies the measured stall/yield rule."""
    for line in CORPUS.open():
        assert control_valid(control(int(json.loads(line)["word"], 16))), line


@pytest.mark.skipif(not CORPUS.exists(), reason="run tools/build_corpus.py sm_80 first")
def test_corpus_guard_and_cbank():
    """Guard predicate and constant-bank operands match nvdisasm on every corpus word."""
    seen = {}
    for line in CORPUS.open():
        r = json.loads(line)
        seen[int(r["word"], 16)] = r["text"]
    for w, text in seen.items():
        f = fields(w)
        g = re.match(r"@(!?)P([T0-6])\s", text)
        if text.startswith("@") and "UP" in text.split()[0]:
            continue  # uniform-predicate guards: encoding not yet characterized
        exp = (7, False) if not g else (7 if g[2] == "T" else int(g[2]), g[1] == "!")
        assert (f.pred, f.pred_neg) == exp, text
        for bank, off in re.findall(r"c\[0x([0-9a-f]+)\]\[0x([0-9a-f]+)\]", text):
            assert (f.cb_bank, f.cb_off) == (int(bank, 16), int(off, 16)), text
