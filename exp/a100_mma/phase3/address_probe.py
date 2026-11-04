"""Discover LDGSTS immediate fields with preserved nvdisasm bit evidence."""
from __future__ import annotations

import json
import random
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from openassembler import oracle
from openassembler.isa.word import control, set_bits


def roundtrips(evidence):
    seed = int(evidence["seed_word"], 16)
    rng = random.Random(1729)
    cases = [(0, 0, dst, src) for dst in (-524288, -16, 0, 16, 524272)
             for src in (-2048, -16, 0, 16, 2032)]
    cases += [(rng.randrange(255), rng.randrange(127)*2,
               rng.randrange(-32768, 32768)*16, rng.randrange(-128, 128)*16)
              for _ in range(64)]
    words = []
    expected = []
    def displacement(value):
        return (f"+-0x{-value:x}" if value < 0 else f"+0x{value:x}") if value else ""
    for dstreg, srcreg, dstoff, srcoff in cases:
        word = set_bits(set_bits(seed, 16, 8, dstreg), 24, 8, srcreg)
        word = set_bits(set_bits(word, 32, 12, srcoff & 4095),
                        44, 20, dstoff & 1048575)
        # Everything outside the two register and two immediate fields remains
        # exactly the seed, including modifiers, predicate and all controls.
        operand_mask = ((1 << 48) - 1) << 16
        assert (word & ~operand_mask) == (seed & ~operand_mask)
        assert control(word) == control(seed)
        words.append(word)
        expected.append(f"LDGSTS.E.BYPASS.128 [R{dstreg}{displacement(dstoff)}], "
                        f"[R{srcreg}.64{displacement(srcoff)}]")
    actual = oracle.disasm_words(words, max_rounds=129)
    assert actual == expected, [(i, want, got) for i, (want, got)
                                in enumerate(zip(expected, actual)) if want != got]
    rows = [{"destination_register": dstreg, "source_register": srcreg,
             "shared_offset": dstoff, "global_offset": srcoff,
             "word": f"{word:032x}", "text": got,
             "control": vars(control(word))}
            for (dstreg, srcreg, dstoff, srcoff), word, got in zip(cases, words, actual)]
    evidence["immediate_fields"] = {
        "global": {"low_bit": 32, "width": 12, "signed": True, "unit": "byte"},
        "shared": {"low_bit": 44, "width": 20, "signed": True, "unit": "byte"},
        "geometry_4096_required_global_chunk_offsets": [i*32*4096 for i in range(8)],
        "geometry_4096_required_shared_chunk_offsets": [i*2048 for i in range(8)],
        "global_chunk_folding_fits": False,
        "shared_chunk_folding_fits": True,
    }
    evidence["roundtrips"] = rows
    target = HERE / "address-ldgsts-roundtrips.json"
    target.write_text(json.dumps(evidence, indent=2) + "\n")
    print(f"{target}: {len(rows)} exact roundtrips; controls and nonoperand bits preserved")


def discover():
    listing = ROOT / "exp/a100_mma/build/p3_epilogue_confirm_square_f16_scalar/sass/epilogue_confirm_square_f16_scalar.sass"
    seed = next(insn.word for insn in oracle.parse_listing(listing.read_text())
                if insn.text.startswith("LDGSTS.E.BYPASS.128"))
    seed = set_bits(set_bits(seed, 16, 8, 16), 24, 8, 8)
    baseline = oracle.disasm_words([seed])[0]
    # All 128 bits are recorded, including opcode, operand and control bits.
    words = [seed ^ (1 << bit) for bit in range(128)]
    texts = oracle.disasm_words(words, max_rounds=129)
    evidence = {"seed_word": f"{seed:032x}", "seed_text": baseline,
                "seed_control": vars(control(seed)),
                "bit_flips": [{"bit": bit, "word": f"{word:032x}",
                               "text": text, "control": vars(control(word))}
                              for bit, word, text in zip(range(128), words, texts)]}
    target = HERE / "address-ldgsts-fields.json"
    target.write_text(json.dumps(evidence, indent=2) + "\n")
    print(target)
    for row in evidence["bit_flips"]:
        if row["text"] != baseline:
            print(row["bit"], row["text"])
    return evidence


if __name__ == "__main__":
    target = HERE / "address-ldgsts-fields.json"
    evidence = json.loads(target.read_text()) if target.exists() else discover()
    roundtrips(evidence)
