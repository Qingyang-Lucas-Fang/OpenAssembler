# sm_80 encoding: measured findings

Everything here is measured against nvdisasm (CUDA 12.8) on our own probe corpus
(`corpus/sm_80/pairs.jsonl`: 541 PTX probes, 13,816 instructions, 1,459 unique
words, 88 opcodes) and the bit-flip run (`tools/bitflip.py` → `specs/sm_80/bitflip.json`:
137 seeds × 128 bits). Tests: `tests/test_word.py`.

## Shared fields (all opcodes)

| Bits | Field | Evidence |
|---|---|---|
| 0–8 | opcode | 88 values, each maps to exactly one mnemonic (0 collisions) |
| 9–11 | operand form: 1 = reg, 4 = imm32, 5 = c[bank][off], 6 = UR (per ALU op) | bit-flip: these bits toggle operand b's kind |
| 12–14, 15 | guard predicate (7 = PT), negate | matches nvdisasm on 1,459/1,459 words |
| 16–23 | Rd | matches first R operand on 197/197 mnemonics |
| 24–31 | Ra | holds for ALU ops; stores, atomics, SHFL, unary ops (IABS, MUFU, F2I, I2F...) use other slots |
| 32–39 / 32–63 | Rb / imm32 | bit-flip |
| 38–53, 54–58 | cbank byte offset, bank | matches all 137 cbank operands |
| 91 | cbank addressing via uniform register: `c[..]` → `cx[URx][..]` | bit-flip on 24 seeds |
| 64–71 | Rc | bit-flip |
| 72–104 | per-opcode modifiers and extra operands (predicate dst at 81–83/84–86, predicate src at 87–90) | bit-flip |

Branch offsets: BRA/CALL/RET hold a signed offset at bits 34–81 (relative to the next
instruction); BSSY uses 34–63.

## Control bits (105–125)

| Bits | Field |
|---|---|
| 105–108 | stall |
| 109 | yield, active-low: 0 = yield. Confirmed by nvdisasm's sm_120 annotations (0 → `?WAITn_END_GROUP`, 1 → `?transN`). Mesa NAK's `yld` names the raw bit. |
| 110–112 | write scoreboard (7 = none) |
| 113–115 | read scoreboard (7 = none) |
| 116–121 | wait mask |
| 122–125 | reuse flags (nvdisasm prints `.reuse` on operands) |

- Validity rule (new): nvdisasm rejects **no-yield (bit 109 = 1) with stall 0 or 12–15**.
  With bit 109 = 0 (yield) every stall value is accepted. Swept on 8 seeds of different
  opcodes, and every compiler-emitted word in the corpus obeys it. The sm_120 listing
  agrees: `?transN` only appears with N between 1 and 7, and `?WAIT13_END_GROUP` exists.
- Scoreboard flow checks out: 1,548 waits, and 1,541 of them have an earlier setter in
  straight-line order. The other 7 are loop back-edges and need the CFG to check.
- The layout is the same on sm_75, 86, 90, 100 and 120 (docs/research/02). ptxas still sets bits 109 and 122–125 on sm_120, even though Mesa NAK leaves them unset.

## Open items

- Per-opcode operand maps: turn bitflip runs into `specs/sm_80/opcodes.json`, then
  sweep each modifier field to get its value names.
- Uniform-predicate guards (`@UPx`) and uniform datapath ops (bit 91 on non-cbank ops).
- Repeat on sm_86, sm_89, sm_90, sm_100 and sm_120, and diff the results.
