"""Bundled SM80 tensor-path encoding seeds and learned operand fields.

Seeds are OA probe encodings verified by nvdisasm, not copied vendor kernels.
Only the measured narrow tensor path is added. Runtime encoding does not run
an oracle, discover fields, write caches or bootstrap a generated corpus.
"""
from __future__ import annotations

from openassembler.emu.operands import Imm, Mem
from openassembler.emu.program import parse_insn
from openassembler.isa.encode import Encoder, EncodeError, Template, key
from openassembler.isa.word import set_bits

# Input hashes make the experimental field/seed evidence auditable.
EVIDENCE = {'corpus/sm_80/pairs.jsonl': 'c619303020b016b27eb8b93eb68652563f11c7a7a78bf7eaba4eca5d172bb491', 'specs/sm_80/synth_templates.json': 'd6debb40dba5b8c74c45d3e6da608dad5ac30fbe196a11907242e8248708ac24', 'specs/sm_80/arithmetic_templates.json': '1c3fc5163ab05bc4305260c12d51bfc270ab83d89fbb487fcbe5854c9179bfe1', 'docs/evidence/a100-mma/encoding/templates.json': 'dc8057a8d83e027c8a90959d60f3fb3e246753e1f9b4bdc0ea09a51f7bdee82f', 'docs/evidence/a100-mma/encoding/address-ldgsts-roundtrips.json': '5168b037a21ffd74cfa9d6bc4644a7bd7acd471d4862b3145f59b5e507199044', 'docs/evidence/a100-mma/encoding/depbar-fields.json': '4d247c2c75c230d183cddb05135d20ea68e1d4697bbfeba2f01f4bdb8db6aab6', 'docs/evidence/a100-mma/encoding/fieldmaps.json': '926822ee069b8eef6c0ad75bd79b1cc4e2af4dd33ccc898d2f06a3749bd806a0'}

# (verified seed text, 128-bit word, learned operand slots)
_TEMPLATES = (
    ('MOV R1, c[0x0][0x28]', '000fe40000000f0000000a0000017a02', (('reg', 'rd'), ('cbank', 'cb'))),
    ('S2R R4, SR_TID.X', '000e2200000021000000000000047919', (('reg', 'rd'), ('sr', 'sr'))),
    ('ULDC.64 UR4, c[0x0][0x118]', '000fd20000000a000000460000047ab9', (('reg', 'urd'), ('cbank', 'cb'))),
    ('IMAD.WIDE.U32 R2, R4, R5, c[0x0][0x168]', '001fca00078e000500005a0004027625', (('reg', 'rd'), ('reg', 'ra'), ('reg', 'rc'), ('cbank', 'cb'))),
    ('STG.E.64 [R4.64], R6', '008fe2000c101b040000000604007986', (('mem', 'mem'), ('reg', 'rb'))),
    ('LDG.E.64 R6, [R2.64+0x8]', '000ea2000c1e1b000000080402067981', (('reg', 'rd'), ('mem', 'mem'))),
    ('FADD R10, R8, R9', '004fca000000000000000009080a7221', (('reg', 'rd'), ('reg', 'ra'), ('reg', 'rb'))),
    ('FMNMX R10, R8, R9, !PT', '004fca000780000000000009080a7209', (('reg', 'rd'), ('reg', 'ra'), ('reg', 'rb'), ('pred', 'p87'))),
    ('FMNMX R6, R0, 1.5, !PT', '004fca00078000003fc0000000067809', (('reg', 'rd'), ('reg', 'ra'), ('imm', 'imm32'), ('pred', 'p87'))),
    ('EXIT', '000fea0003800000000000000000794d', ()),
    ('BRA `(.L_x_0)', '000fc0000383fffffffffff000007947', (('other', None),)),
    ('NOP', '000fc000000000000000000000007918', ()),
    ('IADD3.X R9, R6, R7, RZ, P0, !PT', '000fca00007fe4ff0000000706097210', (('reg', 'rd'), ('reg', 'ra'), ('reg', 'rb'), ('reg', 'rc'), ('pred', 'p87'), ('pred', None))),
    ('MOV R2, R4', '004fe20000000f000000000400027202', (('reg', 'rd'), ('reg', 'rb'))),
    ('IADD3 R6, R0, 0x1234, RZ', '004fca0007ffe0ff0000123400067810', (('reg', 'rd'), ('reg', 'ra'), ('imm', 'imm32'), ('reg', 'rc'))),
    ('IADD3 R10, R8, R9, RZ', '004fca0007ffe0ff00000009080a7210', (('reg', 'rd'), ('reg', 'ra'), ('reg', 'rb'), ('reg', 'rc'))),
    ('LOP3.LUT R6, R0, 0x1234, RZ, 0xc0, !PT', '004fca00078ec0ff0000123400067812', (('reg', 'rd'), ('reg', 'ra'), ('imm', 'imm32'), ('reg', 'rc'), ('imm2', (72, 8)), ('pred', 'p87'))),
    ('LOP3.LUT R10, R8, R9, RZ, 0xc0, !PT', '004fca00078ec0ff00000009080a7212', (('reg', 'rd'), ('reg', 'ra'), ('reg', 'rb'), ('reg', 'rc'), ('imm2', (72, 8)), ('pred', 'p87'))),
    ('BAR.SYNC.DEFER_BLOCKING 0x0', '000fec00000100000000000000007b1d', (('imm', 'imm32'),)),
    ('MOV R8, 0xa', '001fe20000000f000000000a00087802', (('reg', 'rd'), ('imm', 'imm32'))),
    ('LDGDEPBAR', '000e22000000000000000000000079af', ()),
    ('SHF.R.U32.HI R8, RZ, 0x17, R5.reuse', '100fe2000001160500000017ff087819', (('reg', 'rd'), ('reg', 'ra'), ('imm', 'imm32'), ('reg', 'rc'))),
    ('IMAD R10, R8, -0x800000, R10', '000fe400078e020aff800000080a7824', (('reg', 'rd'), ('reg', 'ra'), ('imm', 'imm32'), ('reg', 'rc'))),
    ('IMAD R13, R10, R11, RZ', '000fca00078e02ff0000000b0a0d7224', (('reg', 'rd'), ('reg', 'ra'), ('reg', 'rb'), ('reg', 'rc'))),
    ('IMAD.HI.U32 R7, R7, R13, R6', '000fe200078e00060000000d07077227', (('reg', 'rd'), ('reg', 'ra'), ('reg', 'rb'), ('reg', 'rc'))),
    ('IMAD R11, R11, c[0x0][0x170], RZ', '000fca00078e02ff00005c000b0b7a24', (('reg', 'rd'), ('reg', 'ra'), ('cbank', 'cb'), ('reg', 'rc'))),
    ('STG.E [R4.64+0x8], R7', '0001e8000c1019040000080704007986', (('mem', 'mem'), ('reg', 'rb'))),
    ('LDSM.16.M88.4 R12, [R0]', '000e28000000020000000000000c783b', (('reg', 'rd'), ('mem', 'mem+scale'))),
    ('IMAD.HI.U32 R10, R8, c[0x0][0x170], R2', '000fca00078e000200005c00080a7a27', (('reg', 'rd'), ('reg', 'ra'), ('cbank', 'cb'), ('reg', 'rc'))),
    ('IMAD.HI.U32 R10, R8, 0x1234, R2', '000fca00078e000200001234080a7827', (('reg', 'rd'), ('reg', 'ra'), ('imm', 'imm32'), ('reg', 'rc'))),
    ('HMMA.16816.F32.BF16 R8, R4, R4, R8', '000fee0000041808000000040408723c', (('reg', 'rd'), ('reg', 'ra'), ('reg', 'rb'), ('reg', 'rc'))),
    ('HMMA.16816.F32 R8, R4, R4, R8', '000fee0000001808000000040408723c', (('reg', 'rd'), ('reg', 'ra'), ('reg', 'rb'), ('reg', 'rc'))),
    ('IMAD.WIDE.U32 R6, R8, R10, RZ', '000fc800078e00ff0000000a08067225', (('reg', 'rd'), ('reg', 'ra'), ('reg', 'rb'), ('reg', 'rc'))),
    ('IMAD.WIDE R6, R2, c[0x0][0x170], RZ', '004fca00078e02ff00005c0002067a25', (('reg', 'rd'), ('reg', 'ra'), ('cbank', 'cb'), ('reg', 'rc'))),
    ('IMAD.WIDE R6, R2, 0x1234, RZ', '004fca00078e02ff0000123402067825', (('reg', 'rd'), ('reg', 'ra'), ('imm', 'imm32'), ('reg', 'rc'))),
    ('IMAD.WIDE R6, R2, R3, RZ', '004fca00078e02ff0000000302067225', (('reg', 'rd'), ('reg', 'ra'), ('reg', 'rb'), ('reg', 'rc'))),
    ('IMAD.WIDE.U32 R6, R2, c[0x0][0x170], RZ', '004fca00078e00ff00005c0002067a25', (('reg', 'rd'), ('reg', 'ra'), ('cbank', 'cb'), ('reg', 'rc'))),
    ('IMAD.WIDE.U32 R6, R2, 0x1234, RZ', '004fca00078e00ff0000123402067825', (('reg', 'rd'), ('reg', 'ra'), ('imm', 'imm32'), ('reg', 'rc'))),
    ('IADD3 R2, P0, R8, 0x1, RZ', '001fca0007f1e0ff0000000108027810', (('reg', 'rd'), ('pred', 'p81'), ('reg', 'ra'), ('imm', 'imm32'), ('reg', 'rc'))),
    ('ISETP.LT.U32.AND P0, PT, R3, 0x1, PT', '001fda0003f01070000000010300780c', (('pred', 'p81'), ('pred', 'p84'), ('reg', 'ra'), ('imm', 'imm32'), ('pred', 'p87'))),
    ('IMAD.WIDE R2, R4, R5, c[0x0][0x168]', '001fca00078e020500005a0004027625', (('reg', 'rd'), ('reg', 'ra'), ('reg', 'rc'), ('cbank', 'cb'))),
    ('LDSM.16.M88.2 R12, [R0]', '000e28000000010000000000000c783b', (('reg', 'rd'), ('mem', 'mem+scale'))),
    ('IADD3 R8, P0, R4, R5, RZ', '004fe20007f1e0ff0000000504087210', (('reg', 'rd'), ('pred', 'p81'), ('reg', 'ra'), ('reg', 'rb'), ('reg', 'rc'))),
    ('IADD3 R6, R0, c[0x0][0x170], RZ', '004fca0007ffe0ff00005c0000067a10', (('reg', 'rd'), ('reg', 'ra'), ('cbank', 'cb'), ('reg', 'rc'))),
    ('IADD3 R6, -R0, 0x1f, RZ', '001fca0007ffe1ff0000001f00067810', (('reg', 'rd'), ('reg', 'ra'), ('imm', 'imm32'), ('reg', 'rc'))),
    ('IADD3 R9, R8.reuse, 0x7f, -R9', '040fe20007ffe8090000007f08097810', (('reg', 'rd'), ('reg', 'ra'), ('imm', 'imm32'), ('reg', 'rc'))),
    ('@!P1 IADD3 R2, -R2, RZ, RZ', '000fe40007ffe1ff000000ff02029210', (('reg', 'rd'), ('reg', 'ra'), ('reg', 'rb'), ('reg', 'rc'))),
    ('@!P2 IADD3 R0, R0, -R11.reuse, RZ', '080fe40007ffe0ff8000000b0000a210', (('reg', 'rd'), ('reg', 'ra'), ('reg', 'rb'), ('reg', 'rc'))),
    ('@P0 IADD3 R0, R0, -c[0x0][0x170], RZ', '000fe40007ffe0ff80005c0000000a10', (('reg', 'rd'), ('reg', 'ra'), ('cbank', 'cb'), ('reg', 'rc'))),
    ('LDSM.16.MT88.2 R6, [R0]', '000e280000004100000000000006783b', (('reg', 'rd'), ('mem', 'mem'))),
    ('IADD3 R8, P0, R8, -R10, RZ', '004fc80007f1e0ff8000000a08087210', (('reg', 'rd'), ('pred', 'p81'), ('reg', 'ra'), ('reg', 'rb'), ('reg', 'rc'))),
    ('IADD3.X R9, R9, ~R11, RZ, P0, !PT', '000fca00007fe4ff8000000b09097210', (('reg', 'rd'), ('reg', None), ('reg', 'rb'), ('reg', 'rc'), ('pred', 'p87'), ('pred', None))),
    ('IADD3 R9, R4, R6, -R9', '000fca0007ffe8090000000604097210', (('reg', 'rd'), ('reg', 'ra'), ('reg', 'rb'), ('reg', 'rc'))),
)


class TensorEncoder(Encoder):
    """Template-only encoder plus the two measured special field layouts."""

    def __init__(self):
        # Deliberately bypass Encoder.__init__: tensor compilation must work
        # without a generated corpus or any field-discovery subprocesses.
        self.templates = {}
        self.corpus_words = ()
        for text, word, slots in _TEMPLATES:
            ins = parse_insn(0, text, int(word, 16))
            self.templates[key(ins)] = Template(ins.word, slots, tuple(ins.ops))

    def encode(self, text, ctl=None):
        ins = parse_insn(0, text.replace("+-", "-"), 0)
        if ins.op == "LDGSTS":
            if (ins.guard is not None or ins.mods != ("E", "BYPASS", "128")
                    or len(ins.ops) != 2):
                raise EncodeError("unsupported SM80 tensor async-copy form")
            dst, src = ins.ops
            if (not all(isinstance(arg, Mem) and arg.base is not None
                        and arg.base.file == "R" and arg.ur is None and arg.scale == 1
                        for arg in (dst, src))
                    or dst.wide or not src.wide or src.base.idx % 2
                    or not -524288 <= dst.off < 524288 or dst.off % 16
                    or not -2048 <= src.off < 2048 or src.off % 16):
                raise EncodeError("unproven LDGSTS register/displacement combination")
            # address-ldgsts-roundtrips.json: 89 checked combinations plus
            # independent flips identify Rdst16:23,Rsrc24:31,source32:43,
            # destination44:63. Both displacements are signed byte offsets.
            word = int("0009e4000b901c440000000008107fae", 16)
            word = set_bits(set_bits(word, 16, 8, dst.base.idx), 24, 8, src.base.idx)
            word = set_bits(set_bits(word, 32, 12, src.off & 4095), 44, 20, dst.off & 1048575)
            return self._ctl(word, ctl) if ctl is not None else word
        if ins.op == "DEPBAR":
            if (ins.guard is not None or ins.mods != ("LE",) or len(ins.ops) != 2
                    or ins.ops[0].text != "SB0" or not isinstance(ins.ops[1], Imm)
                    or ins.ops[1].int32() not in (0, 1, 2)):
                raise EncodeError("only DEPBAR.LE SB0 thresholds 0,1,2 are supported")
            # The retained encoding evidence independently identifies bits38:43.
            word = set_bits(int("000fc80000000000000080000000791a", 16), 38, 6,
                            ins.ops[1].int32())
            return self._ctl(word, ctl) if ctl is not None else word
        return super().encode(text, ctl)
