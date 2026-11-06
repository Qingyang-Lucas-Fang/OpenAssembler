"""Mine WARPSYNC register-mask template from ptxas's own -O0 sync helper.

Run once with the local toolkit. No instruction bits are supplied by hand.
Template register forms are independently swept through nvdisasm.
"""
from pathlib import Path
import hashlib
import json


def generate():
    from openassembler import oracle
    from openassembler.isa import encode
    from openassembler.emu.program import parse_insn
    root=Path(__file__).resolve().parent
    source=root/"reductions_warpsync.ptx"
    cubin=root/"reductions_warpsync_o0.cubin"
    oracle.ptxas(source,cubin,"sm_80","-O0")
    rows=[{"word":f"{i.word:032x}","text":i.text}
          for i in oracle.disasm_cubin(cubin) if i.text.startswith("WARPSYNC ")]
    if len(rows)<2:raise RuntimeError("ptxas did not expose distinct WARPSYNC mask registers")
    members=[parse_insn(0,r["text"],int(r["word"],16)) for r in rows]
    template=encode.Encoder._learn(members)
    if template.slots != [("reg","rb")]:
        raise RuntimeError(f"unproven WARPSYNC register field: {template.slots}")
    encoder=encode.Encoder("sm_80")
    encoder.templates[encode.key(members[0])]=template
    regs=(0,2,4,8,15,31,64,127,254)
    words=[encoder.encode(f"WARPSYNC R{reg}") for reg in regs]
    decoded=oracle.disasm_words(words)
    expected=[f"WARPSYNC R{reg}" for reg in regs]
    if decoded != expected:raise RuntimeError((decoded,expected))
    # Cache the bit-flip ownership map under exp before any production promotion.
    old=encode.FIELDMAP_CACHE
    encode.FIELDMAP_CACHE=root/"reductions_warpsync_fieldmaps.json"
    try:
        for member in members:encode.field_probe(member.word)
    finally:encode.FIELDMAP_CACHE=old
    evidence={"ptx_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
              "ptxas_flags":["-O0","-arch=sm_80"],"templates":rows,
              "register_sweep":dict(zip(expected,decoded))}
    (root/"reductions_warpsync_evidence.json").write_text(json.dumps(evidence,indent=2)+"\n")
    return evidence


def install():
    from openassembler.build import encoder
    from openassembler.emu.program import parse_insn
    from openassembler.isa.encode import Encoder,key
    evidence=json.loads((Path(__file__).resolve().parent/"reductions_warpsync_evidence.json").read_text())
    members=[parse_insn(0,r["text"],int(r["word"],16)) for r in evidence["templates"]]
    encoder("sm_80").templates[key(members[0])]=Encoder._learn(members)


if __name__ == "__main__":
    print(json.dumps(generate(),indent=2))
