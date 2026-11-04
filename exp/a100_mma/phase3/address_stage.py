"""Specialize a safe two-stage copy/compute loop to alternating constant stages.

Pairs consume two tiles per loop; the last two tiles are explicitly peeled.
The original wait/copy/compute/barrier order is retained for every tile, and
there are no copies past K. Only full, even K64 tile counts are accepted.
"""
from __future__ import annotations

import json
from pathlib import Path


def specialize_two_stage(source: str, *, k: int, bm=128, bn=128, bk=64):
    stride = 2*bk*(bm+bn)
    if k % (2*bk) or k < 2*bk or stride & (stride-1):
        raise ValueError("require an even number of full K tiles and power-of-two stage stride")
    if source.count("LOOP:\n") != 1:
        raise ValueError("expected one standard two-stage loop")
    prefix, rest = source.split("LOOP:\n")
    backedge = f"setp.lt.u32 %p,%r8,{k-bk}; @%p bra LOOP;\n"
    counter = f"add.u32 %r8,%r8,{bk}; xor.b32 %r9,%r9,{stride};\n"
    toggle = f"xor.b32 %r10,%r9,{stride};\n"
    if rest.count(backedge) != 1:
        raise ValueError("unexpected loop backedge")
    body, tail = rest.split(backedge)
    if not body.endswith(counter) or body.count(toggle) != 1:
        raise ValueError("unexpected loop counter/stage recurrence")
    body = body[:-len(counter)].replace(toggle, "")
    marker = "shr.u32 %r50,%r3,2; add.u32 %r50,%r50,%r5; add.u32 %r50,%r50,%r1;"
    if tail.count(marker) != 1:
        raise ValueError("unexpected peeled epilogue boundary")
    drain, epilogue = tail.split(marker)
    if not body.startswith("cp.async.wait_group 0; bar.sync 0;\n"):
        raise ValueError("missing copy-completion wait and CTA barrier")
    if body.count("cp.async.commit_group;") != 1 or body.count("bar.sync 0;") != 2:
        raise ValueError("body must contain one copy group and two CTA barriers")
    if "cp.async.cg.shared.global" in drain or drain.count("bar.sync 0;") != 2:
        raise ValueError("drain must compute one tile without a speculative copy")

    def specialize(block, current, following):
        block = block.replace(",%r9;", f",{current};").replace(",%r10;", f",{following};")
        return block

    first = specialize(body, 0, stride)
    second = specialize(body, stride, 0)
    result = specialize(prefix, 0, 0)
    if k > 2*bk:
        result += ("LOOP_STAGE_PAIR:\n" + first + second +
                   f"add.u32 %r8,%r8,{2*bk}; "
                   f"setp.lt.u32 %p,%r8,{k-2*bk}; @%p bra LOOP_STAGE_PAIR;\n")
    # Stage zero holds tile T-2. The first peeled body copies tile T-1 into
    # stage one; the final drain consumes it after its completion wait.
    result += first + specialize(drain, stride, 0) + marker + epilogue
    if ",%r9;" in result or ",%r10;" in result:
        raise ValueError("a dynamic stage operand escaped specialization")
    return result


if __name__ == "__main__":
    import address_shared
    address_shared.search.backend.install()
    suite = []
    for dtype in ("f16", "bf16"):
        candidate = address_shared.compile_case(4096, 4096, 4096, dtype, stage_pair=True)
        root = dict(candidate, build=f"search_combined_square_row_bpair_{dtype}",
                    name=f"address_stage_square_{dtype}_paired", steady_warmup_rounds=16,
                    alternatives=[dict(candidate, build=f"address_shared_square_{dtype}",
                                       name=f"address_shared_square_{dtype}"), candidate])
        suite.append(root)
    target = Path(__file__).resolve().parent/"address-stage-suite.json"
    target.write_text(json.dumps(suite, indent=2)+"\n")
    print(target)
