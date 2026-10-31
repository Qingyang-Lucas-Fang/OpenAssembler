"""Build corpus/<arch>/: probe PTX -> cubin -> (word, text) pairs.

Usage: uv run --no-project python tools/build_corpus.py sm_80 [sm_86 sm_90 ...]

Outputs per arch:
  ptx/<probe>.ptx, cubin/<probe>.cubin   generated, never edited
  failed.txt                             probes ptxas rejected, with its message
  pairs.jsonl                            {"probe","addr","word" (hex), "text"} per instruction
"""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ptx_probes  # noqa: E402
from openassembler import oracle  # noqa: E402

# Runs inside the VM: compile + disassemble every probe, 4 at a time.
# A probe's .sass holds nvdisasm output; .err holds ptxas's complaint.
VM_SCRIPT = r"""
cd {dir} && ls ptx/*.ptx | xargs -P4 -I{{}} sh -c '
  b=$(basename {{}} .ptx)
  if ptxas -arch={target} -O3 -o cubin/$b.cubin {{}} 2> sass/$b.err; then
    nvdisasm -hex -c cubin/$b.cubin > sass/$b.sass 2>> sass/$b.err && rm -f sass/$b.err
  fi'
"""


def build(target: str) -> None:
    out = oracle.ROOT / "corpus" / target
    for sub in ("ptx", "cubin", "sass"):
        (out / sub).mkdir(parents=True, exist_ok=True)
        for f in (out / sub).iterdir():
            f.unlink()

    names = {}
    for name, body, shared in ptx_probes.probes(target):
        kname = ptx_probes.kernel_name(name)
        names[kname] = name
        (out / "ptx" / f"{kname}.ptx").write_text(ptx_probes.module(target, name, body, shared))

    oracle.run_tool("bash", "-c", VM_SCRIPT.format(dir=shlex.quote(str(out)), target=target))

    failed, n_insn = [], 0
    with open(out / "pairs.jsonl", "w") as f:
        for kname, name in sorted(names.items()):
            err = out / "sass" / f"{kname}.err"
            if err.exists():
                failed.append(f"{name}: {err.read_text().strip().splitlines()[0]}")
                continue
            for ins in oracle.parse_listing((out / "sass" / f"{kname}.sass").read_text()):
                f.write(json.dumps({"probe": name, "addr": ins.addr,
                                    "word": f"{ins.word:032x}", "text": ins.text}) + "\n")
                n_insn += 1
    (out / "failed.txt").write_text("\n".join(failed) + ("\n" if failed else ""))
    print(f"{target}: {len(names) - len(failed)}/{len(names)} probes compiled, {n_insn} instructions")


if __name__ == "__main__":
    for t in sys.argv[1:] or ["sm_80"]:
        build(t)
