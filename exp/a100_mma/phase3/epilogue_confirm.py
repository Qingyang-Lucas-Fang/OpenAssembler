"""Preserved paired epilogue confirmation builds for six projection cases."""
from __future__ import annotations
import json
from pathlib import Path

from epilogue import BASE, HERE, backend, compile_variant


if __name__ == "__main__":
    backend.install()
    suites=[]
    for dtype in ("f16","bf16"):
        for tag,m,n,k in (("square",4096,4096,4096),
                          ("mlp",4096,16384,4096),
                          ("tall",16384,4096,4096)):
            configs=[]
            for variant in ("scalar","row_native"):
                name=f"confirm_{tag}_{dtype}_{variant}"
                configs.append(compile_variant(name,m,n,k,dtype,variant))
            root=configs[0]
            root["name"]=f"p3_epilogue_confirm_{tag}_{dtype}"
            root["steady_warmup_rounds"]=16
            root["alternatives"]=configs[1:]
            suites.append(root)
    target=HERE/"epilogue-confirm-suite.json"
    target.write_text(json.dumps(suites,indent=2)+"\n")
    print(target)
