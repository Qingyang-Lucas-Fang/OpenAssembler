"""Bounded joint warp-layout and shared-address-folding experiment.

Only warp64x32/warp32x64 and folded/unfolded copies are varied. All use the
same 128x128x64 CTA, paired B loads, row-vector epilogue, and HMMA8 schedule.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import time

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(HERE))
import address_shared as address
import search_combined as combined
from openassembler import build, oracle
from openassembler.cli import main
from openassembler.isa.word import control, control_valid


FAMILIES = (("w64x32_unfolded",64,32,False), ("w64x32_folded",64,32,True),
            ("w32x64_unfolded",32,64,False), ("w32x64_folded",32,64,True))


def make_source(m,n,k,dtype,wm,wn,folded):
    source = combined.source_for(m,n,k,dtype,bm=128,bn=128,bk=64,wm=wm,wn=wn)
    if folded:
        source = address.fold_shared_addresses(source,bm=128,bn=128,bk=64,wm=wm,wn=wn)
    return source


def context(folded):
    return address.address_backend() if folded else combined.selector_and_allocator()


def resource_model(registers,threads):
    rounded = ((registers+7)//8)*8
    register_blocks = 65536//(rounded*threads)
    shared_blocks = 2  # Two 64-KiB allocations fit the A100 shared-memory limit.
    blocks = min(register_blocks,shared_blocks,2048//threads)
    return dict(registers_per_thread=registers,rounded_registers_per_thread=rounded,
                threads=threads,register_limited_ctas=register_blocks,
                modeled_resident_ctas=blocks,modeled_resident_warps=blocks*threads//32,
                shared_bytes_per_cta=65536,
                limitation="Static resource bound only; CUDA occupancy API and GPU timing remain necessary")


def compile_case(size,dtype,family,wm,wn,folded):
    m,n,k = (256,384,256) if size == "smoke" else (4096,4096,4096)
    combined.search.configure(gap=8,reuse=True)
    name = f"address_occupancy_{size}_{dtype}_{family}"
    source = make_source(m,n,k,dtype,wm,wn,folded)
    path = HERE/f"{name}.ptx"; path.write_text(source)
    output = BASE/"build"/name
    captures = []
    original = build.generate
    def capture(kernel,arch,opt_level):
        result = original(kernel,arch,opt_level); captures.append(result); return result
    combined.search.backend.STAGE_RECORDS.clear()
    build.generate = capture
    started = time.perf_counter()
    try:
        with context(folded):
            rc = main(["compile",str(path),"--out-dir",str(output),"--opt-level","safe"])
    finally:
        build.generate = original
    if rc or len(captures) != 1:
        raise RuntimeError(f"{name}: compile failed rc={rc}")
    manifest = json.loads((output/"manifest.json").read_text())
    records = oracle.parse_listing((output/manifest["entries"][0]["sass"]).read_text())
    code,labels,words,exits,regs = captures[0]
    assert [row.word for row in records[:len(words)]] == words
    norm = lambda text: re.sub(r"\s+","",text.replace(".reuse","").replace("+-","-"))
    for (want,ctl),actual in zip(code,records):
        assert ctl == control(actual.word) and control_valid(control(actual.word))
        if want.startswith(("HMMA","LDSM","LDGSTS","LDGDEPBAR","DEPBAR","BAR","STG")):
            assert norm(want) == norm(actual.text),(want,actual.text)
    threads = 128//wm*(128//wn)*32
    model = resource_model(regs,threads)
    files = [Path(__file__),Path(address.__file__),Path(combined.__file__),
             Path(combined.search.__file__),Path(combined.epilogue.__file__),
             BASE/"backend.py",BASE/"pipeline.py",BASE.parents[1]/"openassembler/backend/ra.py"]
    metadata = dict(compiler="OpenAssembler bounded warp-layout/address-folding experiment",
                    fallback=False, reference_and_ELF_container="NVIDIA ptxas on matching PTX",
                    candidate=dict(shared_chunk_immediates=folded,paired_b=True,row_epilogue=True,
                                   gap=8,reuse=True),
                    geometry=dict(m=m,n=n,k=k,bm=128,bn=128,bk=64,wm=wm,wn=wn,stages=2),
                    resource_model=model,
                    source_ptx_sha256=hashlib.sha256(source.encode()).hexdigest(),
                    compiler_source_hashes={str(p.relative_to(BASE.parents[1])):
                                            hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                    cli_wall_ms=(time.perf_counter()-started)*1000,
                    kernels=combined.search.backend.STAGE_RECORDS.copy(),
                    sass_opcode_counts=dict(Counter(row.text.split()[0] for row in records)),
                    exact_nvdisasm_tensor_memory_store_operands=True,
                    spliced_words_and_controls_exact=True)
    (output/"experiment.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(f"RESOURCE {name}: {json.dumps(model)}",flush=True)
    return dict(build=name,name=name,m=m,n=n,k=k,dtype=dtype,tile_m=128,tile_n=128,
                threads=threads,smem=65536),dict(build=name,**model)


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size",choices=("smoke","large","both"),default="both")
    args = parser.parse_args()
    combined.search.backend.install()
    resources = []
    sizes = ("large","smoke") if args.size == "both" else (args.size,)
    for size in sizes:
        suite = []
        for dtype in ("f16","bf16"):
            if size == "large":
                root = dict(build=f"search_combined_square_row_bpair_{dtype}",
                            name=f"address_occupancy_large_{dtype}",m=4096,n=4096,k=4096,
                            dtype=dtype,tile_m=128,tile_n=128,threads=128,smem=65536)
            else:
                root,resource = compile_case(size,dtype,"w64x64_baseline",64,64,False)
                resources.append(resource)
                root["name"] = f"address_occupancy_smoke_{dtype}"
            root["steady_warmup_rounds"] = 16
            root["alternatives"] = []
            for family,wm,wn,folded in FAMILIES:
                candidate,resource = compile_case(size,dtype,family,wm,wn,folded)
                resources.append(resource);root["alternatives"].append(candidate)
            suite.append(root)
        target = HERE/f"address_occupancy_{size}_suite.json"
        target.write_text(json.dumps(suite,indent=2)+"\n")
        (HERE/"address_occupancy_resources.json").write_text(json.dumps(resources,indent=2)+"\n")
        print(target,flush=True)


if __name__ == "__main__":
    cli()
