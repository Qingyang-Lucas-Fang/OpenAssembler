# SASS Reverse Engineering — Workflow

Goal: recover NVIDIA SASS (Volta+ 128-bit ISA) from scratch: decode instruction
words, rebuild control flow, track registers and semantics, lift to an IR, and
map that IR back to PTX and CUDA C. The PTX bridge (Stage 5) is the success
criterion; every earlier stage exists to make it possible.

Primary target: **sm_80/sm_86 (Ampere)**. Extend to sm_89, sm_90 (Hopper),
sm_100/sm_120 (Blackwell) once the pipeline is proven on Ampere.

## Language split

- **Subject**: SASS itself. We work at the bit level: raw 128-bit words,
  hand-written SASS probes, bit-flip fuzzing against `nvdisasm`.
- **Tooling**: Python (`openassembler/`). A decoder's speed depends on its tables and
  lookup structure, not on being written in assembly; Python lets us iterate
  on the encoding knowledge quickly. Hot paths can move to Rust/C++ later
  without changing the specs.
- **Specs** (`specs/<arch>/*.json`): the real product. Language-neutral
  encoding tables any implementation can consume.

## Ground truth (oracle)

NVIDIA's own tools (CUDA 12.8 linux-sbsa redist, fetched by `env/fetch_cuda.sh`
and sha256-verified against NVIDIA's own manifest into the gitignored
`env/cuda-12.8/`), run inside the colima Linux VM since this Mac has no CUDA:
`ptxas`, `cuobjdump`, `nvdisasm`, `nvcc`, via `env/cuda.sh <cmd>`.
The VM has no host C++ compiler yet, so probes are written in PTX for `ptxas`.
They compile and disassemble without a GPU. Executing kernels (semantic
validation on hardware) needs a cloud GPU; that's deferred to Stage 4/5.

NVIDIA tools are used only as a black-box oracle to generate and check
data. We don't decompile or redistribute their binaries.

## Pipeline

| # | Stage | Input → Output | Validation |
|---|-------|----------------|------------|
| 0 | Extract | host ELF / fatbin / cubin → per-kernel `.text` + metadata (`.nv.info`, cbank0, relocs) | byte-identical to `cuobjdump -xelf`; EIATTR fields match `cuobjdump -elf` |
| 1 | Disassemble | 128-bit words → instruction objects + control bits (stall, yield, SBW, SBR, wait mask, reuse) | text matches `nvdisasm` on 100% of corpus; round-trip re-encode is bit-exact |
| 2 | CFG | instructions → basic blocks, edges, functions (BRA/BRX/CALL/RET/EXIT, BSSY/BSYNC, WARPSYNC, predicated branches) | edge set matches `nvdisasm -cfg` on corpus |
| 3 | Registers & semantics | CFG → def-use, liveness, SSA over R/UR/P/UP/SR; per-opcode semantics (LOP3 LUT, IADD3 carries, ISETP chains, SHF, PRMT, IMAD.WIDE...) | concrete interpreter vs. hardware results (cloud GPU); reg count vs. EIATTR_REGCOUNT |
| 4 | Lift | SSA → structured IR (loops, if/else from predication, memory spaces, param loads from `c[0x0][...]`) | IR interpreter equals SASS interpreter on random inputs |
| 5 | **PTX/CUDA bridge** | IR → PTX (re-compilable) + readable CUDA C; idiom table PTX ↔ SASS | `ptxas` our PTX → SASS, equivalence test vs. original (modulo regalloc/scheduling); score SASS→PTX line mapping against `-lineinfo` cubins (`.nv_debug_line_sass` + embedded `.nv_debug_ptx_txt`) |

Stage notes from research (docs/research/03): BSSY/BSYNC pairs give the divergence
region tree for structuring; predication is the norm, so the IR uses guarded defs
(ψ-SSA); internal helpers (`__internal_*` div/rcp/sqrt slow paths, `BRA.CONV` shfl
fallbacks) are recognized and collapsed to one PTX op; cbank0 param base is per-arch
data (0x160 sm_80, 0x210 sm_90, 0x380 sm_100).

## Method for Stage 1 (encoding discovery)

1. **Corpus**: compile probe kernels (`probes/`) plus real code (CUTLASS, cuBLAS-style
   kernels) for each arch; dump `cubin` + `nvdisasm -hex` → `(word, text)` pairs.
2. **Field discovery**: for each opcode, take a seed word, flip each of the 128 bits,
   disassemble, and diff the text. Bits that change the same operand form a field;
   bits that make the word invalid mark constraint bits.
3. **Value tables**: enumerate each field to get modifier names, register
   numbering, and immediate/cbank formats.
4. **Spec emit**: `specs/sm_80/opcodes.json` (opcode → fields, operand kinds, modifiers).
5. **Decoder/encoder** generated from the spec; validate against the whole corpus.

## Layout

```
WORKFLOW.md          this file
docs/research/       research notes (prior art, control bits, containers, lifting)
env/                 container wrapper for NVIDIA tools
probes/              CUDA/PTX probe kernels targeting specific instructions
corpus/<arch>/       generated cubins + reference disassembly (generated, not edited)
specs/<arch>/        discovered encoding tables (JSON)
openassembler/              Python package
  container/         ELF / fatbin / .nv.info parsing           (Stage 0)
  isa/               decoder, encoder, control bits            (Stage 1)
  cfg/               basic blocks, CFG, call graph             (Stage 2)
  analysis/          def-use, liveness, SSA, semantics, interp (Stage 3)
  lift/              structured IR                             (Stage 4)
  bridge/            PTX + CUDA C emission, idiom tables       (Stage 5)
tools/               fuzzers, corpus builders, diff harnesses
tests/               pytest suites, one per stage
```

## Rules

- A stage is done only when its validation column passes on the full corpus.
- Specs are generated by tools, never hand-edited; hand corrections live in
  `specs/<arch>/overrides.json` with a comment explaining the evidence.
- Every claim about an encoding cites either a corpus example or a fuzz run.
