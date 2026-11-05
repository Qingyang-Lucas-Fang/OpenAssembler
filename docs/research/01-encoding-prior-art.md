# 01 — SASS encoding: prior art, methodology, and the Volta+ 128-bit layout

Scope: how others discovered NVIDIA SASS encodings, what the Volta→Blackwell (sm_70…sm_120) instruction word looks like, how to automate discovery, legal positioning, and a concrete plan. Primary target: Ampere sm_80/sm_86. Control/scheduling bits and cubin/fatbin containers are covered elsewhere and only touched on here.

## 0. Evidence quality (read first)

- This note began as a prior-art survey. Its links are pointers to primary sources; claims about projects should be rechecked against those sources before reuse.
- The exact bit ranges in §2 come from my working knowledge of public assembler and compiler sources (Mesa NAK `sm70` encoder, CuAssembler, turingas). I could not re-open those files here, so every bit range is tagged:
  - [V]: decodes consistently on the reference words in §2.7. I decoded those words with a script, but the words themselves are recalled from typical sm_80 `nvdisasm -hex` dumps, not produced here. Regenerate them as step 1 of §5.
  - [R]: recalled from public source code, needs to be confirmed against the code or by bit-flipping (§3) before we rely on it.
- Licenses marked "(verify)" weren't visible in the snippets I got.
- **Update 2026-10-24:** a live CUDA 12.8 oracle was available through `env/cuda.sh`. Hand checks on real sm_80 words confirmed opcode bits 0-8 plus form 9-11, guard 12-15, Rd 16-23, cbank offset 38-53, and control fields in 105-121. Systematic confirmation is in the current compiler and evidence ledgers.
- Some project names in §1 (especially 2025-2026 entries) come only from search snippets. Treat them as leads to check, not facts.

## 1. Prior art

### 1.1 Pre-Volta lineage (64-bit words plus separate control words)

| Project | Arch | License / status | Key technique |
|---|---|---|---|
| decuda / cudasm (Wladimir van der Laan) | G80 (sm_1x) | Open source, dormant | Hand-built disassembler and assembler for the earliest CUDA ISA. Predates nvdisasm. [repo](https://github.com/laanwj/decuda) |
| asfermi (Hou Yunqing) | Fermi (sm_20) | Open source, imported from Google Code, last commit 2015 | Hand-reverse-engineered Fermi encodings, rule tables per instruction. [repo](https://github.com/hyqneuron/asfermi) |
| envytools (envydis/envyas) | G80…Maxwell (GM107) | Open source (nouveau community) | Fully independent decoder tables written without nvdisasm. Reads hex bytes on stdin. No Volta+ coverage. [docs](https://envytools.readthedocs.io/en/latest/envydis/index.html), [repo](https://github.com/envytools/envytools/) |
| maxas (Scott Gray) | Maxwell / Pascal (sm_5x/6x) | Open source (Nervana, later Intel). Intel has stopped development. | Used `cuobjdump` output to learn encodings, then reverse-engineered the control codes (stall/yield/barriers/reuse) and showed their impact through a hand-scheduled SGEMM. [repo](https://github.com/NervanaSystems/maxas), [control codes wiki](https://github.com/NervanaSystems/maxas/wiki/Control-Codes), [intro](https://github.com/NervanaSystems/maxas/wiki/Introduction) |
| KeplerAs (Xiuxia Zhang et al.) | Kepler (sm_3x) | Open source, PPoPP'17 artifact | Built on maxas. Added an "instruction solver" that generates instruction variants and solves for bit fields automatically. This is the ancestor of the fuzzing approach. [repo](https://github.com/xiuxiazhang/KeplerAs), [artifact](https://github.com/PAA-NCIC/PPoPP2017_artifact), [paper](https://dl.acm.org/doi/10.1145/3018743.3018755) |
| DeepPerf Solver (PAA-NCIC) | sm_30…sm_62 | Open source | Generalized solver from the same group. Supported archs are listed as sm_30–sm_62. [Solver README](https://github.com/PAA-NCIC/DeepPerf/blob/master/Solver/README.md), [DeepPerf](https://github.com/PAA-NCIC/DeepPerf) |
| Decoding CUDA Binary (Hayes, Hua, Zhang et al., CGO 2019) | CC 3.2–6.2 per tool README | Open source tools (decode, asm2bin, bin2asm) | Automatically discovers encodings by mutating instruction bits and observing how nvdisasm's text changes. Enables cross-arch binary analysis and rewriting. [repo README](https://github.com/decodecudabinary/Decoding-CUDA-Binary/blob/master/tools/README.txt), [project page](https://people.cs.rutgers.edu/zz124/projects/9_project/), [ResearchGate](https://www.researchgate.net/publication/331606734_Decoding_CUDA_Binary), [Hayes thesis 2022](https://rucore.libraries.rutgers.edu/rutgers-lib/67370/PDF/1/play) |

### 1.2 Volta through Ampere

| Project | Arch | License / status | Key technique |
|---|---|---|---|
| Jia et al., "Dissecting Volta" (Citadel, 2018) | sm_70 | arXiv tech report | Microbenchmarks plus instruction-set disassembly. Documented the 128-bit Volta word with embedded control information and mapped the encoding space. [arXiv 1804.06826](https://arxiv.org/abs/1804.06826) |
| Jia et al., "Dissecting Turing T4" (2019) | sm_75 | arXiv | "We map Turing's instruction space, finding the same encoding as Volta, and additional instructions." Also covers the uniform datapath. [arXiv 1903.07486](https://arxiv.org/abs/1903.07486) |
| turingas (Da Yan, HKUST) | sm_70, sm_75, sm_80 | Open source (verify license), dormant | Python assembler with hand-written per-instruction encoding tables, used for Winograd convolution at up to 93% of peak (PPoPP'20). [repo](https://github.com/daadaada/turingas), [paper](https://cse.hkust.edu.hk/~weiwa/papers/yan-ppopp20.pdf), [forum post](https://forums.developer.nvidia.com/t/an-open-source-sass-assembler-for-volta-and-turing-gpus/109011) |
| GAS / GASS (same group) | Volta, Turing, Ampere | Research | LLVM-based open compiler that emits SASS directly. [PPoPP'22 poster](https://cse.hkust.edu.hk/~weiwa/papers/gass-ppopp22-poster.pdf), [GAS paper](https://www.researchgate.net/publication/349469700_Simplifying_low-level_GPU_programming_with_GAS) |
| CuAssembler (cloudcores) | sm_60/61/70/75/80/86 ("…") | Open source (verify, believed MIT), forks exist | Learns encodings from (nvdisasm text, binary code) pairs harvested from real cubins. Per-arch "InsAsmRepos" are solved with linear algebra (§3.4). Ships cubin I/O and a control-code module. [repo](https://github.com/cloudcores/CuAssembler), [README](https://github.com/cloudcores/CuAssembler/blob/master/README.md), [UserGuide](https://github.com/cloudcores/CuAssembler/blob/master/UserGuide.md), [Tutorial](https://github.com/cloudcores/CuAssembler/blob/master/Tutorial.md), [CuControlCode.py](https://github.com/cloudcores/CuAssembler/blob/master/CuAsm/CuControlCode.py), forks: [gpuocelot/cuasm](https://github.com/gpuocelot/cuasm), [RCAsm](https://github.com/RetiredC/RCAsm/blob/main/cuAssembler/UserGuide.md) |
| NBAssembler (QianyanTech) | Maxwell…Ampere | Open source (verify) | Assembler plus decompiler. [repo](https://github.com/QianyanTech/NBAssembler) |
| Yan / Abdelkhalik et al., "Demystifying Ampere" (HPEC 2022) | sm_80 | arXiv | Latency of PTX vs. SASS instructions and memory units. No encoding tables, but useful for semantics and latency. [arXiv 2208.11174](https://arxiv.org/html/2208.11174v1) |
| Huerta et al., "Analyzing Modern NVIDIA GPU cores" (2025) | Ampere focus | arXiv | Hand-written SASS sequences with explicit control bits, used to reverse-engineer the issue, register-file, and dependency hardware for Accel-Sim. [arXiv 2503.20481](https://arxiv.org/html/2503.20481v1), [thesis/UPC](https://upcommons.upc.edu/server/api/core/bitstreams/3d75bf4b-84c5-4901-a435-413ba8036f1b/content) |
| Mesa NVK / NAK (Faith Ekstrand et al.) | Encoder for Volta+ (sm_70…), later Maxwell, Kepler, and consumer Blackwell | MIT (Mesa), very active | Production Rust compiler that emits SASS. Its `sm70` encoder is the cleanest public statement of field bit ranges. It's validated by running on hardware and has unit tests against a CPU model. [Phoronix: NAK merged](https://www.phoronix.com/news/NAK-Merged-Mesa-24.0), [unit-testing blog](https://www.collabora.com/news-and-blog/blog/2024/08/15/a-shifty-tail-about-unit-testing/), [NVK docs](https://docs.mesa3d.org/drivers/nvk.html), [Mesa 25.2 Blackwell](https://www.collabora.com/news-and-blog/news-and-events/mesa-25.2-brings-new-hardware-support-for-nouveau-users.html), [external HW docs](https://docs.mesa3d.org/drivers/nvk/external_hardware_docs.html) |
| Pound emulator | sm_86 | Open source | Code generation pipeline for the SM86 SASS instruction set. [DeepWiki](https://deepwiki.com/pound-emu/pound/4.2-configuration-and-setup) |

### 1.3 Hopper, Blackwell, and 2024–2026 work

| Project | Arch | License / status | Key technique |
|---|---|---|---|
| nv_isa_solver (Kuter Dinel) | sm_89 (Ada), sm_90a (Hopper) | Open source (verify) | Fuzzing-based spec generator. The README credits the fuzzing algorithm to Zhang et al. (PPoPP'17) and the parser to CuAssembler. Publishes human-readable ISA specs for SM90a and SM89, which Mesa lists as a reverse-engineered source. [repo](https://github.com/kuterd/nv_isa_solver), [Mesa external docs](https://docs.mesa3d.org/drivers/nvk/external_hardware_docs.html) |
| DocumentSASS (0xD0GF00D) | Multiple | Open source (verify) | "Unofficial description of the CUDA assembly (SASS) instruction sets". Data comes from NVIDIA tooling rather than fuzzing. OTHER.md surveys MaxAs, AsFermi, CuAssembler, TuringAs, KeplerAs, Decuda, and the Citadel papers. [repo](https://github.com/0xD0GF00D/DocumentSASS), [OTHER.md](https://github.com/0xD0GF00D/DocumentSASS/blob/main/OTHER.md) |
| EoSS (Avalos, Payer, Rogers; ASPLOS 2027) | Eight archs, SM70–SM121 | Released artifacts | "Extracts machine-readable instruction specifications from the Rosetta String embedded in NVIDIA's tooling". Generates valid encodings, runs them on real hardware via binary patching, and reports 443–1,239 validated instruction classes per generation. [paper page](https://engineering.purdue.edu/tgrogers/publication/avalos-asplos-2027/), [GitHub org](https://github.com/eoss-framework), [repo](https://github.com/eoss-framework/eoss-framework) |
| F2Asm, "Learning Exact NVIDIA SASS Encoders with F₂ Linear Algebra" (2026) | SM90/90a, SM100, SM107 | arXiv | Learns exact 128-bit encoders from paired (disassembly, original cubin word) data. Uses Gaussian elimination over GF(2) to build a basis incrementally, detect inconsistencies, and reject inputs outside the learned span. Control bits, relocations, and metadata are kept separate from the learner. Trained on about 3,225 cubins. It's CuAssembler's idea made exact. [arXiv 2608.20532](https://arxiv.org/abs/2608.20532), [html v2](https://arxiv.org/html/2608.20532v2) |
| Automated Instruction Encoding Synthesis for GPU ISA Compression (2026) | Blackwell, plus Ampere and Hopper | arXiv | Normalizes "raw public encoding text" into a machine-readable spec. A SASS disassembler is validated against nvdisasm on 3.78M instructions, then re-encodes 33% denser. Shows that a full spec-driven disassembler reaching nvdisasm parity exists. [arXiv 2609.18662](https://arxiv.org/abs/2609.18662), [html](https://arxiv.org/html/2609.18662v1) |
| Dale, "Characterizing Warp Divergence from Pascal to Blackwell" (2026) | Pascal…Blackwell | arXiv | Uses controlled bit-flip experiments to find a new 2-bit barrier-class field on Blackwell (e.g. RECONVERGENT) that doesn't exist on Ampere or Hopper. This is the bit-flip method applied to semantics. [arXiv 2607.23402](https://arxiv.org/abs/2607.23402) |
| basalt (Sunny Patel) | sm_120 | Open source (verify) | sm_120 assembler and scheduler matched byte for byte against ptxas, plus a hazard checker. It notes that 21 of the 128 bits are the scheduling control word and that sm_120 has no interlock on fixed-latency instructions. [repo](https://github.com/sunnypatell/basalt) |
| blackwell-isa (kacper-daftcode) | SM120, SM103a, SM121a | Open source (verify) | Machine-readable reverse-engineered ISA databases for Blackwell. [repo](https://github.com/kacper-daftcode/blackwell-isa) |
| blackwell-isa-probe (jethac) | sm_100a/103a/110a vs. sm_120a/120f/121a | Open source | Splits instructions into datacenter and consumer camps: 68 consumer-only, 7 datacenter-only, 20 lowered differently. [CAMP_SPLIT.md](https://github.com/jethac/blackwell-isa-probe/blob/main/results/CAMP_SPLIT.md) |
| SASS King (Florian Mattana) | SM120 first | Open source | Instruction dictionary, kernel audits, and pattern recognition. Not an encoder. [repo](https://github.com/florianmattana/sass-king) |
| JEB 5.31 (PNF Software) | sm_70…sm_121 | Commercial | Generic SASS disassembler plus experimental decompiler. [blog](https://www.pnfsoftware.com/blog/reversing-nvidia-cuda-sass-code/) |
| Hopper / Blackwell microbenchmark papers | sm_90, sm_100/120 | arXiv | Latency and throughput only, no encodings. Luo et al. IPDPS'24 [2402.13499](http://arxiv.org/abs/2402.13499v1). Hopper multi-level [2501.12084](https://arxiv.org/abs/2501.12084v2). Blackwell [2507.10789](https://arxiv.org/abs/2507.10789). Jarmusch & Chandrasekaran B200 [2512.02189](https://arxiv.org/abs/2512.02189). |
| ROCetta (Sreepathi Pai, Rochester) | PTX/SASS | Research | Formal specification of GPU ISAs, plus mutation and equivalence-checking test generation. [project](https://www.cs.rochester.edu/~spai4/rocetta.html), [Fuzzing'22 paper](https://www.cs.rochester.edu/u/spai4/papers/shitrit-fuzzing2022.pdf) |

### 1.4 What the lineage teaches

1. Three ways to get encodings, in increasing order of automation and legal exposure:
   - (a) Hand reverse-engineering from compiler output: asfermi, maxas, turingas.
   - (b) Black-box learning against nvdisasm, either mutation/bit-flip (KeplerAs solver, Hayes, nv_isa_solver) or linear solves over (text, code) pairs (CuAssembler, F2Asm).
   - (c) Extracting NVIDIA's own embedded spec from tool binaries (DocumentSASS, EoSS's "Rosetta String"). This is the fastest route, but it means disassembling NVIDIA binaries (§4).
2. Nobody stops at nvdisasm agreement. maxas, NAK, Huerta, EoSS, and Dale all run code on hardware, because nvdisasm accepts encodings the hardware rejects or treats differently.
3. The Volta word has been stable through sm_120. Jia showed Turing shares Volta's encoding. CuAssembler covers sm_70–86 with one engine. F2Asm and basalt use the same 128-bit, 21-control-bit framing on sm_90/100/120. What changes between generations is the opcode set and modifier tables (TMA/WGMMA on sm_90, tcgen05/TMEM on sm_100 but not sm_120, and the consumer/datacenter split).

## 2. The Volta+ 128-bit instruction word (sm_70 … sm_120)

Notation: bit 0 is the least significant bit of the first little-endian 64-bit word. Bits 64–127 are the second 64-bit word. nvdisasm `-hex` prints these as `/* 0x<lo64> */` then `/* 0x<hi64> */`.

### 2.1 Overall split

| Bits | Content | Tag |
|---|---|---|
| 0–11 | Opcode, with the low 9 bits as the operation and bits 9–11 as the operand-form selector for ALU-style ops (§2.2) | [V] |
| 12–15 | Guard predicate: 3-bit index plus 1 negate bit | [V] |
| 16–23 | Rd (destination GPR, or URd for uniform-datapath ops) | [V] |
| 24–31 | Ra (src0) | [V] |
| 32–63 | Src1 slot: Rb in bits 32–39 (register form), a 32-bit immediate, a constant-bank reference, or a URb | [V] |
| 64–71 | Rc (src2). Also where the displaced register goes when src2 is cbuf/imm/ureg | [V] |
| 72–104 | Per-opcode modifiers, extra predicates, secondary destinations, sizes, rounding, and so on | [R] |
| 105–125 | Scheduling control (21 bits: stall 4, yield 1, write barrier 3, read barrier 3, wait mask 6, reuse 4). Owned by another doc. | [V] for framing; basalt also says "21 of them" |
| 126–127 | Unused / zero in observed code | [R] |

Disagreement: basalt states 21 control bits ([repo](https://github.com/sunnypatell/basalt)). Some older write-ups describe the Volta word as roughly 91 operation bits plus ~23 control bits plus unused bits. Jia et al.'s exact wording should be checked in the [PDF](http://arxiv.org/pdf/1804.06826.pdf), which I couldn't fetch. The encoder doesn't depend on which description is right as long as bits 105–125 are treated as control and bits 126–127 are kept zero.

### 2.2 Opcode and operand form

- The 12-bit field at 0–11 is best treated as `op9 | form<<9` for ALU-class instructions. Form values, checked against known encodings (§2.7):

  | form | src1 (bits 32–63) | src2 | Example opcode word |
  |---|---|---|---|
  | 1 | Rb | Rc | `0x7202` MOV R,R. `0x7223` FFMA R,R,R,R |
  | 2 | Rb | imm32 in 32–63, Rb moves to 64–71 | `0x7424` IMAD.MOV.U32 R,RZ,RZ,imm |
  | 3 | Rb | cbuf in 32–63, Rb moves to 64–71 | `0x7624` IMAD.MOV.U32 R,RZ,RZ,c[][] |
  | 4 | imm32 | Rc | `0x7802` MOV R,imm |
  | 5 | cbuf | Rc | `0x7a02` MOV R,c[][]. `0x7ab9` ULDC |
  | 6 | URb | Rc | `0x7c02` MOV R,UR |
  | 7 | Rb | URc | (by pattern) [R] |

- Non-ALU instructions (memory, branch, barrier, MMA) use the full 12 bits as an opcode without this form structure. Some opcodes also take sub-opcode bits in the 72–104 range. The opcode is not purely bits 0–11 for every instruction.
- Disagreement: NAK-style encoders name the 12-bit field and pass `form` separately. CuAssembler doesn't name fields at all. It learns a per-"instruction key" linear map (§3.4), so its repos can't be used to "look up the opcode width". Nobody publicly asserts a single global opcode width. Measure it by bit-flipping (§3).

### 2.3 Register operands

- GPRs: 8-bit fields, R0–R254, RZ = 255 [V]. 64-bit and 128-bit operands use the base register of an aligned pair or quad (`R2` means R2:R3). Alignment is enforced by ptxas, and nvdisasm still prints misaligned bases. Hardware behavior on misalignment hasn't been checked [R].
- Rd 16–23, Ra 24–31, Rb 32–39, Rc 64–71 [V]. Second destinations (IMAD.WIDE high half, MUFU etc.) and extra sources live in 72–104 per opcode [R].
- Operand-reuse flags are part of the control field, not the operand fields.

### 2.4 Predicates

- Guard: bits 12–14 hold the index P0–P6 with 7 meaning PT, and bit 15 is negate. `0x7…` in the opcode nibble means `@PT` (unguarded). `0xF…` means `@!PT` (never executes) [V]. Examples: `EXIT` = `…794d`, `@P0 BRA` has `0x0` in bits 12–15.
- Predicate destinations and sources for compare ops (ISETP/FSETP/PLOP3…) are 3-bit fields in 72–104. Recalled: Pd 81–83, second Pd 84–86, combining Pp 87–89 with negate at 90 [R].
- Uniform predicates UP0–UP6, UPT = 7. They use the same 3-bit index encoding in the predicate fields of uniform ops (UISETP, UPLOP3…) [R]. Whether a uniform instruction's guard at bits 12–15 names a P or a UP should be confirmed by flipping bits on a `UIADD3` [R].

### 2.5 Uniform registers (sm_75+)

- UR0–UR62, URZ = 63, encoded as 6-bit fields in the same slots: URd 16–21, URa 24–29, URb 32–37 [R]. The Turing uniform datapath was first reported by Jia et al. ([1903.07486](https://arxiv.org/abs/1903.07486)) and discussed on [NVIDIA forums](https://forums.developer.nvidia.com/t/whats-uniform-register-in-turing/65406).
- Checked by hand: `ULDC.64 UR4, c[0x0][0x118]` = `0x0000460000047ab9`. Bits 16–23 = 4 gives UR4 [V].
- Whether sm_90/sm_100 widen the UR file isn't covered by any source I found. Measure it [R].

### 2.6 Immediates and constant bank `c[bank][offset]`

- imm32: bits 32–63, raw bits. FP32 immediates are the IEEE bit pattern, e.g. `MOV R0, 0x3f800000` puts `0x3f800000` in the upper half of the low word [V]. Some ops use 20/24-bit or sign-extended immediates in other positions, for example branch offsets and memory address offsets on LDG/STG [R].
- cbuf (form 3/5): offset field at bits 38–53 as a byte offset. Equivalently, bits 40–53 hold offset/4 for 4-byte-aligned accesses. Bank is bits 54–58 [V for offset, R for bank width]. Check: `0x00000a00` in bits 32–63 gives `0x0a<<2 = 0x28` → `c[0x0][0x28]`. `0x00004600` gives `0x118`.
- Bindless / UR-indexed constant access (`c[UR][...]`, `ULDC` with UR base, `LDC` with register index) uses other fields [R].
- Kernel params are at `c[0x0][0x160]` on Volta–Ampere ([forum](https://forums.developer.nvidia.com/t/how-to-understand-the-lea-assembly-behind-the-cuda-c/257679/2)). The offset changes on later archs, so treat it as ABI, not encoding.

### 2.7 Reference words (sm_80), decoded with the field map above

| Text | lo64 | hi64 | Decoded |
|---|---|---|---|
| `MOV R1, c[0x0][0x28]` | `0x00000a0000017a02` | `0x000fe40000000f00` | op9=0x002, form=5 (cbuf), pred=7 (PT), Rd=1, offset=0x28, bank=0. Bits 72–75 = 0xf (MOV lane mask). ctrl: stall=2, yield=1, wbar=rbar=7 (none). |
| `EXIT` | `0x000000000000794d` | `0x000fea0003800000` | op=0x94d, pred=PT. The "form" bits are meaningless here: a non-ALU opcode. |
| `ULDC.64 UR4, c[0x0][0x118]` | `0x0000460000047ab9` | `0x000fe40000000a00` | op9=0x0b9, form=5, URd=4, offset=0x118. Bits 73–75 hold the `.64` size. |

### 2.8 Per-generation deltas (encoding-relevant only)

- sm_70 → sm_75: same encoding, plus uniform datapath (UR/UP, U* opcodes) and new MMA shapes ([Jia, Turing](https://arxiv.org/abs/1903.07486)).
- sm_80/86/89: same framework. New ops include LDGSTS (async copy), new HMMA/IMMA shapes, LDSM, and REDUX. sm_86 and sm_89 add small deltas over sm_80. CuAssembler treats 80 and 86 as separate repos ([README](https://github.com/cloudcores/CuAssembler/blob/master/README.md)), so sm_80 ↔ sm_86 equivalence must be measured, not assumed.
- sm_90/90a: TMA (UTMALDG…), WGMMA (HGMMA…), clusters, and descriptor operands in UR. nv_isa_solver publishes SM90a and SM89 specs ([repo](https://github.com/kuterd/nv_isa_solver)).
- sm_100/103/110 (datacenter Blackwell): tcgen05 / TMEM / UTC* ops. sm_120/121 (consumer) has no TMEM and keeps HMMA-style MMA ([gist](https://gist.github.com/solatticus/aab6ec3a0436748b021cbbdd12e8c739), [camp split](https://github.com/jethac/blackwell-isa-probe/blob/main/results/CAMP_SPLIT.md)). Blackwell adds a 2-bit barrier-class field ([Dale](https://arxiv.org/abs/2607.23402)).
- Instruction counts: NVIDIA's per-arch mnemonic tables are in the CUDA Binary Utilities "Instruction Set Reference" ([12.8 docs](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-binary-utilities/index.html), with a history in [CUDA Handbook ch. 8.7](https://cudahandbook.com/book/ch8/instruction-sets) and a [gist](https://gist.github.com/herrmann/f721da109e0c5c7c34c847ff2cf3da1e)). They list on the order of a few hundred mnemonics per Volta+ arch. The encoding-relevant count is larger: EoSS reports 443–1,239 validated instruction classes per generation (opcode × operand-form variants) across SM70–SM121 ([EoSS](https://github.com/eoss-framework/eoss-framework)). Budget about 500–700 classes for sm_80/86 and over 1,000 for sm_90a/sm_100a. This is an estimate, not a measured figure.

## 3. Discovery methodology

### 3.1 Getting raw words in and text out

- Raw mode: `nvdisasm -b SM80 file.bin` (`--binary SMxy`) treats the input as "a raw instruction binary, that is, a sequence of binary instruction encodings as they occur in instruction memory" ([forum quoting docs](https://forums.developer.nvidia.com/t/what-should-i-feed-into-nvdisasm-binary-sm52-command/44701), [-b vs cubin](https://forums.developer.nvidia.com/t/cupti-cuda9-1-and-sass-source-map-example/59003)). Write 16 bytes per instruction, little-endian lo64 then hi64.
- Useful flags (check `nvdisasm --help` for your toolkit): `-hex` prints the encoding next to each line ([overlay gist](https://gist.github.com/sebftw/de3e368f95083b7afb210a91b03a7ccd), [-hex issue](https://forums.developer.nvidia.com/t/nvdisasm-hex-in-cuda-9-2-does-not-work/64651)). `--print-raw` turns off pseudo-op beautification such as `IMAD.MOV` vs. `MOV`, which matters because aliasing hides fields. ([CUDA Binary Utilities 13.0](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-binary-utilities/index.html))
- Cubin patching is the alternative: compile a kernel, overwrite the `.text.<kernel>` bytes, then run nvdisasm on the cubin. It's slower, but needed when raw mode differs from cubin mode (relocations, symbolic branch targets) and for running on hardware.
- Batching: put N candidate words in one file (one per 16-byte slot) so one process launch disassembles thousands. Process startup dominates the cost otherwise. Put a known-good `NOP` between candidates so one bad word can't desynchronize the rest, and bisect any batch that fails as a whole.
- Host: nvdisasm hasn't shipped for macOS since CUDA 10.2 ([announcement](https://www.packtpub.com/en-br/learning/tech-news/nvidia-announces-cuda-10-2-will-be-the-last-release-to-support-macos), [NVIDIA mac tools page](https://developer.nvidia.com/nvidia-cuda-toolkit-developer-tools-mac-hosts)). Use Linux (x86_64 or sbsa/aarch64 in Docker on this arm64 Mac). nvdisasm is a standalone redistributable component ([packages.nvidia.com](https://packages.nvidia.com/components/cuda_nvdisasm/)). No GPU is needed for disassembly.

### 3.2 Bit-flip fuzzing (Hayes CGO'19, KeplerAs solver, nv_isa_solver, Dale'26)

For each seed instruction `w` with text `T(w)`:

1. Parse `T(w)` into a structured tuple: predicate, mnemonic, dot-modifiers, and a typed operand list (reg/ureg/pred/upred/imm/cbuf/address/label), each with its value. The parser is the hardest engineering part. CuAssembler's and nv_isa_solver's parsers exist to reuse ideas from.
2. For i in 0..104 (skip control bits 105–127), form `w ^ (1<<i)`, disassemble, and parse.
3. Classify the diff per bit:
   - no change: a don't-care or reserved bit for this opcode (keep it zero when encoding)
   - one operand's value changed by ±2^k: bit k of that operand's field. Consecutive bits on the same operand form a field, and the step sizes tell you scale (e.g. cbuf offsets step by 4 at bit 40)
   - a modifier appeared, disappeared, or changed: a modifier-field bit
   - mnemonic changed: an opcode bit (or a form bit if operand kinds changed but the mnemonic didn't)
   - the operand kind changed (R → imm → c[][]): a form bit
   - invalid or error output: an illegal combination, recorded as a constraint
4. Group bits into fields: adjacent bits whose flips affect the same semantic slot.
5. Enumerate values: for each modifier field of width ≤ 6–8, sweep all 2^n values, holding the rest at the seed, and record the text for each to build the value→string table (e.g. `.RN/.RM/.RP/.RZ`, ISETP `.LT/.EQ/…`, `.U32/.S32`, LOP3 LUT). Many fields are only meaningful when another field has a given value, so sweep pairs where single sweeps produce invalid output.
6. Repeat from several seeds per opcode with different operand kinds and modifiers. A bit that's a don't-care in one seed may be live in another, for example only when `.WIDE` is set.

Cost: about 105 flips per seed plus value sweeps. With batching, a few hundred seeds per opcode across about 600 classes is roughly 10⁶–10⁷ disassembled words. That's hours on one machine.

### 3.3 Handling invalid encodings

- nvdisasm signals rejects in several ways: an error exit, garbage output, or a mangled line for that slot. Normalize all of these to INVALID. The exact strings vary by toolkit version and should be recorded, not assumed.
- Keep INVALID results. They define the legal value set per field (e.g. predicate fields can't hold 7 in some ops, register pairs must be aligned, some modifier codes are reserved) and become the assembler's validation rules.
- nvdisasm-valid isn't the same as hardware-valid. EoSS executes encodings on real GPUs via binary patching ([EoSS](https://engineering.purdue.edu/tgrogers/publication/avalos-asplos-2027/)), and Dale found fields with no runtime effect ([2607.23402](https://arxiv.org/abs/2607.23402)). Expect illegal-instruction traps (`cudaErrorIllegalInstruction`) for some nvdisasm-accepted words.
- Non-determinism: none is expected from nvdisasm, but pin one toolkit version per dataset because text formatting changes between releases.

### 3.4 CuAssembler-style linear solving (InsAsmRepos) and its exact version, F2Asm

CuAssembler (see [README](https://github.com/cloudcores/CuAssembler/blob/master/README.md) and [UserGuide](https://github.com/cloudcores/CuAssembler/blob/master/UserGuide.md); details below are recalled from the source and need verifying):

1. Harvest (text, code) pairs from many cubins (CUDA samples, cuBLAS/cuDNN cubins, etc.) disassembled with nvdisasm.
2. Canonicalize each text into an instruction key: mnemonic plus operand-type signature, e.g. `FFMA_R_R_R_R`, `FFMA_R_R_II_R`. Then into a value vector: the numeric operand values (register indices, immediates, cbuf bank/offset) and one indicator entry per modifier present.
3. Assume the code is linear in that vector: `code = Σ v_i · W_i` (integer arithmetic over 128-bit values, with modifiers contributing fixed bit patterns). Solve for the weight vectors `W_i` from the collected samples. Each operand's weight is effectively 2^(field start), and each modifier's weight is its bit pattern.
4. Store the per-arch, per-key matrices as the InsAsmRepos (one pickled repo per SM version). At assembly time, look up the key, dot the value vector with the weights, OR in the control bits, and you have the word.
5. Self-check: re-encode every harvested instruction and require a bit-exact match. Unseen modifiers or operand kinds fail loudly, so you grow the repo by compiling more varied PTX.

Weaknesses: only modifiers and operand types seen in the corpus can be encoded. Integer linearity breaks for non-linear fields such as negative immediates and LUT-like modifiers, which CuAssembler special-cases. It gives no field names.

F2Asm makes this exact over GF(2): a "compact basis" learned by incremental Gaussian elimination, with inconsistency detection and rejection of inputs outside the learned span ([2608.20532](https://arxiv.org/abs/2608.20532)). It works for sm_90/100/107, so the method carries across generations.

Best practice is a hybrid: linear solving (cheap, uses real compiler output) to bootstrap keys and fields, then bit-flip fuzzing (§3.2) to fill the modifier values the compiler never emitted and to label fields.

### 3.5 Alternative: the embedded spec

EoSS extracts the "Rosetta String" from NVIDIA tooling ([EoSS org](https://github.com/eoss-framework)), and DocumentSASS documents per-arch instruction descriptions from tooling ([repo](https://github.com/0xD0GF00D/DocumentSASS)). This gives a near-complete spec quickly. Its legal profile is different (§4). If we use it, do it only as a cross-check in a separate, clearly marked step, or not at all.

## 4. Legal notes (not legal advice)

What the license says:
- NVIDIA's SDK EULA, which covers the CUDA Toolkit ([current EULA](https://docs.nvidia.com/cuda/eula/index.html), [12.4 PDF](https://docs.nvidia.com/cuda/archive/12.4.0/pdf/EULA.pdf)), contains the usual restriction against reverse engineering, decompiling, or disassembling SDK components provided in binary form. The wording "You may not reverse engineer, decompile or disassemble the Software components provided in binary form, nor attempt in any other manner to obtain source code" appears in NVIDIA SDK EULAs ([SDK Manager EULA](http://docs.nvidia.com/sdk-manager/eula/index.html)). Read the exact current clause text before deciding anything. I could only see snippets.
- Since about CUDA 11.6 there's an added clause: "You may not reverse engineer, decompile or disassemble any portion of the output generated using SDK elements for the purpose of translating such output artifacts to target a non-NVIDIA platform" ([TrendForce](https://www.trendforce.com/news/2024/03/11/news-nvidias-eula-amendments-tighten-grip-suppressing-third-party-cuda-emulation/), [Gigazine](https://gigazine.net/gsc_news/en/20240305-nvidia-cuda/)). It's aimed at translation layers such as ZLUDA. A SASS assembler/disassembler that targets NVIDIA GPUs isn't for that purpose. Our lifting/PTX work should not be framed or used as an NVIDIA→other-vendor translator without a legal review.

Statutory backdrop:
- EU: Directive 2009/24/EC Art. 6 permits decompilation needed for interoperability of an independently created program, and Art. 8 voids contract terms contrary to Art. 6 ([EUR-Lex](http://data.europa.eu/eli/dir/2009/24/oj/eng/xhtml), [summary](https://eur-lex.europa.eu/legal-content/EN/LSU/?uri=celex:32009L0024), [Wikipedia](https://en.wikipedia.org/wiki/Computer_Programs_Directive)). Results can't be used for other purposes.
- US: 17 U.S.C. §1201(f) is an interoperability exemption to the anti-circumvention rules ([Cornell LII](https://www.law.cornell.edu/uscode/text/17/1201)), but contracts can still bind. Bowers v. Baystate held a shrink-wrap reverse-engineering ban enforceable despite fair use ([Wikipedia](https://en.wikipedia.org/wiki/Bowers_v._Baystate_Technologies,_Inc.), [EFF amicus](https://www.eff.org/files/filenode/Bowers_v_Baystate/20020918_baystate-amicus.pdf)).

Risk ranking of techniques, lowest risk first:
1. Analyzing our own compiled output: compile our own CUDA/PTX, then read our own cubins and the hex that nvdisasm prints for them. That's the data CuAssembler-style solving uses, and it doesn't touch NVIDIA's binaries.
2. Black-box use of nvdisasm as a documented tool: feeding raw words (`-b`) and reading its output is ordinary use of a documented interface. Bit-flip fuzzing is this at scale. Some argue mass querying is "reverse engineering the tool". That hasn't been tested in court as far as I found.
3. Harvesting from NVIDIA-shipped library cubins (cuBLAS/cuDNN): the inputs are NVIDIA binaries. Use them for statistics only, never redistribute them, and prefer our own corpus.
4. Extracting tables from inside nvdisasm/ptxas (the "Rosetta String", DocumentSASS-style): this is literally disassembling or inspecting an SDK binary, which is the clause's core target. Highest risk under the US contract theory. It may be defensible in the EU as interoperability. Avoid it, or isolate it.

How existing projects position themselves (observed, not legal opinions): maxas, turingas, and CuAssembler call themselves "unofficial" assemblers built to write faster kernels for NVIDIA hardware, which is interoperability on NVIDIA's own platform ([maxas intro](https://github.com/NervanaSystems/maxas/wiki/Introduction), [CuAssembler](https://github.com/cloudcores/CuAssembler)). Mesa NVK/NAK is an independent open driver in the nouveau tradition, and Mesa lists its reverse-engineered sources openly ([Mesa external docs](https://docs.mesa3d.org/drivers/nvk/external_hardware_docs.html)). Academic tools (EoSS, F2Asm, Hayes) publish as research. I found no public record of NVIDIA taking legal action against any of these projects. Absence of enforcement isn't permission.

Project policy recommendation: do everything we can to reverse engineer Nvidia SASS.

## 5. Recommended plan

Start with sm_80, then do sm_86 as a diff. sm_80 is the best-covered Volta+ target (CuAssembler, turingas, NAK, Huerta), A100 hardware is easy to rent, and sm_86 hardware (RTX 30xx) is common. Treat sm_86 as "sm_80 + measured delta": run the identical corpus for both and diff the learned specs. Then sm_75/sm_70 (mostly subsets), then sm_89, then sm_90a, sm_100a, and sm_120 (new opcode families, same word).

Phase 0, tooling (about 1 day)
- Set up a Linux container (arm64 sbsa or x86_64) with a pinned CUDA 12.x `ptxas` + `nvdisasm` + `cuobjdump`. Record the version in every dataset.
- Write a raw-word harness: encode a list of 128-bit words to a `.bin`, run `nvdisasm -b SM80 -hex --print-raw`, and parse the output back per slot. Pad with NOPs and bisect failures (§3.1).
- First test: regenerate the §2.7 reference words and confirm or correct the [V] fields.

Phase 1, corpus and linear bootstrap (CuAssembler/F2Asm style)
- Generate PTX programmatically: every PTX instruction × type × modifier × operand-kind combination we can express (register, immediate, constant param, uniform-friendly patterns), at -O0 and -O3, with varied register pressure. Compile with ptxas for sm_80/86. This corpus is ours (§4, method 1).
- Disassemble, then canonicalize to (key, value vector, word) using a shared text parser.
- Per key: run a GF(2) solve with Gaussian elimination. Record the basis, the residual (the constant bits), and consistency. Every pair must re-encode bit-exact.

Phase 2, bit-flip refinement
- For each key, take 3–5 seeds from the corpus and flip bits 0–104 (§3.2). Name the fields, find don't-care bits, and identify form bits.
- Sweep every modifier field to fill in value tables the compiler never emitted. Record INVALIDs as constraints.
- Output a declarative spec per arch (YAML/JSON): `{opcode, form, fields:[{name, lo, width, kind, enum?}], fixed_bits, constraints}`. Generate both the encoder and the decoder from it, so they can't drift.

Phase 3, validation (gates)
1. Round-trip on the corpus: decode(word) → text equals nvdisasm text, and encode(text) → word is bit-exact, at 100% on the training corpus.
2. Held-out cubins: compile a different program set (CUTLASS, our own kernels) and require the same 100%. Misses point to keys not covered yet.
3. Differential fuzzing: random words drawn from the spec's legal space go through our decoder and nvdisasm, and must match. Random words outside the spec should be INVALID in both.
4. Whole-cubin reassembly: disassemble → reassemble → `cmp` against the original `.text` (CuAssembler's standard test).
5. Hardware semantics on A100/RTX 30xx: run patched kernels on input vectors and compare against a reference model. This catches nvdisasm-valid-but-hardware-different cases (NAK/EoSS practice).
6. Independent cross-check: compare field positions against Mesa NAK's `sm70` encoder (MIT), reading it as documentation. Disagreements become bit-flip test cases.

Exit criterion for sm_80: gates 1–4 at 100% on held-out cubins from ptxas at two toolkit versions, and gate 5 passing for the ALU, memory, and branch core set.

## 6. Open questions to resolve empirically

- The exact opcode width and which instructions use form bits 9–11 (§2.2).
- UR/UP encoding of guards on uniform ops. Whether the UR count grows on sm_90+.
- The bit ranges for predicate fields and the FP abs/neg/sat/rounding bits, all [R].
- sm_80 vs. sm_86 vs. sm_89 deltas.
- Whether the `-b` raw mode and cubin mode ever disassemble the same word differently (relocations, branch targets).

## Sources

Every URL is inline next to the claim it supports. The main ones: [CuAssembler](https://github.com/cloudcores/CuAssembler), [turingas](https://github.com/daadaada/turingas), [maxas](https://github.com/NervanaSystems/maxas), [KeplerAs](https://github.com/xiuxiazhang/KeplerAs), [asfermi](https://github.com/hyqneuron/asfermi), [decuda](https://github.com/laanwj/decuda), [envytools](https://envytools.readthedocs.io/en/latest/envydis/index.html), [Decoding-CUDA-Binary](https://github.com/decodecudabinary/Decoding-CUDA-Binary/blob/master/tools/README.txt), [DocumentSASS](https://github.com/0xD0GF00D/DocumentSASS), [nv_isa_solver](https://github.com/kuterd/nv_isa_solver), [EoSS](https://engineering.purdue.edu/tgrogers/publication/avalos-asplos-2027/), [F2Asm](https://arxiv.org/abs/2608.20532), [Encoding synthesis](https://arxiv.org/abs/2609.18662), [Jia Volta](https://arxiv.org/abs/1804.06826), [Jia Turing](https://arxiv.org/abs/1903.07486), [Ampere](https://arxiv.org/html/2208.11174v1), [Hopper](http://arxiv.org/abs/2402.13499v1), [Blackwell](https://arxiv.org/abs/2507.10789), [Warp divergence](https://arxiv.org/abs/2607.23402), [basalt](https://github.com/sunnypatell/basalt), [blackwell-isa](https://github.com/kacper-daftcode/blackwell-isa), [Mesa NVK](https://docs.mesa3d.org/drivers/nvk.html), [CUDA EULA](https://docs.nvidia.com/cuda/eula/index.html), [EU 2009/24/EC](http://data.europa.eu/eli/dir/2009/24/oj/eng/xhtml).
