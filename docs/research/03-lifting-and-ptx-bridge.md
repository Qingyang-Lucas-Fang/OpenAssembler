# 03 - Lifting SASS and the PTX bridge

Status: research note consolidated from 2026-10-24 probes. Temporary probe inputs and scripts were removed after recording the observed idioms. Toolchain used for those observations: CUDA 12.8 ptxas/nvdisasm, -arch=sm_80, via env/cuda.sh.
Tag legend: [V-live] = compiled and disassembled in the original probe run; [src] = checked against a public source snapshot at that time; [web] = cited URL; [inf] = inference.

## 1. Prior art: lifters, decompilers, semantic tables

The original source comparison found **Mesa NAK** (MIT) to be a source of executable SASS semantics whose `Foldable::fold` implementations are checked against real GPUs in upstream `hw_tests.rs`. Accel-Sim contributes opcode-to-functional-unit classes. NVLift and CuLifter are academic SASS-to-LLVM lifters; their licensing must be checked before code reuse. This is a dated comparison, not a current survey of every lifter.

### 1.1 NVLift (Wan, Tan, Tian; Purdue; BAR@NDSS 2026) [public paper and project]
- **Pipeline:** `cuobjdump` → `nvdisasm` text, then tokenize, build a CFG, build an internal IR, compute reaching-defs and UD/DU chains, run bidirectional type inference to a fixpoint (default i32), apply per-opcode rules, and emit LLVM IR with `llvmlite`. About 3.5 kLoC of Python. It runs in Docker to pin the nvdisasm text format.
- **Semantics:** written by hand from docs, PTX analogies, TuringAs and DocumentSASS, then checked per instruction in **cuda-gdb**: set registers, `si`, compare, and occasionally hex-patch operands. Validation is kernel-level differential testing. The lifted IR goes through `llc` → PTX → SASS and runs on random inputs alongside a clang-built reference.
- **Scope:** **sm_75 only**, 47 of 163 Turing opcodes (88.4% dynamic coverage of 90 M instructions from NVIDIA .so files; IMAD 16.6%, ISETP 9.2%, MOV 7.7%, IADD3 7.5%). 11 of 11 kernels lift, but only 5 pass the differential test (6 hit IMA).
- **Modeling choices we should *not* copy:**
  - `@P` on a non-branch instruction gets its own basic block, which blows up the CFG given §2.2's aggressive if-conversion.
  - BSSY/BSYNC/BMOV are dropped.
  - SHFL, VOTE, MATCH and BRX are unsupported.
  - CALL is handled by inlining.
  - Control bits are discarded.
- **Choices worth keeping:**
  - Address spaces: global/shared/const/local map to LLVM addrspace 1/3/4/5.
  - A 64-bit value is the pair R(n+1):Rn.
  - Params start at `c[0x0][0x160]`.
  - Uniform-register ops are supported.
- Code: https://github.com/purseclab/Sass-LLVM-Lifter (no license; the README calls it a PoC).

### 1.2 CuLifter (Georgia Tech, arXiv 2604.27486, 2026) [web]
SASS → LLVM IR for sm_75/90/100/120 plus legacy sm_35/52. It uses **ψ-SSA for predication**, which matches our design in §6, and constraint-based type recovery with conflict detection. It has pattern aggregation for **IADD3/IADD3.X carry chains** and **cp.async**, and emulates SHFL/VOTE/BAR on the CPU with mutexes and condvars. 11,977 sm_90 kernels lift, with over 90% execution correctness. https://github.com/gthparch/CuLifter (no license).

### 1.3 Semantic and encoding tables
| Source | What it gives | Gaps | License |
|---|---|---|---|
| **Accel-Sim** (public source) | Ampere/Hopper opcode inventories and functional-unit classes; a trace frontend maps memory instructions to address spaces and recognizes cache hints. Hopper adds TMA/GMMA and asynchronous barrier classes. | Inventory/classification only; not instruction semantics, operand grammar or encoding. Some classes are simulator conveniences. | Check individual source-file headers before reuse |
| **Mesa NAK** (MIT, public source) | IR with distinct GPR/uniform/predicate/barrier/memory files, source modifiers and explicit scheduling dependencies. The sampled revision had executable constant folding for integer/address/bitwise families and an SM70+ encoder. | The sample had gaps in IMAD, SEL, MOV, floating operations and some predicate transfers; this is version-specific. | MIT |
| **NVBit** (NVIDIA binary instrumentation toolkit) | Runtime instruction/operand APIs and device register read/write helpers can support dynamic SASS instrumentation. | Requires its supported Linux/CUDA/GPU environment; distributions are governed by NVIDIA's toolkit EULA. We did not run it as part of this research. | NVIDIA EULA |
| DocumentSASS, nv_isa_solver, CuAssembler, TuringAs, maxas, F2Asm | Encodings and operand grammar, from nvdisasm's internal tables, fuzzing nvdisasm, or learned F2-affine encoders | Encoding only, no semantics | MIT (F2Asm unchecked) |

### 1.4 Other systems
- **GPUOcelot** (BSD-3, unmaintained): a PTX emulator and PTX → LLVM/x86 translator. It is a good reference for **PTX semantics and a PTX IR** (our bridge target), not for SASS.
- **ZLUDA** (Apache-2.0 per GitHub API): CUDA on AMD/Intel. It compiles **PTX** to LLVM and never touches SASS. Its PTX parser and PTX → LLVM semantic mapping are useful for the reverse direction of our bridge.
- **JEB 5.31** (commercial, 2025): SASS disassembler and decompiler for sm_70–121. It is the only shipping SASS decompiler. It leaves BRX tables, VOTE and 5-operand FMNMX untranslated, and documents the param base as c[0x0][0x160] (sm70–89), 0x210 (sm90) and 0x380 (sm100+).
- **Historical:** decuda (G80), asfermi (Fermi), maxas (Maxwell control codes), TuringAs (sm70–80 assembler).
- **Microbenchmarks:** Jia et al. Volta 1804.06826 and Turing 1903.07486, Abdelkhalik et al. Ampere 2208.11174, Luo et al. Hopper 2402.13499. They give latency and encoding facts, not semantics.
- No public Ghidra/IDA/Binary Ninja SASS module was found. `ghidra_falcon` targets the Falcon microcontroller.

### 1.5 What to reuse
1. Port NAK `fold()` to Python as `openassembler/semantics/` reference evaluators, and port the design of `hw_tests.rs` as our future on-GPU oracle.
2. Seed `specs/sm_80/*.json` opcode inventory from the Accel-Sim Ampere map (174 mnemonics plus a unit class) and cross-check against the nvdisasm-observed set.
3. Use NAK's `RegFile`/`SrcMod`/`Pred` taxonomy as the operand schema for our IR (§6).
4. Take ideas only from NVLift and CuLifter (no license).

URLs: https://github.com/purseclab/Sass-LLVM-Lifter · https://www.ndss-symposium.org/wp-content/uploads/bar2026-28.pdf · https://arxiv.org/abs/2604.27486v2 · https://github.com/gthparch/CuLifter · https://www.pnfsoftware.com/blog/reversing-nvidia-cuda-sass-code/ · https://gitlab.freedesktop.org/mesa/mesa/-/raw/main/src/nouveau/compiler/nak/hw_tests.rs · https://github.com/NVlabs/NVBit · https://github.com/accel-sim/accel-sim-framework · https://github.com/0xD0GF00D/DocumentSASS · https://github.com/kuterd/nv_isa_solver · https://arxiv.org/abs/2608.20532 · https://github.com/cloudcores/CuAssembler · https://github.com/daadaada/turingas · https://github.com/NervanaSystems/maxas · https://github.com/laanwj/decuda · https://github.com/gtcasl/gpuocelot · https://github.com/vosen/ZLUDA · https://arxiv.org/abs/2208.11174

## 2. Volta+ control flow on sm_80

The original control-flow probes used CUDA 12.8 `ptxas` and `nvdisasm`; temporary PTX inputs and run scripts were removed after the observations were consolidated. Everything below is [V-live] unless tagged otherwise.

### 2.1 Opcode inventory and CFG role

| SASS | Seen as | CFG semantics for our lifter |
|---|---|---|
| `BRA target` / `@P BRA` | `@!P0 BRA \`(.L_x_1)` | Direct branch. A predicated branch is a 2-way terminator: taken edge plus fall-through. The trailing `.L_x_N: BRA .L_x_N` after the last EXIT is a self-loop pad, so treat it as unreachable filler. |
| `BRA.CONV ~URZ, target` | in p_cf3, before a shfl in divergent code | Branches if the warp is **converged**. ptxas uses it to choose between an inline fast path (`SHFL.DOWN`) and a slow-path CALL to `$__internal_0_$__cuda_sm70_shflsync_down`, which runs `WARPSYNC 0xffffffff; SHFL...; RET`. The lifter should pattern-match both arms back to a single `shfl.sync`. |
| `BRX Rn -off (*"BRANCH_TARGETS a,b,c"*)` | p_cf2 from `brx.idx` | Indirect jump. ptxas builds the jump table in **c[0x2]** (`LDC R2, c[0x2][R0]` with R0 = idx*4), stores relative offsets there, and emits `SHF.R.S32.HI R3,RZ,0x1f,R2`, a sign-extension to 64-bit. nvdisasm's dataflow pass recovers the target list and prints it in the comment. That is our oracle for jump-table recovery. |
| `JMX` / `JMP` | not emitted on sm_80 | [inf] These are absolute-address forms. We never saw them. ptxas emits only relative BRX/CALL.REL. Keep them in the decoder anyway. |
| `CALL.REL.NOINC target` | p_cf2 | Direct call. The return address is materialised **by software**: `MOV R2, 0x60` sets R2 to the absolute offset of the instruction after the CALL. No hardware return stack is visible in the ISA. |
| `CALL.REL.NOINC Rn` | p_cf3 indirect call | Calls through a register. `MOV R2, 0x1f0; @P0 MOV R2, 0x210` picks the callee offset, and R20/R21 holds the return address. Function pointers become **in-module code offsets** (64-bit; the high half is set by `IMAD.MOV.U32 R21,RZ,RZ,0x0`). |
| `RET.REL.NODEC Rn \`(func)` | callee epilogue | Return to the address in Rn. The ABI return-address register varies per call site (R2, R20), so the lifter must track it with dataflow and not assume a fixed register. |
| `EXIT` / `@P EXIT` | p_cf2 `@P0 EXIT` | Thread termination, so a terminator with no successors. A predicated EXIT turns `if (tid>100) return;` into a 2-way node. |
| `BPT.TRAP 0x1` | from `trap` | Terminator (abort). |
| `BSSY Bn, target` | before every divergent region | Pushes a reconvergence point (target = the label of the matching BSYNC's fall-through) into convergence-barrier register Bn (B0..B15). Has no effect on dataflow. |
| `BSYNC Bn` | at the join | Waits until every thread recorded in Bn arrives. The BSSY target label is the instruction right after BSYNC, i.e. `.L_x_0:` follows `BSYNC B0`. |
| `WARPSYNC mask` | only in the shfl slow path | Explicit warp reconvergence. `bar.warp.sync 0xffffffff` in already-converged code became a plain `NOP` (p_cf2 @0x140), and `bar.warp.sync %r` with a parameter mask was **deleted** (p_cf5). ptxas proves convergence and drops the sync. |
| `BAR.SYNC.DEFER_BLOCKING 0x0` | from `bar.sync 0` | CTA barrier. Not a CFG edge, but it is a memory and convergence fence. |
| `NOP` | padding / removed sync | Ignore. |

The public Accel-Sim Ampere opcode inventory also lists, but the original probes did not elicit: `BREAK Bn`, `BMOV`, `KILL`, `NANOSLEEP`, `YIELD`, `RTT` and `BPT`. This is an inventory comparison, not a claim that every form is valid on every SM80 device.

### 2.2 Structural observations
1. **If-conversion is aggressive.** For a divergent if/else whose arms are pure ALU (p_cf1), ptxas fully predicates both arms (`@P0 IMAD`, `@!P0 IADD3`) under one `BSSY B0` and keeps no branch at all. Even with memory side effects (p_cf5 inner if), the arms became `@P1 STG` / `@!P1 STG`. A real `BRA` survives only when an arm is large, contains a call, or contains a warp-synchronous op. Consequence: the lifter must handle **predicated straight-line code as the normal case** and must not depend on branches for if-structure recovery. PTX `selp`/`@p` structure has to be re-created from the guards.
2. **BSSY/BSYNC bracket each divergent region**, including loops with divergent exits. In p_cf4 a `nounroll` loop with a data-dependent `@P1 BRA` break sits inside `BSSY B0 ... BSYNC B0`, and the BSYNC is placed at the loop exit before the warp-collective `SHFL.IDX`/`VOTE`. The bracket pairs give **a free single-entry/single-exit region tree**, which we should use as a structuring hint. They are also the place where warp collectives are legal again.
3. **Nesting uses separate B registers.** In p_cf1, the outer region uses B0 and the unrolled loop's remainder uses B1. In p_cf3, B6 brackets an indirect call because the call may diverge. B-register allocation tells us nesting depth.
4. **Loops**: plain backward `@!P BRA`. With unknown trip count, ptxas unrolls by 4, adds a remainder ladder (p_cf1: main body `.L_x_4`, then a 2-iteration block and a 1-iteration block guarded by `@!P1`), and uses `PLOP3.LUT P0, PT, PT, PT, PT, 0x8, 0x0` to set predicate constants (LUT 0x8 with all-PT inputs gives false; this is PLOP3's mov-false idiom). Recovering the original loop requires undoing the unroll, or accepting an unrolled CFG as equivalent.
5. **Divergence affects warp ops.** A shfl inside possibly-divergent code produces the BRA.CONV + internal-function double path. Outside divergence, it is a single `SHFL.*`. `vote.sync.ballot.b32` lowers to `VOTE.ANY Rd, PT, Pp` (the ballot mask goes to Rd and the any result to a predicate that is PT here). `vote.sync.all.pred` lowers to `VOTE.ALL P0, P0`.
6. **Calls**: callees sit after the caller's final EXIT in the same `.text.<kernel>` section (p_cf2 0x1b0..0x1e0, p_cf3 0x1b0..0x220). There is no separate symbol per callee, so nvdisasm labels them `$cf_call$helper` or not at all. Function discovery has to come from CALL targets plus the `MOV Rret, imm` immediate before each CALL, not from symbols.

### 2.3 nvdisasm CFG/liveness outputs (ground truth for stages 2-3)
- `-bbcfg [-poff]`: graphviz `digraph`, one `subgraph "cluster_<kernel>"`, nodes named by label (`".L_x_2"`) or `BBn` for unlabeled blocks. Each node ends in a port `<exit0>` on the terminator. Edges: `exitK:e` = taken edge, `exitK:s` = fall-through, `entry:s` = unconditional fall-through. -poff adds the `0060:` offsets. This is directly parseable into an edge list for diffing against our CFG.
- `-cfg`: "hyperblock" CFG. Predicated branches do **not** split nodes. Instead a node holds several `<exitK>`/`<exitpostK>` ports (side exits), so p_cf4's loop body is one node with two exits. Use -bbcfg for comparison.
- `-plr` (`-lrm wide|narrow|count`): per-instruction register liveness for **GPR, PRED and UGPR** columns with `^` def, `v` use, `x` use+redef, `:` live, and `#` live count. 64-bit pairs show as two columns (`R2,R3` in `IMAD.WIDE`). It reflects post-allocation physical liveness, which is exactly what our register-tracking stage must reproduce. Note UR4/UR5 (`ULDC.64 UR4, c[0x0][0x118]`, the global memory descriptor) stay live across the whole kernel and are used by every LDG/STG (`v v` at each memory op). The `.E` memory ops implicitly read UR4:UR5, so our def-use model needs that implicit operand.
- `-json`: structured disassembly (see section 5).

## 3. PTX -> sm_80 idiom table

Method: the original probe generator emitted one `.entry` per idiom family (integer, conversion, floating point, memory, tensor, warp, special register and hard cases). Each entry loaded operands from an output buffer and stored results back, preventing constant folding and dead-code removal. Every row below is [V-live] from the original CUDA 12.8 `ptxas -O3` disassembly unless marked otherwise. Notation: a, b, c are 32-bit inputs and A, B are 64-bit pairs (lo, hi). The temporary generated PTX and dumps have been removed.

Pitfall found during probing: **warp-uniform inputs get folded.** If every lane loads the same address, `shfl.sync` is deleted outright (the value is already uniform), and `atom.add` is replaced by a single-lane atomic times `popc(activemask)`. The warp rows below therefore mix in `%tid.x` (the "warpv" group). Our corpus generator must always use lane-varying data.

### 3.1 Integer
| PTX | sm_80 SASS | Notes |
|---|---|---|
| `mov.u32 r, imm` (+use) | folded: `IADD3 R5, R0, 0x12345678, RZ` | Immediates fold into consumers. A standalone constant uses `MOV`, `IMAD.MOV.U32 Rd, RZ, RZ, imm`, or **`HFMA2.MMA Rd, -RZ, RZ, 0, 2.38e-07`** (that is 0x00000004 written as two f16 halves) as a load-imm on the FMA-pipe path. All three are `mov`, so the lifter must canonicalise them. |
| `add.s32` / `sub.s32` | `IADD3 Rd, a, b, RZ` / `IADD3 Rd, a, -b, RZ` | Two-input adds pad with RZ. |
| `add; add` (3 operands) | `IADD3 R5, R4, R0, R5` | Fuses into one op. |
| `neg.s32` | `IADD3 Rd, -a, RZ, RZ` | |
| `abs.s32` | `IABS Rd, a` | |
| `mul.lo.s32` | `IMAD Rd, a, b, RZ` | No IMUL on sm_80. |
| `mad.lo.s32` | `IMAD Rd, a, b, c` | |
| `mul.hi.s32/u32` | `IMAD.HI Rd, a, b, RZ` / `IMAD.HI.U32` | |
| `mul.wide.s32/u32` | `IMAD.WIDE Rd:Rd+1, a, b, RZ` / `IMAD.WIDE.U32` | Writes a 64-bit pair. Also used for `idx*4 + base` addressing: `IMAD.WIDE R2, R4, R5(=4), c[0x0][0x168]`. |
| `mul.wide.u32 + add.u64` | `IMAD.WIDE.U32 R4, R7, 0x12345, R4` | Fused. The 64-bit addend comes from a register pair. |
| `add.u64` | `IADD3 lo, P0, Alo, Blo, RZ ; IADD3.X hi, Ahi, Bhi, RZ, P0, !PT` | Carry travels through P0. |
| `sub.u64` | `IADD3 lo, P0, Alo, -Blo, RZ ; IADD3.X hi, Ahi, ~Bhi, RZ, P0, !PT` | Two's-complement split: -B = ~B + 1, where the +1 comes from the negated low half and `~` is applied to the high half. |
| `neg.s64` | `IADD3 lo, P0, RZ, -Alo, RZ ; IADD3.X hi, RZ, ~Ahi, RZ, P0, !PT` | |
| `mul.lo.u64` | `IMAD R3, Ahi, Blo, RZ ; IMAD.WIDE.U32 Rd, Alo, Blo, RZ ; IMAD R3, Alo, Bhi, R3 ; IADD3 Rd+1, Rd+1, R3, RZ` | Schoolbook multiply. `mad.lo.u64` is the same with C as the WIDE addend. |
| `mul.hi.u64` | 4x `IMAD.WIDE.U32` with carry-out `P0` + `IMAD.WIDE.U32.X` + `IADD3 RZ, P0, ...` | `IADD3 RZ, P0, x, y, RZ` writes nothing except the carry, so it acts as a carry-generate op. |
| `add.cc; addc` | `IADD3 R5, P0, a, b, RZ ; IADD3.X R5, b', c, a, !PT, P0` | PTX's implicit CC register becomes an explicit predicate. |
| `sub.cc; subc` | `IADD3 R5, P0, a, -b, RZ ; IMAD.X R0, c, 0x1, ~a, P0` | **IMAD.X** with multiplier 1 works as an add-with-carry. |
| `mad.lo.cc; madc.hi` | merged: `IMAD.WIDE.U32` | ptxas recognises the 64-bit pattern. |
| `min.s32` / `max.u32` | `IMNMX Rd, a, b, PT` / `IMNMX.U32 Rd, a, b, !PT` | Trailing predicate: **PT = min, !PT = max**. |
| `shl.b32 imm/reg` | `SHF.L.U32 Rd, a, sh, RZ` (also `IMAD.SHL.U32 Rd, a, 2^k, RZ` for constant shifts) | |
| `shr.u32` / `shr.s32` | `SHF.R.U32.HI Rd, RZ, sh, a` / `SHF.R.S32.HI Rd, RZ, sh, a` | The value sits in the **high** slot and the low slot is RZ. |
| `shl.b64 13` | `SHF.L.U64.HI hi, lo, 0xd, hi ; SHF.L.U32 lo, lo, 0xd, RZ` | |
| `shr.s64 reg` | `SHF.R.S64 lo, lo, sh, hi ; SHF.R.S32.HI hi, RZ, sh, hi` | |
| `shr.u64` of `{a,b}` by 20 | `SHF.R.U64 lo, lo, 0x14, hi ; SHF.R.U32.HI hi, RZ, 0x14, hi` | |
| rotate `(a<<7)|(a>>25)` | `SHF.L.U32.HI Rd, a, 0x7, a` | Funnel shift with the same register on both sides. |
| `shf.l.wrap.b32` | `SHF.L.W.U32.HI Rd, lo, sh, hi` | `.W` means wrap (shift & 31) and the default is clamp. |
| `and/or/xor.b32` | `LOP3.LUT Rd, a, b, RZ, 0xc0/0xfc/0x3c, !PT` | 0xc0 = 0xf0&0xcc, 0xfc = 0xf0\|0xcc, 0x3c = 0xf0^0xcc. |
| `not.b32` | `LOP3.LUT Rd, RZ, a, RZ, 0x33, !PT` | 0x33 = ~0xcc, with a in slot b. |
| `(a & ~b) \| c` | `LOP3.LUT R5, c, a, b, 0xf4, !PT` | Operands get reordered. 0xf4 = 0xf0 \| (0xcc & ~0xaa) in that order. |
| `lop3.b32 ..., 0xE8` (majority) | `LOP3.LUT R5, a, b, c, 0xe8, !PT` | Passes through 1:1. |
| `and.b32 r, a, 1; setp.eq` | `LOP3.LUT P0, RZ, a, 0x1, RZ, 0xc0, !PT` | LOP3 can write a **predicate** (result != 0) and discard the GPR. |
| `bfe.u32 a, 4, 8` | `SHF.R.U32.HI t, RZ, 0x4, a ; SGXT.U32 Rd, t, 0x8` | No BFE on sm_70+. SGXT is sign/zero-extend from bit n. |
| `bfi.b32 a, b, 8, 4` | `SHF.L.U32 t, a, 0x8, RZ ; LOP3.LUT Rd, t, 0xf00, b, 0xe2, !PT` | 0xe2 is a mux: `(x & m) \| (z & ~m)`. |
| `prmt.b32 a, b, 0x5140` | `PRMT Rd, a, 0x5140, b` | The selector is the **middle** operand. |
| byte extract + `shl 24` + or | `SHF.R.U32.HI; LOP3 &0xff; LEA R5, b, t, 0x18` | **LEA** used as `(b<<24)+t`. |
| `popc` / `brev` | `POPC` / `BREV` | |
| `clz.b32` | `FLO.U32 t, a ; IADD3 Rd, -t, 0x1f, RZ` | FLO = find leading one (bit index, 0xffffffff if none), so clz = 31 - flo. |
| `setp.lt.s32 + selp a,c` | `ISETP.GE.AND P0, PT, a, b, PT ; @P0 LDG.E R5, [c-addr]` | ptxas inverted the compare and **sank the selp into a predicated load of c**. Selects are not preserved as SEL. |
| `setp + setp.ne.and + selp 1,0` | `ISETP.GE.AND P0 ; ISETP.NE.U32.AND P0, PT, c, RZ, !P0 ; SEL R5, RZ, 0x1, !P0` | PTX `.and` with a negated input maps to the combining field. |
| `setp.lt.u64` | `ISETP.GE.U32.AND P0, PT, Alo, Blo, PT ; ISETP.GE.U32.AND.EX P0, PT, Ahi, Bhi, PT, P0` | `.EX` chains through the low compare. |
| `setp.lt.s64; setp.eq.or.s64` | `ISETP.GE.U32.AND P1; ISETP.EQ.U32.AND P0; ISETP.GE.AND.EX P1,..,P1; ISETP.EQ.OR.EX P0, PT, hi, hi, !P1, P0` | Only the high half is signed (`.EX` without `.U32`). |
| `cvt` / `shl` / `add` (u64 index) | `LEA lo, P0, idx, base_lo, 0x3 ; LEA.HI.X hi, idx, base_hi, RZ, 0x3, P0` | Address formation `base + (u64)idx<<3`. |
| same with s64 | `SHF.R.S32.HI s, RZ, 0x1f, idx ; LEA lo, P0, idx, Blo, 0x4 ; LEA.HI.X hi, idx, Bhi, s, 0x4, P0` | The sign word goes in LEA.HI.X's third source. |
| `(u64)a << 36 + B` | `IADD3 lo, P0, RZ, Blo, RZ ; IMAD.U32.X hi, a, 0x10, Bhi, P0` | Shift >= 32 only touches the high half. ptxas used IMAD.X as a "shift by 4 + add + carry". |
| `div.u32` | ~20 ops: `I2F.U32.RP ; MUFU.RCP ; IADD3 +0xffffffe ; F2I.FTZ.U32.TRUNC.NTZ ; IMAD.HI.U32 (Newton) ; IMAD ; 2x {ISETP.GE.U32 ; @P IADD3 q+1} ; @!P2 LOP3 ~0` (div-by-0 gives 0xffffffff) | Inline idiom. Lift the whole window into one `udiv` node. The `+0xffffffe` trick biases the float reciprocal down so the estimate never exceeds the true quotient. |
| `rem.s32` | same core with `IABS` on both inputs, `@!P2 IADD3 -r` for the sign, and `@!P0 LOP3 0x33` | |
| `div.s32 a, 7` | same reciprocal sequence on **constant 7** (`I2F.U32.RP R6, 0x7`) | ptxas 12.8 does **not** use a magic-multiply here, which is unusual. [V-live] |
| `div.u64` | `CALL.REL.NOINC $__internal_0_$__cuda_sm20_div_u64` (~70-op subroutine with I2F.U64.RP / MUFU.RCP / IMAD.WIDE.U32 chains) | 64-bit div/rem are **out-of-line internal functions** placed after EXIT. Recognise them by symbol name. |

### 3.2 Conversions
| PTX | sm_80 SASS | Notes |
|---|---|---|
| `cvt.s64.s32` | `SHF.R.S32.HI hi, RZ, 0x1f, lo` | Sign word only; the low half is reused in place. |
| `cvt.u64.u32` | `HFMA2.MMA hi, -RZ, RZ, 0, 0` (hi = 0) | |
| `cvt.u32.u64` | none (use the low register) | Free. The lifter sees a register-pair half reference. |
| `cvt.s32.s8` | `PRMT Rd, a, 0x8880, RZ` | Byte 0 = a.b0, bytes 1-3 = sign(a.b0) (selector nibble 8 = msb-replicate of byte 0). |
| `cvt.u32.u16` | `LOP3.LUT Rd, a, 0xffff, RZ, 0xc0, !PT` | |
| `cvt.sat.u8.s32` | `I2I.U8.S32.SAT t, a ; LOP3 &0xff` | |
| `cvt.rn.f32.s32` | `I2F Rd, a` | Defaults (.S32, .RN) are not printed. |
| `cvt.rzi.s32.f32` | `F2I.TRUNC.NTZ Rd, a` | |
| `cvt.f64.f32` / `cvt.rn.f32.f64` | `F2F.F64.F32` / `F2F.F32.F64` | |
| `cvt.rni.f32.f32` | `FRND Rd, a` (and `.TRUNC/.FLOOR/.CEIL` for the other modes, [src NAK]) | |
| `cvt.rn.f16.f32` | `F2FP.PACK_AB Rd, RZ, a` | Packs into the low half and zeroes the high half. |
| `cvt.f32.f16` | `HADD2.F32 Rd, -RZ, a.H0_H0` | f16 to f32 is an HADD2 with -0. |
| `cvt.rn.bf16.f32` | `F2FP.BF16.PACK_AB Rd, RZ, a` | |
| `cvt.rn.f16x2.f32 a, b` | `F2FP.PACK_AB Rd, a, b` | a goes to the high half, b to the low half, matching PTX's `d = {a→hi, b→lo}`. |
| `mov.b32 r, {h, h}` | `PRMT R0, R0, 0x5410, R0` | 0x5410 = {a.lo16, b.lo16}, the standard half-pack selector. |

### 3.3 Floating point
| PTX | sm_80 SASS | Notes |
|---|---|---|
| `add/sub/mul/fma.rn.f32` | `FADD a, b` / `FADD a, -b` / `FMUL` / `FFMA` | |
| `abs; neg; add` | `FADD R5, -\|a\|, b` | Modifiers `-` and `\|\|` fold into sources. |
| `mul.rz.ftz.sat.f32` | `FMUL.FTZ.RZ.SAT` | |
| `min.f32` | `FMNMX Rd, a, b, PT` (PT = min) | `max(x, 0)` gives `FMNMX Rd, RZ, x, !PT`. |
| `setp.gtu.f32 + selp` | `FSETP.GTU.AND P0, PT, b?, a?, PT ; @!P0 LDG` | Compare operands are swapped/inverted and the select is sunk into a predicated load again. |
| `setp.lt.f32; setp.ge.and.f32` + selp | `FSETP.GEU.AND P0 ; FSETP.GE.AND P0, PT, c, 1, !P0 ; FSEL Rd, a, b, P0` | lt is negated to GEU (unordered-aware). FSEL is the f32 select. |
| `div.rn.f32` | `MUFU.RCP ; FCHK P0, a, b ; 4x FFMA Newton ; @!P0 BRA` + slow path `CALL $__internal_3_$__cuda_sm3x_div_rn_noftz_f32_slowpath` | **FCHK** flags operands that need the slow path. |
| `div.approx.f32` | `FSETP.GEU \|b\|, 1.17e-38 ; @!P0 FMUL b, 2^24 ; @!P0 FMUL a, 2^24 ; MUFU.RCP ; FMUL` | Denormal pre-scale. `div.full` adds the 2^126 overflow scale (×0.25). |
| `rcp.rn.f32` | exponent range check, then the `MUFU.RCP ; FFMA -1 ; FADD.FTZ ; FFMA` fast path, else `CALL ..._rcp_rn_f32_slowpath` | |
| `rcp.approx.ftz.f32` | `MUFU.RCP` | |
| `sqrt.rn.f32` | `MUFU.RSQ` + 2 FMUL + 2 FFMA, or `CALL ..._sqrt_rn_f32_slowpath` | |
| `sqrt.approx.f32` / `rsqrt.approx` | `[denorm scale] MUFU.SQRT` / `MUFU.RSQ` | |
| `ex2.approx.ftz` | `MUFU.EX2` | Without .ftz: `FSETP.GEU a, -126 ; @!P0 FMUL a, 0.5 ; MUFU.EX2 ; @!P0 FMUL r, r, r`. |
| `lg2.approx` | `[denorm scale ×2^24] MUFU.LG2 ; @!P0 FADD -24` | |
| `sin/cos.approx` | `FMUL.RZ t, a, 0.15915494 (1/2π) ; MUFU.SIN/COS` | The MUFU input is in **revolutions**. |
| `tanh.approx.f32` | `MUFU.TANH` | sm_75+. |
| `add/fma.rn.f64` | `DADD` / `DFMA` | |
| `div.rn.f64` | `MUFU.RCP64H` + DFMA Newton + `CALL $__internal_0_$__cuda_sm20_div_rn_f64_full` | |
| `add.f16x2` | `HFMA2.MMA Rd, a, 1, 1, b` | f16x2 add is lowered to an **FMA with 1.0**, i.e. a*1+b. HADD2 also exists (see the scalar case). |
| `fma.rn.f16x2` | `HFMA2.MMA` (`.MMA` = issued on the tensor/FMA-heavy pipe variant; same semantics) | |
| `add.f16` (scalar halves) | `HADD2 R0, R0.H0_H0, R0.H1_H1` | Swizzle suffixes `.H0_H0`, `.H1_H1`, `.F32`. |
| `fma.rn.bf16x2` | `HFMA2.BF16_V2` | sm_80 bf16 arithmetic is only fma/min/max. `add.bf16x2` needs sm_90 (ptxas error). |
| `max.bf16x2` | `HMNMX2.BF16_V2 Rd, a, b, !PT` | |
| `setp.lt.f16x2 p\|q` (use q) | `HSETP2.LT.AND P0, PT, a.H1_H1, b.H1_H1, PT` | ptxas dropped the unused lane-0 predicate. |

### 3.4 Memory
| PTX | sm_80 SASS | Notes |
|---|---|---|
| `ld.param` | **no instruction**: becomes `c[0x0][0x160+off]` operands | Kernel params live in constant bank 0 starting at 0x160 on sm_80. `ld.param.u64` gives `MOV R2, c[0x0][0x160]; MOV R3, c[0x0][0x164]`. |
| `cvta.to.global` | none | Generic and global addresses are identical on this ABI. |
| (implicit) | `ULDC.64 UR4, c[0x0][0x118]` | Global-memory **descriptor** (64-bit). Every `.E` global/generic access reads UR4:UR5 implicitly (shown in `-plr`). |
| `ld.global.u32` / `.v4.u32` / `.u64` | `LDG.E Rd, [Ra.64+imm]` / `LDG.E.128` / `LDG.E.64` | `.E` means a 64-bit address with the descriptor. The offset is a signed 24-bit immediate. The vector destination is consecutive registers Rd..Rd+3. |
| `ld.global.nc` | `LDG.E.CONSTANT` | |
| `ld.global.cg.u8` / `.s16` | `LDG.E.U8.STRONG.GPU` / `LDG.E.S16` | `.cg` (cache-global) shows up as `.STRONG.GPU`. |
| `ld.volatile.global` / `st.volatile` | `LDG.E.STRONG.SYS` / `STG.E.STRONG.SYS` | |
| `ld.acquire.gpu` ; `st.release.gpu` | `LDG.E.STRONG.GPU ; CCTL.IVALL ; MEMBAR.ALL.GPU ; ERRBAR ; STG.E.STRONG.GPU` | Acquire = strong load + L1 invalidate. Release = membar before the store. |
| `membar.gl` ; `fence.sc.cta` | `MEMBAR.SC.GPU ; ERRBAR ; CCTL.IVALL` ; `MEMBAR.SC.CTA` | |
| `ld/st` (generic) | `LD.E` / `ST.E` | |
| `st.shared` / `ld.shared` | `STS [Ra+imm]` / `LDS Rd, [Ra+imm]` | Shared addresses are 32-bit and **CTA-relative**: `mov %t, smem` with the only smem symbol gives `[RZ]` or `[0x10]` absolute offsets. `.v2.u32` gives `STS.64`. |
| `.local` array | `IADD3 R1, R1, -0x80, RZ` (stack alloc) ; `STL [R5]` ; `LDL Rd, [R1+0x8]` | **R1 is the stack pointer**, initialised from `c[0x0][0x28]` at kernel entry (every kernel starts with that MOV). Local addresses are offsets relative to R1. |
| `ld.const` of a module `.const` | `LDC R0, c[0x3][R0]` ; immediate offset gives the inline `c[0x3][0x4]` operand | **User .const goes in bank 3.** Bank 2 holds BRX jump tables. Bank 4 holds global-symbol addresses: `ld.global [gvar+8]` became `MOV R2, c[0x4][0x0]; MOV R3, c[0x4][0x4]; LDG [R2.64+0x8]`, i.e. the relocated address of `gvar` is stored in c[0x4]. |
| `atom.global.add.u32` (lane-varying addr) | `ATOMG.E.ADD.STRONG.GPU PT, Rd, [Ra.64], Rb` | The PT is the unused success-predicate output. |
| `atom.global.add.u32` (uniform addr) | **warp-aggregated**: `VOTE.ANY R4, PT, PT` (activemask) → if all lanes are active, an inclusive scan via `SHFL.UP` ×5 and a leader `ATOMG`; else a per-lane ATOMG. For constant/uniform data: `VOTEU.ANY UR, UPT, PT ; UFLO ; UPOPC ; S2R SR_LANEID ; ISETP.EQ leader ; @P0 ATOMG ; SHFL.IDX broadcast ; S2R SR_LTMASK ; POPC ; IMAD` | ptxas does atomic aggregation itself. To recover a single `atom` the lifter must recognise this ~15-op template. |
| `red.global.add` | `REDUX.SUM UR10, R0` (warp sum to a uniform register) + leader `@P0 RED.E.ADD.STRONG.GPU` | |
| `atom.cas.b32` | `ATOMG.E.CAS.STRONG.GPU PT, Rd, [Ra], Rcmp, Rnew` | Note `[R4]` without .64 here, an nvdisasm print quirk. |
| `atom.add.f32` | `ATOMG.E.ADD.F32.FTZ.RN.STRONG.GPU` | |
| `atom.exch.b64` | `ATOMG.E.EXCH.64.STRONG.GPU` | |
| `atom.sys.max.s32` | `ATOMG.E.MAX.S32.STRONG.SYS` | Scope goes to `.STRONG.{CTA,GPU,SYS}`. |
| `atom.shared.add` | aggregated + `@P0 ATOMS.ADD Rd, [0x10], Rv` | |
| `cp.async.ca/.cg ... 16` | `LDGSTS.E.128 [Rs], [Rg.64+imm]` / `LDGSTS.E.BYPASS.128` | `.cg` = BYPASS (L1). Size 4 gives `LDGSTS.E`. |
| `cp.async.commit_group` / `wait_group 0` / `wait_all` | `LDGDEPBAR` / `DEPBAR.LE SB0, 0x0` | wait_group N gives `DEPBAR.LE SB0, N`. SB0 is a scoreboard counter. |

### 3.5 Tensor core
| PTX | sm_80 SASS |
|---|---|
| `mma.sync.m16n8k16.row.col.f32.f16.f16.f32` | `HMMA.16816.F32 Rd, Ra, Rb, Rc` (A = 4 regs, B = 2, C/D = 4, all consecutive) |
| `mma.sync.m16n8k8 ... bf16` | `HMMA.1688.F32.BF16` |
| `mma.sync.m16n8k8 ... tf32` | `HMMA.1688.F32.TF32` |
| `mma.sync.m16n8k32 ... s8.s8.s32` | `IMMA.16832.S8.S8 Rd, Ra.ROW, Rb.COL, Rc` |
| `mma.sync.m8n8k4 ... f64` | `DMMA.884` |
| `ldmatrix.sync.m8n8.x4.shared.b16` | `LDSM.16.M88.4 Rd, [Ra]` |
| `ldmatrix ... .x2.trans` | `LDSM.16.MT88.2` |

Fragments are register tuples, and ptxas inserts `MOV`s to make them contiguous. The lifter has to model HMMA as a vector op on `Rd..Rd+3`. Fragment-to-lane layout comes from the PTX ISA "Matrix fragments for mma.m16n8k16" tables.

### 3.6 Warp-level and barriers (lane-varying inputs)
| PTX | sm_80 SASS | Notes |
|---|---|---|
| `shfl.sync.idx/up/down/bfly b32 r\|p, a, b, c, mask` | `SHFL.IDX/UP/DOWN/BFLY Pout, Rd, Ra, Rb\|imm, Rc\|imm` | The c operand is clamp/segmask, printed as `0x1f`, `RZ`, or `0x1800`. **membermask is not encoded.** A 0x0000ffff mask disappears (sm_80 converged shuffles ignore it), and the mask only matters through the BRA.CONV/WARPSYNC slow path (§2). Pout = PT if `\|p` is unused. |
| `xor %b, %tid; shfl idx` | `LOP3.LUT R5, R4, R5, RZ, 0x3c ; SHFL.IDX PT, R5, R0, R5, 0x1f` | `and 31` was folded into a LOP3 (0x48). |
| `vote.sync.ballot.b32` | `VOTE.ANY Rd, PT, Pp` | |
| `vote.sync.any/all/uni.pred` | `VOTE.ANY P0, P0` / `VOTE.ALL P0, P0` / `VOTE.EQ P0, P0` | |
| `activemask.b32` | `VOTE.ANY Rd, PT, PT` | |
| `match.any.sync.b32` | `MATCH.ANY Rd, Ra` | |
| `redux.sync.add.u32` | `REDUX.SUM URd, Ra ; MOV Rd, URd` | Result goes to a **uniform** register. |
| `bar.sync 0` / `bar.sync 1, 64` / `bar.arrive 2, 64` / `barrier.sync.aligned 3` | `BAR.SYNC.DEFER_BLOCKING 0x0` / `BAR.SYNC.DEFER_BLOCKING 0x1, 0x40` / `BAR.ARV 0x2, 0x40` / `BAR.SYNC.DEFER_BLOCKING 0x3` | |
| `bar.red.popc.u32` | `BAR.RED.POPC.DEFER_BLOCKING 0x0, P0 ; B2R.RESULT Rd` | |
| `bar.warp.sync` | `NOP` or nothing when convergent; `WARPSYNC mask` otherwise | §2 |

### 3.7 Special registers
| PTX sreg | sm_80 SASS |
|---|---|
| `%tid.{x,y,z}` | `S2R Rd, SR_TID.{X,Y,Z}` |
| `%ctaid.x` | `S2R SR_CTAID.X`, or **`S2UR URd, SR_CTAID.X`** when the value is used uniformly |
| `%ntid.x` / `%nctaid.x` | `c[0x0][0x0]` / `c[0x0][0xc]` (cbank0 launch params; y,z at +4, +8) |
| `%laneid` / `%lanemask_lt` | `S2R SR_LANEID` / `S2R SR_LTMASK` |
| `%warpid` | `S2R SR_VIRTID ; SHF.R.U32.HI 8 ; SGXT.U32 7` (bits 8..14 of VIRTID) |
| `%nwarpid` | immediate `0x40` (constant 64 on sm_80) |
| `%smid` / `%nsmid` | `S2UR SR_VIRTUALSMID` / `c[0x0][0x10c]` |
| `%clock` / `%clock64` / `%globaltimer` | `S2UR UR, SR_CLOCKLO` + `DEPBAR.LE SB1` / `CS2R Rd, SR_CLOCKLO` (64-bit pair) / `CS2R SR_GLOBALTIMERLO` |
| `%gridid` / `%dynamic_smem_size` / `%envreg0` | `c[0x0][0x30]` / `c[0x0][0x2c]` / `c[0x0][0x88]` |
| `%total_smem_size` | `S2R SR_SMEMSZ` minus `c[0x0][0x114]` |
| stack pointer | `R1 = c[0x0][0x28]` |

These cbank0 offsets give our lifter's `c[0x0][k]` → PTX special-register map: 0x0/0x4/0x8 = ntid.xyz, 0xc/0x10/0x14 = nctaid.xyz, 0x28 = stack top, 0x2c = dynamic smem, 0x30 = gridid, 0x88+ = envreg, 0x10c = nsmid, 0x114 = reserved smem base, 0x118 = global memory descriptor, and 0x160+ = params. [V-live, including the y/z offsets (p_chk.ptx); `cp.async.wait_group 1` gives `DEPBAR.LE SB0, 0x1`]

## 4. Semantics of the hard opcodes

Sources: NAK `Foldable::fold` implementations in the upstream [Mesa NAK source tree](https://gitlab.freedesktop.org/mesa/mesa/-/tree/main/src/nouveau/compiler/nak), cross-checked against the original §3 probes [V-live]. Operand order is the **SASS print order**.

### 4.1 LOP3.LUT  `LOP3.LUT [Pd,] Rd, a, b, c, lut, Pin`
```
Rd = OR over i in 0..7 with lut bit i set of minterm_i(a, b, c),
     where i = (a_bit<<2)|(b_bit<<1)|c_bit, bitwise across 32 lanes
   ≡ for each bit: Rd[k] = (lut >> ((a[k]<<2)|(b[k]<<1)|c[k])) & 1
Pd = (Rd != 0)          // optional predicate output (LOP3.LUT P0, RZ, R3, 0x1, RZ, 0xc0)
```
- Constructing the LUT: evaluate the boolean function on `a=0xF0, b=0xCC, c=0xAA` (NAK `LogicOp3::SRC_MASKS`). Examples: and=0xC0, or=0xFC, xor=0x3C, not-b=0x33, mux `(a&b)|(~a&c)` = 0xCA. bfi's `0xE2` means `(a & b) | (c & ~b)` with the mask in slot b, so `LOP3 R5, t, 0xf00, R4, 0xe2` = `(t & 0xf00) | (R4 & ~0xf00)`. [V-live]
- The trailing `!PT` is a predicate *input* (in the sampled NAK field map, bits 87..90 plus invert bit 90). NAK always encodes it as false, i.e. `!PT`. Bit 80 is `.PAND`, and bits 81..84 are the predicate destination (7 = PT = discard). [src] [inf] semantics: `Pd = (Rd != 0) OR Pin`, or `AND Pin` with `.PAND`. Since ptxas only printed `!PT` in the original corpus, `Pd = (Rd != 0)` remains an inference for this use. Confirm nontrivial Pin on hardware before relying on it.
- Lift rule: canonicalise the LUT by dropping unused inputs (NAK `src_used`, `fix_src`) and pattern-match to and/or/xor/not/andn/orn/mux. Otherwise emit PTX `lop3.b32 d, a, b, c, lut` directly (round-trips 1:1, as `lop3_ptx` showed).
- `PLOP3.LUT Pd, Pq, a, b, c, lut0, lut1` is the same thing on predicates with two outputs. `PLOP3.LUT P0, PT, PT, PT, PT, 0x8, 0x0` is a mov-const-false idiom: all inputs are true, i = 7, and bit 7 of 0x08 is 0. In p_cf1 this initialises a loop-carried predicate. [V-live]

### 4.2 IADD3 / IADD3.X  `IADD3 Rd, [P0, [P1,]] a, b, c` / `IADD3.X Rd, [P0, P1,] a, b, c, Pc0, Pc1`
```
IADD3:   s = (u64)a' + (u64)b' + (u64)c'   with x' = (-x mod 2^32) as (~x + 1) if negated
         Rd = s[31:0];  P0 = s >= 2^32;  P1 = s >= 2^33
IADD3.X: s = (u64)a'' + b'' + c'' + Pc0 + Pc1   with x'' = ~x for the `~` modifier (bitwise NOT, no +1)
         Rd = s[31:0];  P0/P1 as above
```
- Two carry-outs exist because three 32-bit addends can carry 0, 1 or 2. NAK `OpIAdd3::fold` sets `overflow[0] = sum >= 1<<32` and `overflow[1] = sum >= 2<<32`, which are two thresholds, not two bits. A full 2-bit carry is `P0 + P1`. [src]
- **Negation in IADD3 is literal two's complement per operand**: `-x` adds `~x + 1` into the 64-bit sum, so `a + (-b)` with b = 0 produces a carry (`(~0)+1 = 2^32`). That is exactly what makes `IADD3 lo, P0, a, -b ; IADD3.X hi, ah, ~bh, RZ, P0` a correct 64-bit subtract. The carry is the "no borrow" flag. [src + V-live sub_u64]
- Lift: pair `IADD3 (carry-out Pk)` with the `IADD3.X/IMAD.X/LEA.HI.X` that consumes Pk into a single 64-bit add/sub. Only fall back to PTX `add.cc/addc` when the pair spans unrelated values.

### 4.3 ISETP / FSETP / DSETP / HSETP2  `ISETP.cmp[.U32].bop[.EX] Pd, Pq, a, b, Pacc[, Plow]`
```
c   = cmp(a, b)                    (signed unless .U32; FSETP: ordered unless *U)
.EX: if a == b then c = Plow       (high-word compare chains to the low-word result; sm70+ semantics)
Pd  = c  bop Pacc                  bop ∈ {AND, OR, XOR}, Pacc may be !P
Pq  = !c bop Pacc                  (second output; PT if unused)
```
- Lifting `.EX`: `ISETP.GE.U32.AND P0, PT, lo_a, lo_b, PT ; ISETP.GE.AND.EX P0, PT, hi_a, hi_b, PT, P0` is `setp.ge.s64 a, b`. Only the high word carries signedness. [V-live isetp_s64]
- The `Pacc` combine is exactly PTX `setp.cmp.bop`, e.g. `ISETP.NE.U32.AND P0, PT, c, RZ, !P0` = `setp.ne.and.u32 p, c, 0, !p0`. The PTX→SASS mapping often **inverts** cmp (lt→GE plus a negated use), so the lifter must normalise polarity.
- FSETP comparisons: EQ/NE/LT/LE/GT/GE are ordered (false on NaN); EQU/NEU/LTU/LEU/GTU/GEU are unordered (true on NaN); NUM/NAN test ordering. `|x|` and `-x` source modifiers apply. `.FTZ` flushes denormal inputs.
- Ampere also has `ISETP ... P0, P1` dual outputs (`Pq = !c bop acc`); NAK only uses the first. [src, inf for Pq]

### 4.4 SHF  `SHF.{L,R}[.W].{U32,S32,U64,S64}[.HI] Rd, lo, sh, hi`
```
x  = (u64)hi << 32 | lo
s  = .W ? sh & (bits-1) : min(sh, bits)        bits = 32 for *32, 64 for *64
r  = L ? x << s : (S ? (i64)x >> s : x >> s)   (shift out → 0 / sign)
Rd = .HI ? r[63:32] : r[31:0]
```
Lift table [V-live]: `shl.b32 a,n` = `SHF.L.U32 Rd, a, n, RZ`. `shr.u32` = `SHF.R.U32.HI Rd, RZ, n, a`. `shr.s32` = `SHF.R.S32.HI Rd, RZ, n, a`. A 64-bit shift is a pair with `.U64/.S64` on the low half and `.HI` on the high half. A funnel rotate is `SHF.L.U32.HI Rd, a, n, a`. Note the NAK quirk: on sm < 70 `!right` reads the high half implicitly. For sm_70+ follow the formula above. [src]

### 4.5 PRMT  `PRMT Rd, a, sel, b`
```
bytes = [a.b0, a.b1, a.b2, a.b3, b.b0, b.b1, b.b2, b.b3]   (index 0..7)
for k in 0..3: n = (sel >> 4k) & 0xF
               v = bytes[n & 7];  if (n & 8) v = (v & 0x80) ? 0xFF : 0x00   // msb replicate
               Rd.byte[k] = v
```
- This is the default (index) mode. NAK `PrmtSelByte`: bit 2 = source select, bits 0-1 = byte, bit 3 = sign-replicate. Other modes `.F4E .B4E .RC8 .ECL .ECR .RC16` correspond to PTX `prmt.b32.{f4e,b4e,rc8,ecl,ecr,rc16}`. [src]
- Common selectors [V-live]: `0x8880` = sign-extend byte 0 (cvt.s32.s8). `0x5410` = pack the low halves {a.lo, b.lo}. `0x7632` = pack the high halves. `0x3210` = identity. PTX `prmt.b32 d, a, b, s` maps to `PRMT d, a, s, b`. **Only the operand order changes**, which makes this a 1:1 lift.

### 4.6 IMAD family
```
IMAD[.U32]    Rd = (a * b + c)[31:0]                 (signedness only matters for .HI/.WIDE)
IMAD.HI[.U32] Rd = ((s|u 64)a * b)[63:32] + c    (c = RZ in plain mul.hi; carry-out P optional; see note)
IMAD.WIDE[.U32] Rd:Rd+1 = (s|u 64)a * b + C64     (C = 64-bit pair c:c+1, or a cbank 64-bit pair)
IMAD.X        Rd = a * b + c + Pc   (carry-in; with b=1: add-with-carry, e.g. IMAD.X R0, c, 0x1, ~a, P0)
IMAD.WIDE.U32.X  as WIDE with carry-in Pc and optional carry-out P (mul.hi.u64 sequence)
IMAD.MOV.U32 Rd, RZ, RZ, x  ≡ mov  ;  IMAD.SHL.U32 Rd, a, 2^k, RZ ≡ shl ;  IMAD.IADD Rd, a, 0x1, b ≡ add
IMAD.MOV Rd, RZ, RZ, -x ≡ neg   ;  IMAD.U32 Rd, RZ, RZ, URx ≡ mov from uniform
```
Note on IMAD.HI: in the `mul.hi.u64` sequence ptxas uses `IMAD.HI.U32 R8, P1, R7, R11, R8`. The c operand is added to the **high word**, so it acts as a 64-bit accumulator high-half, and the P output is the carry. We measured `mul.hi.u32` = `IMAD.HI.U32 Rd, a, b, RZ` [V-live]. The exact c-add-and-carry semantics of .HI with a non-RZ c are [inf] and should be confirmed by differential testing (§6.6).

The lifter must treat these IMAD aliases as **canonical ALU ops** (mov/shl/add/neg), because ptxas picks IMAD instead of IADD3/MOV/SHF to balance the FMA and ALU pipes. They carry no semantic intent. The same goes for `HFMA2.MMA Rd, -RZ, RZ, imm16, imm16` as a 32-bit mov-imm.

### 4.7 LEA / LEA.HI  `LEA Rd, [P,] a, b, sh` / `LEA.HI[.X] Rd, a, b, ahi, sh[, Pc]`
```
LEA:       Rd = (a << sh) + b                ; P = carry-out
LEA.HI:    Rd = (((u64)ahi<<32 | a) >> (32-sh))[31:0] + b   ; i.e. high bits of the 64-bit (ahi:a)<<sh
LEA.HI.X:  same + Pc (carry-in), modifiers: ~ on (a<<sh) and on b (bnot, for subtraction chains)
```
The canonical 64-bit address idiom is `LEA lo, P0, idx, base_lo, s ; LEA.HI.X hi, idx, base_hi, idx_sign_or_RZ, s, P0`, which computes `base + ((s64|u64)idx << s)`. The lift is a single `add.s64 d, base, idx64 << s`, or `mad.wide` when s ∈ {2,3}. [src NAK OpLea/OpLeaX + V-live lea_s64_idx]

### 4.8 SEL / FSEL / IMNMX / FMNMX
`SEL Rd, a, b, P` = `P ? a : b` (`!P` allowed). `FSEL` is the same on f32 (no NaN handling: bitwise select). `IMNMX Rd, a, b, P` = `P ? min(a,b) : max(a,b)`, signed unless `.U32`. `FMNMX` is the same with IEEE minNum/maxNum (NaN-avoiding). Probed: `min.s32` → PT, `max.u32` → !PT. [V-live]

### 4.9 P2R / R2P
```
P2R  Rd, PR, Ra, mask   : Rd = (Ra & ~mask) | (PRbits & mask)    PRbits = P6..P0 as bits 6..0
                          (.B1/.B2/.B3 variants place the byte at bits 8/16/24)
R2P  PR, Ra, mask       : for i in 0..6: if mask bit i: Pi = (Ra >> i) & 1
```
Observed [V-live p_p2r, r2p_test]:
- **Spill/save**: under predicate pressure ptxas emits `P2R R6, PR, RZ, 0x4` (save P2 into bit 2 of R6) and later restores with `ISETP.NE.AND P3, PT, R6, RZ, PT`, not R2P. So P2R is the spill, and the restore can land in a *different* predicate (P2→P3). The lifter must model P2R/R2P bitwise and let SSA dissolve the spill.
- **Bit test**: two `and a,1/2; setp.ne` became one `R2P PR, R0, 0x3`, which sets P0 = a[0] and P1 = a[1].
- Formula for the merge with a non-RZ Ra: [inf from NAK encoding + observed RZ-only use]. Verify by differential test.

## 5. Debug-info bridge (-lineinfo, nvdisasm -gp) [V-live]

The original `-lineinfo` probe used a SAXPY kernel with two `.file` entries and `.loc` directives, compiled with `ptxas -arch=sm_80 -lineinfo`. A second input added an inline-location record and `.debug_str` section.

### 5.1 What -lineinfo puts in the cubin
| Section | Content | nvdisasm flag |
|---|---|---|
| `.debug_line` (+ `.rel.debug_line`) | Standard **DWARF v2 line program**. SASS address → (CUDA file from `.file`, line from `.loc`) | `-g` (`-gi` adds the inline chain) |
| `.nv_debug_line_sass` (+ rel) | A second DWARF v2 line program with **no file table**. SASS address → **line number in `.nv_debug_ptx_txt`** | `-gp` |
| `.nv_debug_ptx_txt` | **The PTX source embedded verbatim**: the input with `.file`/`.loc` lines blanked out, NUL-separated lines (39 here) | (read directly) |
| `.debug_frame` | CFA info | – |

So the cubin carries its own PTX together with an exact SASS-pc → PTX-line map. **For any cubin built with -lineinfo (or -G), the bridge's ground truth is embedded in the binary.** nvcc `-lineinfo` produces the same sections, and many shipping libraries are built that way.

### 5.2 Decoded maps for p_line.cubin
The original small DWARF decoder reproduced nvdisasm's mapping exactly:
```
.debug_line         files=[saxpy.cu, helper.cuh]
  0000:1:3 0010:1:4 0040:1:5 0060:1:6 00c0:2:10 00d0:1:6 00f0:1:7
.nv_debug_line_sass (ptx line numbers)
  0000:7 0010:15 0020:17 0030:18 0040:20 0050:21 0060:26 0070:7 0080:26 0090:27
  00a0:28 00b0:29 00c0:31 00d0:33 00e0:34 00f0:37
```
Joined against the embedded PTX:

| SASS | PTX line (`.nv_debug_ptx_txt`) | CUDA |
|---|---|---|
| `0000 MOV R1, c[0x0][0x28]` | 7 `{` (prologue: stack pointer) | saxpy.cu:3 |
| `0010 S2R R4, SR_CTAID.X` | 15 `mov.u32 %r2, %ctaid.x` | :4 |
| `0020 S2R R3, SR_TID.X` | 17 `mov.u32 %r4, %tid.x` | :4 |
| `0030 IMAD R4, R4, c[0x0][0x0], R3` | 18 `mad.lo.s32 %r5, %r2, %r3, %r4` (`%ntid.x` from line 16 was folded into the c[0x0][0x0] operand, so line 16 owns **no** SASS) | :4 |
| `0040 ISETP.GE.AND P0 ...` | 20 `setp.ge.s32` | :5 |
| `0050 @P0 EXIT` | 21 `@%p1 bra DONE` (branch to ret becomes predicated EXIT) | :5 |
| `0060 HFMA2.MMA R5, ... (=4)` | 26 `mul.wide.s32 %rd5, %r5, 4` (the constant 4) | :6 |
| `0070 ULDC.64 UR4, c[0x0][0x118]` | 7 (prologue: memory descriptor, hoisted) | :6 |
| `0080 IMAD.WIDE R2, R4, R5, c[0x0][0x168]` | 26 (mul.wide fused with `add.s64` line 26 and `cvta` line 23) | :6 |
| `0090 IMAD.WIDE R4, ...c[0x0][0x170]` | 27 `add.s64 %rd7` | :6 |
| `00a0/00b0 LDG.E` | 28 / 29 | :6 |
| `00c0 FMUL R7, R2, R2` | 31 `mul.f32` | **helper.cuh:10** |
| `00d0 FFMA` / `00e0 STG.E` | 33 / 34 | :6 |
| `00f0 EXIT` | 37 `ret` | :7 |

### 5.3 Observations that matter for the bridge
1. **The granularity is one PTX line per SASS instruction**, finer than `.loc`. The `.nv_debug_line_sass` program changes row at almost every instruction. ptxas records which PTX instruction each SASS op was "born from", so this is a true per-instruction provenance map.
2. **Many-to-one and fusion.** When several PTX ops fuse (mul.wide + add.s64 + cvta → one IMAD.WIDE), the SASS op is attributed to **one** of them (line 26). The others (23, 24, 16 here) own no SASS. The bridge must treat attributions as "anchor" edges and recover fused partners by dataflow.
3. **Line 7 (`{`) is the bucket for synthesized prologue code**: the R1 stack init and the global-descriptor ULDC.64. These are ABI artefacts with no PTX counterpart and should be modeled as implicit in the lifter.
4. **Inlining** (`-gi`) shows `//## File "/src/helper.cuh", line 10 inlined at "/src/saxpy.cu", line 6`. DWARF's inlined_at chain survives, while `-g` shows only the innermost location.
5. **The JSON output (`-json`) carries no line info** even with `-gp`. Its keys are `function-name, start, length, other-attributes, sass-instructions[{predicate, opcode, operands, other-attributes{control-flow}}]`. So we decode `.nv_debug_line_sass` ourselves, which the decoder already does.
6. Without -lineinfo, none of these sections exist and the bridge must work from semantics alone (§6.6). The -lineinfo corpus is then the **training and evaluation set** for that semantic matcher: compile PTX with -lineinfo, strip the debug sections, lift, and score the recovered SASS→PTX map against the embedded one.

### 5.4 Bridge accuracy metric (proposed)
For each SASS instruction `i` with ground-truth PTX line `L(i)`, the lifter predicts a set `L'(i)`. Report hit rate = mean[L(i) ∈ L'(i)], and also per-PTX-line coverage: the fraction of PTX instruction lines that own at least one SASS op and are recovered. Keep prologue (line = `{`) and padding NOPs as separate classes.

## 6. Recommended IR design ("SIR")

The guiding constraint is that the target is PTX, which is itself a typed, virtual-register SSA-like ISA with predicates, explicit state spaces and warp intrinsics. SIR should therefore be **"PTX semantics over SSA values"** instead of LLVM IR. Lowering SIR → PTX is then nearly 1:1, and the hard work happens SASS → SIR. Two levels:

- **L0 (machine SSA):** one SIR op per SASS op with exact semantics (NAK-style), operating on 32-bit lanes and predicates. It is produced directly from the decoder.
- **L1 (PTX-level):** typed ops (`add.u64`, `div.u32`, `setp.lt.s64`, `atom.add`, `shfl.sync`) produced by idiom recognition on L0. PTX is printed from L1.

### 6.1 Values and SSA
- **Register files** (NAK taxonomy): `GPR` (R0–R254, RZ = const 0), `UGPR` (UR0–UR62, URZ), `Pred` (P0–P6, PT = const true), `UPred` (UP0–UP6, UPT), `Bar` (B0–B15 convergence barriers), and `SB` (scoreboards for DEPBAR/LDGDEPBAR). Also cbank operands `c[b][off]` and `c[b][Rx+off]`.
- **SSA over 32-bit physical lanes first**, then **merge pairs**. Decoding gives `R2.64` = (R2, R3). L0 keeps them as two 32-bit values, and a pass forms 64-bit values when every def and use of a pair is pairwise: IADD3/IADD3.X carry pairs, IMAD.WIDE, LEA/LEA.HI.X, `SHF.*.U64/.HI` pairs, LDG.E.64/STG.E.64 and `.64` addresses. This mirrors how `-plr` shows pairs as two columns (§2.3).
- Standard SSA construction (Cytron or Braun et al.) on the -bbcfg CFG. Phis at join blocks, with BSYNC blocks as natural join points.
- **Implicit operands made explicit**:
  - R1 = stack pointer (init `c[0x0][0x28]`).
  - UR4:UR5 = global memory descriptor (`c[0x0][0x118]`), an implicit input of every `.E` access. It is dropped at L1.
  - The return-address register of CALL/RET (§2.1).
  - SB tokens for cp.async.
- **Canonicalization** on L0 to remove ptxas pipe-balancing aliases: IMAD.MOV/MOV/HFMA2.MMA-imm/IMAD.U32-from-UR → `mov`, IMAD.SHL → `shl`, IMAD.IADD → `add`, IMAD.X with b=1 → `addc`, LOP3 → `and/or/xor/not/mux/lop3`, `IADD3 RZ, P0, x, y` → carry-only op, PLOP3 const → `mov.pred`.

### 6.2 Predication
SASS is predicated everywhere, so do **not** split blocks per predicated op (NVLift's approach). Use **ψ-SSA** (Stoutchinin & de Ferrière), as CuLifter does:
- `@P op d, ...` becomes `d' = ψ(P: op(...), d_old)`, a guarded definition that reads the old value as the fall-through. Predicated stores, atomics and EXIT keep their guard as an operand, since they have no value to merge.
- **Grouping:** runs of ops guarded by `P` / `!P` on one value become a single `selp` (`SEL`) or a recovered `if/else` region. The pairs `@P0 IMAD R0 ; @!P0 IADD3 R2 ; @P0 LOP3 R0 ; @!P0 IMAD.SHL R0` in p_cf1 recover the original if/else bracketed by `BSSY B0`.
- **Lowering ψ to PTX** is trivial: PTX allows `@%p` on any instruction. Keep the guard (`@%p add.u32 %r, ...`) and emit `selp` only where the merge is explicit. This is the main reason PTX rather than C is the right first target.
- Predicate algebra: normalise polarity (ISETP inverted compares, `!P` uses), fold PLOP3/combining `bop Pacc` into boolean expressions, and dissolve P2R/R2P spills into SSA booleans.

### 6.3 Control flow and warp semantics
- **CFG:** blocks end at BRA, BRX, CALL, RET, EXIT, BPT and predicated branches. BRX targets come from the c[0x2] jump table, cross-checked with nvdisasm's `BRANCH_TARGETS` comment. Functions are discovered from CALL targets. Internal helpers (`$__internal_*_$__cuda_sm20_div_u64`, `..._sqrt_rn_f32_slowpath`, `..._shflsync_down`, ...) are recognised by name or hash and **replaced by their PTX op** (`div.u64`, `sqrt.rn.f32`, `shfl.sync`), not lifted.
- **Thread-level semantics with convergence annotations.** SIR models a single thread, like PTX. Warp behaviour is expressed by:
  - (a) **convergence regions** from BSSY Bn / BSYNC Bn pairs, kept as `region.begin(Bn)/region.end(Bn)` markers. They give structured if/loop recovery and the places where warp collectives are legal, and they are dropped when printing PTX (ptxas recreates them).
  - (b) **warp collective ops** as first-class SIR ops with an explicit `membermask` operand: `shfl.sync{idx,up,down,bfly}(val, lane, clamp, mask) → (val, pred)`, `vote.sync{any,all,uni,ballot}`, `match.any/all`, `redux.sync`, `activemask`, `bar.warp.sync`.
- SASS SHFL has **no mask**, so the lifter must recover it, defaulting to `0xffffffff` when the shfl sits outside any divergent region (true for all our probes) and otherwise reading the WARPSYNC operand of the slow path. BRA.CONV fast/slow pairs collapse into one `shfl.sync`.
- **CTA-level:** `bar.sync id, cnt`, `bar.arrive`, `bar.red.popc` (BAR.RED + B2R) and memory fences (MEMBAR/ERRBAR/CCTL patterns → `fence.{sc,acq_rel}.{cta,gpu,sys}`, `ld.acquire`, `st.release`).
- **Uniform registers** (UR/UP) are a scheduling artefact: values computed once per warp. In SIR they are ordinary SSA values with a `uniform` attribute (useful for analysis), and they print as normal PTX registers.
- **Structuring for CUDA C** later uses the region tree from BSSY/BSYNC plus standard structural analysis (loops from back-edges, reversing ptxas's ×4 unroll optional).

### 6.4 Memory spaces
- Every memory op carries `space ∈ {global, shared, local, const, param, generic}` from the opcode (LDG/STG/ATOMG/RED → global, LDS/STS/ATOMS/LDSM → shared, LDL/STL → local, LDC/c[] → const, LD/ST/ATOM → generic, LDGSTS → global→shared). It also carries `sem ∈ {weak, relaxed(.STRONG), volatile(.STRONG.SYS), acquire, release}`, `scope ∈ {cta, gpu, sys}`, `cache ∈ {ca, cg(BYPASS / .STRONG.GPU), nc(.CONSTANT)}`, and width/vector shape.
- **Constant banks are typed**:
  - `c[0x0]` = driver/launch params: map offsets to `%ntid`, `%nctaid`, `%gridid`, etc. (§3.7 table), and `0x160+` to **`ld.param` of kernel parameter k**, using the param layout from `.nv.info` `EIATTR_KPARAM_INFO` (see doc 02).
  - `c[0x2]` = BRX jump tables.
  - `c[0x3]` = module `.const` data.
  - `c[0x4]` = relocated addresses of global symbols.
- **Address expressions** are lifted to `base + index*scale + imm` (from IMAD.WIDE / LEA / LEA.HI.X chains) so they print as `mad.wide` / `add.s64` and pointer types can be recovered.
- **Shared and local** addresses are 32-bit offsets. Shared offsets map to `.shared` symbols via `.nv.shared.<kernel>` section size and symbols. Local accesses relative to R1 become `.local` frame slots after recovering `IADD3 R1, R1, -size`.

### 6.5 Types
SASS is untyped bits. Run constraint-based inference (as NVLift and CuLifter do): opcode constraints (`FFMA` → f32, `DADD` → f64 pair, `HFMA2` → f16x2, `IMAD.WIDE` → s64/u64 result, `.U32` compares → unsigned), memory widths and `.S16/.U8`, and param types from `EIATTR_KPARAM_INFO`. The PTX output falls back to `.b32/.b64` whenever a value is ambiguous. That is always legal PTX, so types are a readability feature and not a correctness requirement.

### 6.6 Validating the bridge
1. **Per-op semantic oracle (offline):** port NAK `fold()` to Python, then for each opcode compile tiny PTX probes, run *our* L0 evaluator, and compare with **NAK fold + PTX ISA semantics** on random inputs. Without a GPU, cross-check through two independent routes, PTX semantics and NAK semantics. Ops with no NAK fold (IMAD.HI with c, P2R merge, LOP3 Pin) are marked [inf] in §4 until checked on hardware via NVBit or NAK `hw_tests`.
2. **Round-trip test (ptxas as oracle):** SASS → SIR → PTX′ → `ptxas` → SASS′. Compare SASS′ against SASS **modulo register renaming and scheduling** (canonicalise by dataflow graph isomorphism). Exact equality is the strongest bridge evidence available without a GPU, and it is achievable for most of the idioms in §3, since ptxas is deterministic.
3. **Line-map test:** on -lineinfo cubins, score the lifter's SASS → PTX-line map against `.nv_debug_line_sass` (§5.4).
4. Corpus generator rules learned here: always use lane-varying inputs (tid-mixed) for warp/atomic probes, keep results live with stores, and avoid constants that ptxas folds.

### 6.7 Open questions
- How ptxas allocates BSSY B-registers and when BMOV/BREAK appear (deep nesting, recursion, break-out-of-loop with warp ops). This needs bigger probes.
- `nvdisasm -cfg` hyperblocks vs `-bbcfg` when there are many predicated exits. Fine for comparison, but needs a parser test.
- sm_90+ changes. The param base moves to **0x210 on sm_90** (descriptor at c[0x0][0x208]) and **0x380 on sm_100** (descriptor at 0x358), [V-live] with probes/smoke.ptx retargeted, which matches JEB. The cbank0 map in §3.7 is therefore per-arch spec data. New ops to handle: VIADD, ELECT, SYNCS, UTMA*, HGMMA, plus Hopper cluster/mbarrier semantics. (VIADD, ELECT, SYNCS, UTMA*, HGMMA), and Hopper cluster/mbarrier semantics.
