# 02 - Control bits and containers (cubin ELF, fatbin, cbank0)

Tags: [V-live] verified against CUDA 12.8 tools/bytes in this repo; [cited] from a source; [R] recalled, unverified.

## A. Volta+ control bits

Each Volta+ instruction is 128 bits. nvdisasm prints it as two u64 words: word0 holds bits 0-63 (printed first) and word1 holds bits 64-127. In the file both are little-endian. Bit `n >= 64` is `(word1 >> (n-64)) & 1`.

### A.1 Field layout (sm_70 .. sm_120)

| Bits | Width | Field | Encoding | Status |
|---|---|---|---|---|
| 105-108 | 4 | stall / delay | cycles 0-15 before the next instruction of this warp may issue. Padding NOPs and the self-loop `BRA` use 0. | [V-live] [cited NAK `set_field(105..109, deps.delay)`] |
| 109 | 1 | yield, **active-low** | 0 = yield (nvdisasm sm_120: `?WAITn_END_GROUP`; CuAssembler prints `Y`). 1 = no yield / keep issuing (nvdisasm sm_120: `?transn`; CuAssembler prints `-`). | [V-live] polarity from sm_120 annotations; [cited] CuAssembler `isYield(): Yield == 0` |
| 110-112 | 3 | write barrier (SB index set when the result is written) | 0-5; 7 = none | [V-live] [cited NAK `wr_bar().unwrap_or(7)`] |
| 113-115 | 3 | read barrier (SB index released when the sources have been read) | 0-5; 7 = none | [V-live] (always 7 in our probes) [cited NAK] |
| 116-121 | 6 | wait mask: bit i = wait for SB i to reach 0 before issue | bitmask | [V-live] [cited NAK `116..122 wt_bar_mask`] |
| 122-125 | 4 | operand reuse cache: bit0 = src A (Ra), bit1 = src B, bit2 = src C, bit3 = 4th slot | bitmask | [V-live] `R5.reuse` as 3rd operand of FFMA gives reuse=0b0100; `VIADD R9, R0.reuse` gives 0b0001 |
| 126-127 | 2 | not control; always 0 in our corpus | - | [V-live] |

Compact form (CuAssembler, maxas lineage): the 21-bit control word = `(word1 >> 41)`. Within it: stall = bits 0-3, yield = bit 4, wbar = bits 5-7, rbar = bits 8-10, wait = bits 11-16, reuse = bits 17-20 [cited]. CuAssembler stopped splitting out reuse because some ops (HMMA.SP, TEXS, BRA) reuse those bits for opcode fields [cited CuControlCode.py]. Treat 122-125 as "reuse unless the opcode table says otherwise."

Source: https://github.com/cloudcores/CuAssembler/blob/master/CuAsm/CuControlCode.py (fetched via curl). Background: maxas Control-Codes wiki; Jia et al. arXiv:1804.06826 / 1903.07486.

### A.2 Live verification on corpus/sm_80/smoke.cubin [V-live]

The original control-word decoder read `nvdisasm -hex -c` output.

```
addr stall y wbar rbar wait   reuse  insn
0000   2   1  7    7  000000  0000  MOV R1, c[0x0][0x28]
0010   4   1  0    7  000000  0000  S2R R4, SR_CTAID.X      <- sets SB0
0020   2   1  0    7  000000  0000  S2R R3, SR_TID.X        <- sets SB0 (counter now 2)
0030   5   0  7    7  000001  0000  IMAD R4, R4, c[0x0][0x0], R3   <- waits SB0
0040  13   0  7    7  000000  0000  ISETP.GE.AND P0, PT, R4, c[0x0][0x160], PT
0050   5   1  7    7  000000  0000  @P0 EXIT
0070   9   0  7    7  000000  0000  ULDC.64 UR4, c[0x0][0x118]
00a0   4   1  2    7  000000  0000  LDG.E R2, [R2.64]       <- sets SB2
00b0   2   1  2    7  000000  0000  LDG.E R7, [R4.64]       <- sets SB2
00c0   5   0  7    7  000100  0000  FFMA R7, R2, c[0x0][0x164], R7 <- waits SB2
00f0   0   0  7    7  000000  0000  BRA `(.L_x_0)           (self-loop pad)
```

Example raw word: IMAD @0x30, word1 = `0x001fca00078e0203`. `word1>>41` = `0x00fe5` gives stall=5, y=0, wbar=7, rbar=7, wait=0b000001. The scoreboards are counters: two producers can share an SB, and the consumer waits for it to drain to 0. ptxas packs independent long-latency ops onto one SB (in the 12-load probe, loads 5-12 all use SB5).

### A.3 DEPBAR / LDGDEPBAR [V-live]

The original CUDA 12.8 probe used `cp.async` + `commit_group` + `wait_group N`:

| insn | word0 | word1 | notes |
|---|---|---|---|
| `LDGDEPBAR` | `0x00000000000079af` | `0x000e240000000000` | opcode 0x9af. Ctrl sets wbar=SB0: it increments SB0 once per commit group. |
| `DEPBAR.LE SB0, 0x1` | `0x000080400000791a` | `0x000fc80000000000` | opcode 0x91a. Waits until SB0 <= 1. |
| `DEPBAR.LE SB0, 0x0` | `0x000080000000791a` | `0x000fc80000000000` | count field at bits 38+ (0x1 gives `0x40<<32`). Bit 47 (`0x8000<<32`) = `.LE`. The SB index (0) sits at bits 44-46 [R: exact SB-index bits not yet probed with SB!=0]. |

DEPBAR is the counted alternative to the wait mask: the 6-bit mask only waits for "== 0", while `DEPBAR.LE SBk, n` waits for "<= n". That is what `cp.async.wait_group n` needs. The same encoding was observed on sm_80, sm_90, and sm_120 (identical words). The sampled NAK sm70 source had no DEPBAR (only `OpTexDepBar` for Kepler); see its public source tree in the references.

### A.4 Cross-architecture check [V-live]

Compiled `probes/smoke.ptx` plus two probes for sm_75, 86, 90, 100, and 120 with ptxas 12.8:

- Bits 105-125 have identical meaning on all of them. S2R/LDC/LDG/S2UR/LDCU set wbar, and consumers wait via the mask. Padding NOP/BRA have stall=0.
- Reuse is still emitted on sm_120: `FFMA R5, R5, UR4, R17.reuse` gives reuse=0b0100. NAK skips bits 109 and 122-125 for `sm >= 120` (`if self.sm < 120 {...}` in `set_instr_deps`), but ptxas sets both on sm_120. NAK's choice is conservative, not an encoding change.
- sm_90+: `LDC` (constant load to GPR) is a variable-latency op with a write barrier, unlike sm_80's direct `c[][]` operands. This matters for scheduling analysis, not field layout.
- nvdisasm 12.8 annotates schedule info only for sm_120 (`&wr=0xN`, `&req={i,..}`, `?transN` / `?WAITN_END_GROUP`). This gives an NVIDIA-sourced decoding of our fields: `&wr` = bits 110-112, `&req` = bits 116-121, `N` = bits 105-108, `trans`/`END_GROUP` = bit 109 = 1/0. Not printed for sm_80/90/100.

Yield semantics caveat: NAK sets its `yld` flag (raw bit 109 = 1) on BAR/BSSY/BSYNC/NANOSLEEP. ptxas also emits bit 109 = 1 on BAR.SYNC and NANOSLEEP. So NAK's name `yld` refers to the raw bit=1 state, which CuAssembler and nvdisasm call "not yield"/"trans". The raw polarity is settled. The naming (and which hardware behaviour 1 vs 0 triggers) is contested. Record the raw bit and both names. ptxas pattern: bit=0 on dependent fixed-latency ALU chains (stall >= 4, "end group"); bit=1 on back-to-back independent issue and on memory ops.


## B. cubin ELF

The original probes parsed CUDA ELF headers, sections, `.nv.info` TLVs, symbols and relocations. Test cubins covered a one-kernel image and linked/relocatable modules with device functions, extern calls, globals, constant memory, static/dynamic shared memory, local arrays, `.maxntid` and `.reqntid`.

### B.1 ELF header [V-live]

Little-endian ELFCLASS64. `e_machine = 190` (EM_CUDA) on every arch.

| Field | sm_50..sm_90 (ABI v7) | sm_100 / sm_120 (ABI v8, new) |
|---|---|---|
| EI_OSABI | 0x33 (ELFOSABI_CUDA) | **0x41** |
| EI_ABIVERSION | 7 | **8** |
| e_type | 2 ET_EXEC (`ptxas`), 1 ET_REL (`ptxas -c`) | same |
| e_version | 128 = toolkit 12.8 (cuobjdump: `toolkit=128`) | **1**. Toolkit version moves to `.note.nv.tkinfo`. |
| e_flags (smoke) | sm_80 `0x00500550` | sm_100 `0x06006402`, sm_100a `0x0600640a`, sm_120 `0x06007802` |
| phdrs | yes in ET_EXEC (PT_PHDR + PT_LOAD per segment), none in ET_REL | same |

e_flags, ABI v7. Decoded by flipping each bit of smoke.cubin and reading nvdisasm `.headerflags` [V-live]:

| Bits | Mask | Meaning |
|---|---|---|
| 0-7 | `0xff` | EF_CUDA_SM: real SM number (0x50 = 80, 0x5a = 90, 0x4b = 75) |
| 8 | `0x100` | EF_CUDA_TEXMODE_UNIFIED |
| 9 | `0x200` | EF_CUDA_TEXMODE_INDEPENDENT |
| 10 | `0x400` | EF_CUDA_64BIT_ADDRESS |
| 11 | `0x800` | EF_CUDA_ACCELERATORS (the "a" suffix: sm_90a gives `0x005a0d5a`). On sm_80, nvdisasm names this bit EF_CUDA_SW_1729687, so the name is arch-dependent. |
| 12 | `0x1000` | EF_CUDA_SW_1729687_v2 |
| 13-15 | - | unassigned (nvdisasm prints "unrecognized") |
| 16-23 | `0xff0000` | EF_CUDA_VIRTUAL_SM: PTX `.target` SM number. Verified with `.target sm_75` compiled `-arch=sm_80`, giving `0x004b0550` and `EF_CUDA_VIRTUAL_SM(EF_CUDA_SM75)`. |
| 24-31 | - | unassigned |

So sm_80 smoke `0x00500550` = SM80 | TEXMODE_UNIFIED | 64BIT | VIRTUAL_SM(80).

e_flags, ABI v8 (sm_100+): nvdisasm 12.8 no longer prints `.headerflags`, only `.target`. Bit flips [V-live]:

| Bits | Meaning |
|---|---|
| 8-15 | real SM number (0x64 = 100, 0x78 = 120). Flipping bit 8 gives `sm_101`; others give "Cannot decode architecture SM102/96/...". |
| 3 | `a` suffix (`0x...0a` gives `.target sm_100a`) |
| 1 | always set (`0x2`); clearing bits 0 or 9-23 makes the file "not supported" [function unknown, R: probably 64-bit/address-model] |
| 24-31 | `0x06` in all our builds; flips accepted silently [unknown] |

The virtual SM is no longer in e_flags. It lives in `.note.nv.cuver` (`Cuda Virtual SM: sm_100`). PTX `.target sm_75` and `.target sm_90` compiled for sm_100 both give `0x06006402`.

### B.2 Sections and roles [V-live]

| Name | sh_type (ET_REL / ET_EXEC) | Role |
|---|---|---|
| `.text.<k>` | PROGBITS | SASS for kernel/function `<k>`. flags: ALLOC\|EXEC (0x6), plus bits 20-26 = barrier count (`0x100006` = SHF_BARRIERS=1 when `bar.sync` is used). `sh_link` = .symtab. `sh_info` on <= sm_89: `(regcount << 24) \| symidx` (`0x0a000006` = 10 regs, sym 6; nvdisasm `SHI_REGISTERS=10`). On sm_90+: `sh_info` = symidx only, and regcount lives only in EIATTR_REGCOUNT. Align 128. |
| `.nv.info` | 0x70000000 SHT_CUDA_INFO | global attribute table (per-function REGCOUNT / FRAME_SIZE / stack sizes, keyed by sym idx) |
| `.nv.info.<k>` | 0x70000000, flags 0x40 (INFO_LINK), `sh_info` = idx of `.text.<k>` | per-kernel attribute table |
| `.nv.constant0.<k>` | 0x70000064 SHT_CUDA_CONSTANT_B0 (ET_REL), PROGBITS (ET_EXEC); flags 0x42; `sh_info` = .text idx | cbank 0 image: driver area + params. Size = param base + param size (sm_80 `0x160 + 0x18 = 0x178`; sm_90 `0x210 + 0x18 = 0x228`; sm_100 `0x380 + 0x18 = 0x398`). |
| `.nv.constant3` | 0x70000067 _B3 / PROGBITS | user `.const` data (`ctab`); accessed as `c[0x3][...]` |
| `.nv.constant4` | PROGBITS (exec) | per-module pointer table of global variable addresses, filled by the loader via `.rel(a).nv.constant4` R_CUDA_64 (sm_80/90 exec) |
| `.nv.global` | 0x70000007 SHT_CUDA_GLOBAL (rel) / NOBITS (exec) | zero-init `.global` data (`gcount`) |
| `.nv.global.init` | 0x70000008 SHT_CUDA_GLOBAL_INIT / PROGBITS | initialized globals (`gptr`) |
| `.nv.shared.<k>` | NOBITS, flags 0x43, `sh_info` = .text idx (exec only) | static shared size of kernel `<k>` (sm_80 kern: 0x400). In ET_REL shared objects sit in `.nv_debug.shared` (0x7000000a) with STO_CUDA_SHARED symbols. |
| `.nv.shared.reserved.0` | NOBITS | sm_90+: reserved SMEM. sm_100+: size 0x40. Symbols `.nv.reservedSmem.offset0` / `__nv_reservedSMEM_offset_0_alias`. |
| `.nv.callgraph` | 0x70000001 SHT_CUDA_CALLGRAPH, entsize 8 | pairs `(u32 caller_sym, u32 callee_sym)`. `<0,-1>..<0,-4>` (callee = 0xffffffff..0xfffffffc) are fixed sentinel entries in every file. `<22,1>` = kern calls helper; `<22,23>` = kern calls ext_fn. |
| `.nv.prototype` | 0x70000002 SHT_CUDA_PROTOTYPE | prototypes for extern calls (`<23,1(#ii)>`), ET_REL |
| `.nv.rel.action` | 0x7000000b SHT_CUDA_RELOCINFO | describes how the driver applies a relocation type, as bit-slices. smoke: `73000000 00000000 00000011 25000536` = R_CUDA_CONST_FIELD22_37: kind ADDR, value bits [0:17) to insn bit 37, next 5 bits to insn bit 54 (cuobjdump: `EIVALUE_SYM_KIND_ADDR,0,0,17,37,0,5,54`). |
| `.rel.<s>` / `.rela.<s>` | REL (9) / RELA (4) | relocations against section `<s>` (`sh_info` = target). sm_80 uses mostly REL; sm_90+ uses RELA. |
| `.debug_frame` | PROGBITS | DWARF CFI. Always present. Return-address register R20; CFA = R1 (stack pointer). |
| `.note.nv.tkinfo` | NOTE, owner "NVIDIA Corp", type 2000 | sm_100+: toolkit version 128, tool name, version string, command line (`-arch sm_100`) |
| `.note.nv.cuver` | NOTE, type 1000 | sm_100+: desc `{u16 ver=1, u16 vsm=100, u32 1, u32 1}` gives virtual SM |
| `.nv.compat` | 0x70000086 SHT_CUDA_COMPAT_INFO | sm_100+: BVAL TLVs EICOMPAT_ATTR_ISA_CLASS(2)=1, INST_TENSORMAP_V1(3)=0, INST_TCGEN05_MMA(5)=5, ENABLE_OPPORTUNISTIC_FINALIZATION(6)=1 |
| `.nv.capmerc.text.<k>`, `.nv.merc.*` | 0x70000016 / 0x70000082..85, flags 0x10000000 | sm_100+ "Mercury" capsule: a second, compressed or intermediate copy of the code plus its own symtab/nv.info/rela, used for driver-side finalization. Opaque; **skip for disassembly** [V-live existence; content R]. |

### B.3 `.nv.info` TLV format [V-live]

Each record starts with `u8 format, u8 attribute` and is 4-byte aligned:

| format | name | layout |
|---|---|---|
| 0x01 | EIFMT_NVAL | `fmt attr 00 00`: flag only |
| 0x02 | EIFMT_BVAL | `fmt attr u8 val 00` |
| 0x03 | EIFMT_HVAL | `fmt attr u16 val` |
| 0x04 | EIFMT_SVAL | `fmt attr u16 size` + `size` bytes payload |

smoke.cubin `.nv.info` raw: `042f0800 06000000 0a000000 | 04110800 06000000 00000000 | 04120800 06000000 00000000` = REGCOUNT(sym 6, 10), FRAME_SIZE(sym 6, 0), MIN_STACK_SIZE(sym 6, 0). This matches cuobjdump exactly.

EIATTR numbers: the cuobjdump 12.8 name table, ordered so that string index = code. Confirmed on every code we observed (marked V). The table ends at 0x59; ptxas has 4 more names (COROUTINE_RESUME_OFFSETS, AT_ENTRY_FRAGEMENTS[sic], SANITIZE, ERROR_LAST) whose codes are not confirmed.

| code | name | fmt / payload | seen |
|---|---|---|---|
| 0x04 | CTAIDZ_USED | NVAL | |
| 0x05 | MAX_THREADS | SVAL 3x u32 (`.maxntid`) | V `80 1 1` |
| 0x0a | PARAM_CBANK | SVAL `u32 sym(.nv.constant0.k section sym), u16 param_off, u16 param_size` | V `0x2 0x180160` = off 0x160, size 0x18 |
| 0x0f | EXTERNS | SVAL u32 sym[] | V |
| 0x10 | REQNTID | SVAL 3x u32 | V `40 1 1` |
| 0x11 | FRAME_SIZE | SVAL `u32 sym, u32 bytes` | V (helper 0x40 local array) |
| 0x12 | MIN_STACK_SIZE | SVAL `u32 sym, u32` | V |
| 0x17 | KPARAM_INFO | SVAL 12 B: `u32 index, u16 ordinal, u16 offset, u32 flags`. In flags: bits 12-16 = cbank (0x1f), bits 18-31 = size in bytes (`0x0021f000` = 8, `0x0011f000` = 4). Low bits = pointee logAlign/space [inferred]. One record per param, emitted in reverse ordinal order. | V |
| 0x19 | CBANK_PARAM_SIZE | HVAL | V 0x18 |
| 0x1b | MAXREG_COUNT | HVAL | V 0xff |
| 0x1c | EXIT_INSTR_OFFSETS | SVAL u32[] (byte offsets of EXIT) | V `0x50 0xe0` = smoke EXITs |
| 0x1d | S2RCTAID_INSTR_OFFSETS | SVAL u32[] | |
| 0x1e | CRS_STACK_SIZE | | |
| 0x23 | MAX_STACK_SIZE | SVAL `u32 sym, u32` | V |
| 0x27 | ATOM_SYS_INSTR_OFFSETS | SVAL u32[] | |
| 0x28 | COOP_GROUP_INSTR_OFFSETS | | |
| 0x2b | WMMA_USED | | |
| 0x2f | REGCOUNT | SVAL `u32 sym, u32 regs` | V |
| 0x31 | INT_WARP_WIDE_INSTR_OFFSETS | SVAL u32[] (offsets of the BAR/WARPSYNC-class insns + CALL sites) | V 8 entries |
| 0x34 | INDIRECT_BRANCH_TARGETS | | |
| 0x35 | SW2861232_WAR | NVAL | V (sm_80 every kernel) |
| 0x36 | SW_WAR | SVAL u32 | V `8` (sm_90+) |
| 0x37 | CUDA_API_VERSION | SVAL u32 (0x80 = 12.8) | V |
| 0x38 / 0x39 | NUM_MBARRIERS / MBARRIER_INSTR_OFFSETS | | |
| 0x3d / 0x3e / 0x3f | CTA_PER_CLUSTER / EXPLICIT_CLUSTER / MAX_CLUSTER_RANK | | |
| 0x41 / 0x42 | RESERVED_SMEM_USED / RESERVED_SMEM_0_SIZE | | |
| 0x45 | KPARAM_INFO_V2 | | |
| 0x4a | VRC_CTA_INIT_COUNT | BVAL | V (sm_100) |
| 0x4c | NUM_BARRIERS | | |
| 0x50 | SPARSE_MMA_MASK | HVAL | V (sm_90+) |
| 0x51 / 0x52 | TCGEN05_1CTA_USED / TCGEN05_2CTA_USED | | |
| 0x54 | REG_RECONFIG | | |
| 0x5a | (unnamed in 12.8 cuobjdump; SVAL 36 B, appears in `.nv.merc.nv.info.saxpy`) | | V raw |

The original CUDA 12.8 cuobjdump survey covered EIATTR codes 0x00-0x59. It prints record numbers in decimal despite the `0x` prefix, so `<0x10>` is the tenth record, not code 0x10.

### B.4 Relocations [V-live]

The R_CUDA type number is the index into the CUDA 12.8 `R_CUDA_*` string table in nvlink/ptxas/cuobjdump (119 names). This agrees with CuAssembler's public hard-coded values (1, 2, 4, 6, 42-44, 52, 56-58) [cited].

| type | name | observed use (sm_80 ET_REL) |
|---|---|---|
| 1 | R_CUDA_32 | |
| 2 | R_CUDA_64 | `.debug_frame` FDE addresses; `.nv.constant4` pointer table (exec) |
| 4 | R_CUDA_G64 | `.nv.global.init`: `gptr = generic(gcount)` |
| 56 / 57 | R_CUDA_ABS32_LO_32 / ABS32_HI_32 | `UMOV UR4, 32@lo(gcount)` / `32@hi`; also `MOV R20, 32@lo((kern + .L_x_0@srel))` return-address setup before CALL (RELA, addend = label offset) |
| 58 | R_CUDA_ABS47_34 | `CALL.ABS.NOINC \`(helper)` target |
| 59 | R_CUDA_ABS16_32 | `UMOV UR4, \`(ctab)` (const bank 3 offset) |
| 73 | R_CUDA_UNUSED_CLEAR64 | `.rela.debug_frame` |
| 74 | R_CUDA_ABS24_40 | `STS [R2.X4+\`(sbuf)]`: shared-variable offset in the 24-bit imm at bit 40 |
| 115 (0x73) | R_CUDA_CONST_FIELD22_37 | `.nv.rel.action` descriptor |
| 68 / 69 | R_CUDA_YIELD_OPCODE9_0 / YIELD_CLEAR_PRED4_87 | not seen. Names suggest the linker can patch a 9-bit opcode at bit 0 and a predicate at bit 87 (the guard) [R]. |

The naming convention `R_CUDA_<kind><width>_<bitpos>` gives the field width and its starting bit in the 128-bit instruction (ABS47_34 = 47 bits at bit 34; ABS24_40; ABS32_LO_32). That convention is a free source of operand-field positions. After linking (`ptxas` ET_EXEC with no externs), the text relocations are resolved and disappear. Only `.rel(a).nv.constant4`, `.nv.global.init`, and `.debug_frame` remain for the driver.

### B.5 Symbols and st_other [V-live]

`st_other` = `STO_CUDA_*` in the high bits plus ELF visibility in bits 0-1. Values seen, and their nvdisasm `.other` strings:

| st_other | meaning | example |
|---|---|---|
| 0x10 | STO_CUDA_ENTRY \| STV_DEFAULT | `saxpy`, `kern` (st_info 0x12 = GLOBAL FUNC) |
| 0x20 | STO_CUDA_GLOBAL | `gcount`, `gptr` (ET_REL, st_info 0xd = LOCAL, type 13 = STT_CUDA_OBJECT) |
| 0x40 | STO_CUDA_SHARED | `sbuf`, `dyn` (`dyn` is undefined, st_info 0x1d, extern dynamic shared) |
| 0x80 | STO_CUDA_CONSTANT | `ctab` |
| 0x81 | STO_CUDA_CONSTANT \| STV_INTERNAL | `_param` (value 0x160 = param base, size 0xc), `_SREG` (size 0x160 = driver area) |
| 0xa0 | (0x80\|0x20) | `__nv_reservedSMEM_offset_0_alias` (sm_90) [unnamed] |
| 0x00 | non-entry device functions; in ET_EXEC STO bits are cleared on data symbols | `helper` (LOCAL FUNC); after linking it's renamed `$kern$helper`, inlined into `.text.kern` |

Section symbols (STT_SECTION, st_info 0x3) exist for every section that relocations or `.nv.info` reference. `.symtab` `sh_info` = index of first global symbol.


## C. fatbin

Built live with CUDA 12.8 `fatbinary -64` from ELF and PTX images. Header names below follow the old public [Gklee `fatbinary.h`](https://github.com/Geof23/Gklee/blob/master/Gklee/include/cuda/fatbinary.h); every offset was checked on the resulting files [V-live].

### C.1 Container header (16 bytes, align 8)

| off | type | field | value |
|---|---|---|---|
| 0x00 | u32 | magic | `0xBA55ED50` (bytes `50 ed 55 ba`). Old pre-CUDA-4 magic `0x1EE55A01` [cited]. |
| 0x04 | u16 | version | 1 |
| 0x06 | u16 | headerSize | 0x10 |
| 0x08 | u64 | fatSize | bytes of all entries after the header (file = 0x10 + fatSize) |

Entries follow back to back. Walk `o = 0x10; while o < 0x10+fatSize: o += entry.headerSize + entry.payloadSize`. A host binary can concatenate several containers in `.nv_fatbin`, so loop on the magic.

### C.2 Entry header (0x40 base; 0x48/0x50/0x60 with optional strings)

| off | type | field | observed |
|---|---|---|---|
| 0x00 | u16 | kind | 1 = PTX, 2 = ELF (cubin), 4 = OLDCUBIN [cited] |
| 0x02 | u16 | (version/attr) | always 0x0101 |
| 0x04 | u32 | headerSize | 0x40 (ELF), 0x50 (PTX, has options descriptor), 0x60 (sm_100 ELF with compat blob, or with `--ident`) |
| 0x08 | u64 | payloadSize | padded to 8 (PTX 0x2c8 for 0x2c1 bytes) |
| 0x10 | u32 | compressedSize | 0 if uncompressed; else exact compressed length (LZ4 0x3bc, zstd 0x2b6) |
| 0x14 | u32 | optionsOffset | 0 = none; else offset (within the entry header) of a `{u32 off, u32 len}` descriptor for the ptxas options string (`--cmdline "-O3 -lineinfo"` gives `{0x50, 13}`). PTX always has one (len 0 by default). On sm_100 ELF the descriptor points at a 16-byte copy of the `.nv.compat` attributes. |
| 0x18 | u16,u16 | code version minor, major | ELF `[1,7]` (sm_100: `[1,8]` = ELF ABI), PTX `[8,7]` = PTX ISA 8.7 (cuobjdump "code version") |
| 0x1c | u32 | arch | SM number: 80, 90, 100 |
| 0x20 | u32,u32 | identifierOffset, identifierLen | `--ident=myident` gives `{0x40, 7}` + NUL string at +0x40 |
| 0x28 | u64 | flags | see C.3 |
| 0x30 | u64 | (zero) | |
| 0x38 | u64 | uncompressedSize | 0 if uncompressed; PTX includes the trailing NUL (0x2c1) |

PTX payload is the PTX text with comments and indentation stripped (932 B in, 704 B stored), NUL-terminated, padded to 8.

### C.3 Entry flags (u64 at +0x28)

| bit | mask | meaning | evidence |
|---|---|---|---|
| 0 | 0x1 | 64-bit (`-64`) | V (always set) |
| 1 | 0x2 | debug (`-g`) | V: 0x11 becomes 0x13 |
| 2 / 3 | 0x4 / 0x8 | CUDA / OpenCL | cited |
| 4-7 | 0x10 | host OS: 0x10 linux, 0x20 mac, 0x40 windows | V linux; cited others |
| 8-11 | 0xf00 | optimization level | cited |
| 12 | 0x1000 | legacy COMPRESS (pre-11.x, from old header) | cited, not emitted by 12.8 |
| 13 | **0x2000** | **LZ4 compressed** (`--compress-mode=speed`, and the 12.8 default for PTX) | V: payload is a raw LZ4 *block*. A 20-line block decoder reproduces smoke.cubin byte-exact. |
| 15 | **0x8000** | **Zstandard compressed** (`--compress-mode=size` / `balance`) | V: payload starts `28 b5 2f fd`; `zstd -d` reproduces smoke.cubin byte-exact |
| 24 | 0x01000000 | set on the sm_100 ELF entry (with compat blob) | V, meaning inferred: "has compat/options info" or "Blackwell ABI v8" |

`-c` (relocatable) did not change the entry flags (0x11); relocatability is visible only as ET_REL inside the payload. Defaults in 12.8 fatbinary: ELF stored raw and PTX LZ4-compressed (`--compress-all` also compresses ELF). `cuobjdump -lelf/-lptx/-all` lists and decodes entries (`arch`, `code version`, `host`, `compile_size`, `identifier`, `ptxasOptions`).

### C.4 Host-ELF embedding [V-live headers + host compile of `fatbinary --embedded-fatbin` output]

From `env/cuda-12.8/include/fatbinary_section.h` and `crt/host_runtime.h` [cited, local]:

- Fatbin bytes go in section **`.nv_fatbin`** (Mach-O: `__NV_CUDA,__nv_fatbin`) as `fatbinData[]`, align 8. Verified: compiling `e.fatbin.c` with `clang -target x86_64-linux-gnu` yields `.nv_fatbin` (size 0xaf8, starts `50ed55ba`).
- A wrapper `__fatBinC_Wrapper_t { int magic = 0x466243B1 (FATBINC_MAGIC); int version (1 = whole-program, 2 = FATBINC_LINK_VERSION for -rdc); const u64 *data; void *filename_or_fatbins; }` goes in **`.nvFatBinSegment`** (Mach-O `__fatbin`). This wrapper is what the host code passes to `__cudaRegisterFatBinary(&__fatDeviceText)`, which returns `void **handle`. Then `__cudaRegisterFunction(handle, hostStub, deviceName, ...)` / `__cudaRegisterVar` run per kernel/var, then `__cudaRegisterFatBinaryEnd(handle)`. `atexit(__cudaUnregisterBinaryUtil)`.
- `-rdc` / separate compilation: device code goes to **`__nv_relfatbin`** (FATBIN_PRELINK_DATA_SECTION_NAME) for nvlink. Each TU emits `__module_id_str` in section **`__nv_module_id`** (string of `__NV_MODULE_ID`), a wrapper `__fatbinwrap<ID>`, and a call to `__cudaRegisterLinkedBinary<ID>(callback, &__fatbinwrap<ID>, &__module_id_str, dummy)`. The real `__cudaRegisterLinkedBinary_*` functions are generated by nvlink in the host link stub (`a_dlink.reg.c`) [R].
- Extraction recipe: find `.nv_fatbin` in the host ELF (or scan for `50 ed 55 ba` anywhere, which also works for .so/stripped binaries), parse containers, decompress by flag, and dispatch on kind.


## D. Constant bank 0 and special registers

### D.1 cbank 0 layout [V-live unless marked]

Method: one PTX probe per special register, plus `cvta`/`isspacep`, smoke and linked-module probes, for sm_80/90/100 (sm_120 matched sm_100 in the tested case). The driver fills the region below the parameter base (symbol `_SREG`, size = parameter base). Params start at `_param` (EIATTR_PARAM_CBANK offset).

| Content | sm_80 (also sm_75/86) | sm_90 | sm_100 / sm_120 | PTX source |
|---|---|---|---|---|
| ntid.x / .y / .z | 0x0 / 0x4 / 0x8 | 0x0 / 0x4 / 0x8 | 0x360 / 0x364 / 0x368 | `%ntid` |
| nctaid.x / .y / .z | 0xc / 0x10 / 0x14 | 0xc / 0x10 / 0x14 | 0x370 / 0x374 / 0x378 | `%nctaid` |
| shared window base (u64) | 0x18 | via `S2R SR_SWINHI` (no cbank) | via `SR_SWINHI` | `cvta.shared`, `isspacep` |
| local window base (u64) | 0x20 | 0x20 | 0x2f8 | `cvta.local` |
| initial stack pointer (to R1) | **0x28** | **0x28** (`LDC R1`) | **0x37c** | prologue `MOV/LDC R1, c[0x0][..]` |
| dynamic smem size | 0x2c | 0x2c | indexed by CGA: `c[0x0][R*(-0xa0)+0x2ac]` | `%dynamic_smem_size` |
| gridid (u64) | 0x30 | 0x30 | 0x140 | `%gridid` |
| envreg0..N | 0x88 + 4n | 0x40 + 4n | CGA-indexed `0x250 + 4n` | `%envregN` |
| nsmid | 0x10c | 0x10c | 0x2d0 | `%nsmid` |
| reserved smem subtracted from SR_SMEMSZ | 0x114 | 0x114 | 0x16c | `%total_smem_size` |
| aggr smem limit | n/a | 0x13c | CGA-indexed 0x2bc | `%aggr_smem_size` (sm_90+) |
| global memory descriptor (u64, `ULDC.64 URx` then `desc[URx]` / `.E` loads) | **0x118** | **0x208** | **0x358** (`LDCU.64`) | implicit on every LDG/STG/ATOMG |
| **param base** | **0x160** | **0x210** | **0x380** | `ld.param` |
| cbank0 section size (smoke, 0x18 B params) | 0x178 | 0x228 | 0x398 | |

Notes:
- **sm_80 smoke [V-live]**: `MOV R1, c[0x0][0x28]` (stack init), `IMAD R4, R4, c[0x0][0x0], R3` (ctaid*ntid.x+tid), `ISETP ... c[0x0][0x160]` (n = param 0), `FFMA ... c[0x0][0x164]` (a), `IMAD.WIDE ... c[0x0][0x168] / 0x170` (x, y), `ULDC.64 UR4, c[0x0][0x118]` (memory descriptor; on sm_80 nvdisasm hides the `desc[UR4]` operand and prints `LDG.E R2, [R2.64]`, while sm_90 prints `desc[UR4][R2.64]`).
- On sm_90+ cbank operands to ALU ops are rarer. ptxas emits `LDC`/`ULDC`/`LDCU` with scoreboards, so reading `c[0x0][...]` arguments requires following data flow through registers.
- sm_100 multiplies `SR_CgaSize` by -0xa0 to index per-CTA-in-cluster driver records (0xa0-byte stride below a fixed base). Several "per-launch" fields became cluster-indexed.
- Max param space: cbank0 is 64 KiB, and the 12.x param limit is 32764 B [R]. KPARAM_INFO and PARAM_CBANK give the exact layout per kernel, so a decompiler never needs to hardcode the base. Read it from EIATTR_PARAM_CBANK (`u16 offset` = 0x160 / 0x210 / 0x380).

### D.2 Special registers (S2R index = bits 72-79) [V-live encodings; names match NAK `nak_private.h` NAK_SV_* where present]

S2R = opcode 0x919 (GPR dst), S2UR = 0x9c3 (uniform dst), CS2R = 0x805 (bit 80 = .64 pair) [V-live + cited NAK `set_field(72..80, idx)`].

| idx | SASS name | PTX | notes |
|---|---|---|---|
| 0x00 | SR_LANEID | `%laneid` | |
| 0x02 | SR_VIRTCFG | `%nwarpid` = bits 8-15 (sm_90+; sm_80 folds to const 0x40) | |
| 0x03 | SR_VIRTID | `%warpid` = bits 8-14 (`SHF.R 8; SGXT 7`) | |
| 0x0f / 0x10 / 0x11 / 0x13 / 0x1c / 0x1d | ORDERING_TICKET, PRIM_TYPE, INVOCATION_ID, THREAD_KILL, AFFINITY, INVOCATION_INFO | graphics | cited NAK only |
| 0x20 | SR_TID (combined) | | cited NAK |
| 0x21 / 0x22 / 0x23 | SR_TID.X / .Y / .Z | `%tid` | V |
| 0x25 / 0x26 / 0x27 | SR_CTAID.X / .Y / .Z | `%ctaid` | V. sm_90+ often S2UR. |
| 0x32 | SR_SMEMSZ | `%total_smem_size` (minus the reserved c0 field) | V |
| 0x38-0x3c | SR_EQMASK, LTMASK, LEMASK, GTMASK, GEMASK | `%lanemask_*` | V |
| 0x43 | SR_VIRTUALSMID | `%smid` | V |
| 0x50 / 0x51 | SR_CLOCKLO / HI | `%clock` (CS2R.32), `%clock64` (CS2R 64-bit) | V |
| 0x52 | SR_GLOBALTIMERLO | `%globaltimer` (CS2R 64-bit) | V |
| 0x2f | SR_SWINHI | high word of the generic shared window (`cvta.shared` on sm_90+) | V |
| 0x88 | SR_CgaCtaId | CTA rank in cluster; shared address = `LEA(CgaCtaId, 0x400, 24)` on sm_90+ | V |
| 0x8a | SR_CgaSize | sm_100 indexes per-CTA cbank0 records with it (`* -0xa0`) | V |

Constant-folded or cbank-backed PTX sregs (no S2R): `%ntid`, `%nctaid`, `%nsmid`, `%gridid`, `%envreg*`, `%dynamic_smem_size`, and `%nwarpid` on sm_80.

---

## Probe artifacts

Temporary probe binaries, source generators and decoded dumps were removed
after these findings were consolidated. Current encoding provenance, selected
raw JSON and A100 validation files are indexed under `docs/evidence/`.

## Sources

- Mesa NAK: `set_instr_deps`, `InstrDeps`, `calc_instr_deps.rs` and `nak_private.h` in the upstream [Mesa NAK source tree](https://gitlab.freedesktop.org/mesa/mesa/-/tree/main/src/nouveau/compiler/nak). Local copied source snapshots have been removed; the observations above remain comparative source notes, not NVIDIA specifications.
- CuAssembler `CuControlCode.py`, `CuAsmParser.py` (R_CUDA numbers): https://github.com/cloudcores/CuAssembler/blob/master/CuAsm/CuControlCode.py
- maxas control codes: https://github.com/NervanaSystems/maxas/wiki/Control-Codes. Jia et al., "Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking", arXiv:1804.06826.
- SM120 fixed-latency blog (uses `Y` flag notation): https://dev.to/comradepenguin1917/exploring-result-visibility-of-fixed-latency-instructions-on-the-sm120-architecture-5971
- Old fatbinary.h (flags, kinds): https://github.com/Geof23/Gklee/blob/master/Gklee/include/cuda/fatbinary.h
- Local toolkit headers: `env/cuda-12.8/include/fatbinary_section.h`, `include/crt/host_runtime.h`.
- String tables in `env/cuda-12.8/bin/{cuobjdump,nvlink,ptxas}` (EIATTR, R_CUDA, STO_CUDA, EF_CUDA names).
