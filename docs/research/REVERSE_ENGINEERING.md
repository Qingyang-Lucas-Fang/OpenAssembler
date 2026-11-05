# SASS Reverse Engineering and Compiler Evidence

OpenAssembler's low-level work treats NVIDIA's `ptxas`, `nvdisasm` and
`cuobjdump` as black-box measurement oracles. The goal is a defensible SM80
instruction encoder and compiler, not a claim that undocumented encodings are
stable across architectures or toolkits.

## Evidence levels

1. **Round-tripped and disassembled:** generated words are re-read from the
   cubin and `nvdisasm` text agrees with the intended instruction and operands.
2. **Hardware checked:** a cubin executes on an identified GPU with a stated
   output oracle, input precision, checked element count and canaries.
3. **Source or paper inspection:** a public implementation or paper suggests a
   design, but OpenAssembler has not reproduced its behavior or performance.
4. **Hypothesis:** feasibility has not been measured. Do not present it as a
   result.

Instruction counts, minimum encoded stall gaps and successful assembly alone
are not performance measurements. A performance claim requires a paired GPU
comparison on the same worker, matched arithmetic/output contracts, full
correctness checks before timing, shuffled order and retained samples.

## Verified implementation findings

- Volta+ instructions are 128-bit little-endian words. On the examined SM80
  corpus, common register, predicate, immediate, memory and control fields have
  been recovered by corpus variation, bit flips and `nvdisasm` round trips.
- Schedule bits 105–125 are decoded and emitted by the encoder: stall, active-
  low yield, read/write scoreboards, wait masks and operand reuse hints. Some
  bits are opcode-specific and must not be interpreted as generic controls.
- SM80 `LDGSTS` and `DEPBAR.LE SB0` fields were recovered and round-tripped.
  The production tensor path uses the measured 16-byte `cp.async` family and
  reserves SB0 for committed asynchronous-copy groups.
- Cubins are ELF64 `EM_CUDA` containers with NVIDIA-specific metadata. The
  compiler uses a ptxas-built container and replaces selected `.text` bodies;
  it does not reconstruct every possible CUDA binary or ABI feature.
- Constant unsigned/signed integer division and remainder sequences are
  strength-reduced and formally checked by the repository's `divconst` proof
  path. The improvement is for those scalar operations, not a tensor-core
  throughput result.
- The tensor backend preserves full register-group dependencies for HMMA and
  LDSM, asynchronous-copy completion and address WAR edges. Unsupported forms
  are rejected rather than guessed.

## Detailed notes

- [Encoding prior art and bit-field discovery](01-encoding-prior-art.md)
- [Control fields, cubin ELF, fatbins and parameter banks](02-control-bits-and-containers.md)
- [SASS lifting and PTX bridge research](03-lifting-and-ptx-bridge.md)
- [Measured SM80 encoding findings](../findings-sm80-encoding.md)
- [Constant division/remainder findings](../findings-div-const.md)
- [Scalar A100 and emulator findings](SCALAR_COMPILER_RESULTS.md)

## Important limits

The above is strongest for the instruction subset emitted by the current SM80
backend and the tested CUDA 12.8 oracle. It does not establish compatibility
with future NVIDIA architectures, every PTX instruction, every modifier, every
special floating-point input, or undocumented scheduling semantics. H100,
Blackwell and later devices require separate corpora, encoders, synchronization
models and hardware validation.

Reconstructing a public library's SASS at runtime or copying a cubin is not the
compiler strategy used here. The research observes vendor dispatch and
disassembly where legal and available, records what is actually measured, and
implements independently generated kernels. See the A100 results ledger for
the cuBLAS/cuBLASLt observations and comparison limits.
