#!/usr/bin/env bash
# Run an NVIDIA toolkit command (nvcc, ptxas, cuobjdump, nvdisasm, ...) inside
# the colima Linux VM, using the redist toolkit unpacked in env/cuda-12.8.
# The VM mounts $HOME at the same path, so host paths work unchanged.
# Usage: env/cuda.sh nvdisasm -hex corpus/sm_80/foo.cubin
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUDA="${OPENASSEMBLER_CUDA_HOME:-$ROOT/env/cuda-12.8}"

# printf %q quotes each argument so it survives the extra shell hop over ssh.
CMD="$(printf '%q ' "$@")"
exec colima ssh -- bash -c "cd $(printf %q "$PWD") && \
  export PATH=$(printf %q "$CUDA/bin"):\$PATH \
         LD_LIBRARY_PATH=$(printf %q "$CUDA/lib")\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH} && \
  $CMD"
