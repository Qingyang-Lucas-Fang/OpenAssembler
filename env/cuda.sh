#!/usr/bin/env bash
# Run an NVIDIA toolkit command (nvcc, ptxas, cuobjdump, nvdisasm, ...) from the
# CUDA 12.8 redist that env/fetch_cuda.sh unpacks into env/cuda-12.8.
#
#   Linux : runs the tool directly. If env/cuda-12.8 doesn't exist, whatever is on
#           PATH (e.g. a system CUDA install) is used instead.
#   macOS : the toolkit is Linux-only, so the command runs inside a colima Linux VM.
#           colima mounts $HOME at the same path, so host paths work unchanged --
#           which also means the repo (and any file passed to a tool) must live
#           under $HOME.
#
# Usage: env/cuda.sh nvdisasm -hex corpus/sm_80/foo.cubin
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUDA="${OPENASSEMBLER_CUDA_HOME:-$ROOT/env/cuda-12.8}"

case "$(uname -s)" in
  Linux)
    if [ -d "$CUDA/bin" ]; then
      export PATH="$CUDA/bin:$PATH"
      export LD_LIBRARY_PATH="$CUDA/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
    exec "$@"
    ;;
  Darwin)
    if ! command -v colima >/dev/null 2>&1; then
      echo "env/cuda.sh: colima not found. NVIDIA's toolkit only runs on Linux, so on macOS" >&2
      echo "  it runs inside a small VM:  brew install colima && colima start" >&2
      exit 127
    fi
    if ! colima status >/dev/null 2>&1; then
      echo "env/cuda.sh: the colima VM isn't running. Start it with:  colima start" >&2
      exit 1
    fi
    case "$PWD/" in
      "$HOME"/*) ;;
      *)
        echo "env/cuda.sh: $PWD is outside \$HOME, which is all the colima VM mounts." >&2
        echo "  Move the repo under $HOME (or run colima with a mount for this path)." >&2
        exit 1
        ;;
    esac
    # printf %q quotes each argument so it survives the extra shell hop over ssh.
    CMD="$(printf '%q ' "$@")"
    exec colima ssh -- bash -c "cd $(printf %q "$PWD") && \
      if [ -d $(printf %q "$CUDA/bin") ]; then \
        export PATH=$(printf %q "$CUDA/bin"):\$PATH \
               LD_LIBRARY_PATH=$(printf %q "$CUDA/lib")\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}; \
      fi && \
      $CMD"
    ;;
  *)
    echo "env/cuda.sh: unsupported OS $(uname -s). Supported: Linux, macOS (via colima)." >&2
    echo "  On Windows use WSL2 and follow the Linux instructions." >&2
    exit 1
    ;;
esac
