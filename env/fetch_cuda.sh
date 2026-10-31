#!/usr/bin/env bash
# Fetch NVIDIA's CUDA 12.8 toolkit (linux-sbsa / ARM64 redist) that env/cuda.sh
# wraps as a black-box oracle (ptxas, nvdisasm, cuobjdump, nvcc, libcudart).
#
# We never vendor or redistribute this toolkit (it is NVIDIA's own proprietary
# software under env/cuda-12.8/LICENSE once unpacked) -- each user fetches
# their own copy straight from NVIDIA's servers, verified against NVIDIA's own
# published manifest (redistrib_12.8.0.json), the same way NVIDIA's installers
# do it internally.
#
# Usage: env/fetch_cuda.sh
# Result: env/cuda-12.8/{bin,include,lib,nvvm}/... (~1GB unpacked, gitignored)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/env/cuda-12.8"
BASE="https://developer.download.nvidia.com/compute/cuda/redist"
MANIFEST_URL="$BASE/redistrib_12.8.0.json"
PLATFORM="linux-sbsa"   # ARM64 server (matches WORKFLOW.md's ground-truth build)

# Components that make up env/cuda-12.8/{bin,include,lib,nvvm}:
#   cuda_nvcc      -> nvcc, ptxas, cudafe++, nvlink, fatbinary, nvvm/ (libdevice, libnvvm)
#   cuda_nvdisasm  -> nvdisasm
#   cuda_cuobjdump -> cuobjdump
#   cuda_cudart    -> headers (cuda_runtime.h, ...) + libcudart
COMPONENTS=(cuda_nvcc cuda_nvdisasm cuda_cuobjdump cuda_cudart)

if [ -d "$OUT" ]; then
  echo "env/cuda-12.8 already exists; remove it first if you want to re-fetch." >&2
  exit 1
fi

command -v curl >/dev/null || { echo "need curl" >&2; exit 1; }
command -v python3 >/dev/null || { echo "need python3" >&2; exit 1; }
command -v tar >/dev/null || { echo "need tar" >&2; exit 1; }
SHASUM="shasum -a 256"
command -v shasum >/dev/null || SHASUM="sha256sum"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "fetching NVIDIA's redistrib manifest..."
curl -fsSL "$MANIFEST_URL" -o "$TMP/redistrib.json"

mkdir -p "$OUT"
for comp in "${COMPONENTS[@]}"; do
  read -r rel sha256 <<<"$(python3 - "$TMP/redistrib.json" "$comp" "$PLATFORM" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))[sys.argv[2]][sys.argv[3]]
print(d["relative_path"], d["sha256"])
PY
  )"
  url="$BASE/$rel"
  dest="$TMP/$(basename "$rel")"
  echo "downloading $comp ($rel)..."
  curl -fsSL "$url" -o "$dest"
  echo "  $sha256  $dest" | $SHASUM -c - >/dev/null || {
    echo "sha256 mismatch for $comp -- aborting, NOT extracting untrusted archive" >&2
    exit 1
  }
  echo "  verified, extracting..."
  tar -xf "$dest" -C "$TMP"
  # Each archive unpacks to a directory like cuda_nvcc-linux-sbsa-12.8.61-archive/;
  # merge its contents into env/cuda-12.8/.
  srcdir="$(find "$TMP" -maxdepth 1 -type d -name "${comp}-${PLATFORM}-*")"
  cp -R "$srcdir"/. "$OUT"/
  rm -rf "$srcdir" "$dest"
done

echo "done: $OUT"
"$OUT/bin/ptxas" --version
