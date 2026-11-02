#!/usr/bin/env bash
# Fetch NVIDIA's CUDA 12.8 toolkit (the Linux x86_64 or ARM64 redist, whichever
# matches the machine that will run it) that env/cuda.sh wraps as a black-box
# oracle (ptxas, nvdisasm, cuobjdump, nvcc, libcudart).
#
# We never vendor or redistribute this toolkit (it is NVIDIA's own proprietary
# software under env/cuda-12.8/LICENSE once unpacked) -- each user fetches
# their own copy straight from NVIDIA's servers, verified against NVIDIA's own
# published manifest (redistrib_12.8.0.json), the same way NVIDIA's installers
# do it internally.
#
# Usage: env/fetch_cuda.sh
# Result: env/cuda-12.8/{bin,include,lib,nvvm}/... (~300 MB unpacked, gitignored)
#
# Needs: curl, tar, xz, python3, and sha256sum or shasum. Not for NVIDIA Jetson (aarch64
# Tegra boards use a different 'linux-aarch64' build; this picks the server-ARM one).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/env/cuda-12.8"
BASE="https://developer.download.nvidia.com/compute/cuda/redist"
MANIFEST_URL="$BASE/redistrib_12.8.0.json"
# The toolkit has to match the CPU of the machine that will RUN it: this Linux
# box, or on macOS the colima Linux VM (which is aarch64 on Apple Silicon and
# x86_64 on Intel Macs). NVIDIA's redist names the two Linux builds
# linux-x86_64 and linux-sbsa (ARM64 server).
case "$(uname -s)" in
  Linux)  HOST_ARCH="$(uname -m)" ;;
  Darwin)
    command -v colima >/dev/null 2>&1 || { echo "macOS needs colima (the toolkit is Linux-only): brew install colima && colima start" >&2; exit 1; }
    colima status >/dev/null 2>&1 || { echo "the colima VM isn't running: colima start" >&2; exit 1; }
    HOST_ARCH="$(colima ssh -- uname -m | tr -d '\r')" ;;
  *) echo "unsupported OS $(uname -s): use Linux or macOS (colima)" >&2; exit 1 ;;
esac
case "$HOST_ARCH" in
  x86_64|amd64)   PLATFORM="linux-x86_64" ;;
  aarch64|arm64)  PLATFORM="linux-sbsa" ;;
  *) echo "unsupported CPU architecture: $HOST_ARCH" >&2; exit 1 ;;
esac
echo "target platform: $PLATFORM"

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
# NVIDIA ships .tar.xz; a minimal Linux install (Ubuntu, Debian) has tar but no xz.
command -v xz >/dev/null || { echo "need xz to unpack NVIDIA's .tar.xz archives: sudo apt install xz-utils (Debian/Ubuntu), sudo dnf install xz (Fedora), brew install xz (macOS)" >&2; exit 1; }
if command -v sha256sum >/dev/null; then SHASUM="sha256sum"
elif command -v shasum >/dev/null; then SHASUM="shasum -a 256"
else echo "need sha256sum (Linux: coreutils) or shasum (macOS) to verify the download" >&2; exit 1; fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "fetching NVIDIA's redistrib manifest..."
curl -fsSL "$MANIFEST_URL" -o "$TMP/redistrib.json" || {
  echo "couldn't download NVIDIA's manifest from $MANIFEST_URL -- check your network (behind a proxy? set https_proxy)" >&2; exit 1; }

mkdir -p "$OUT"
# A failed or interrupted fetch must not leave a half-filled env/cuda-12.8
# behind: the next run would refuse to start ("already exists") and ptxas
# would be missing. Only armed until the final step succeeds.
FETCH_OK=0
trap 'rm -rf "$TMP"; [ "$FETCH_OK" = 1 ] || rm -rf "$OUT"' EXIT
for comp in "${COMPONENTS[@]}"; do
  read -r rel sha256 <<<"$(python3 - "$TMP/redistrib.json" "$comp" "$PLATFORM" <<'PY'
import json, sys
manifest, comp, platform = sys.argv[1:4]
entry = json.load(open(manifest)).get(comp, {}).get(platform)
if entry is None:
    sys.exit(f"NVIDIA's CUDA 12.8 manifest has no {platform} build of {comp}")
print(entry["relative_path"], entry["sha256"])
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
# Run through env/cuda.sh so this also works on macOS, where ptxas has to run
# inside the colima VM rather than on the host.
"$ROOT/env/cuda.sh" ptxas --version
FETCH_OK=1

# Setup complete -- show the banner (same art `openassembler` shows), if the
# package is importable; harmless to skip if it isn't (e.g. not pip installed
# yet) since the fetch itself already succeeded above.
PYTHONPATH="$ROOT" python3 -c "
from openassembler.ui.banner import render
print()
print(render(include_logo=True))
" 2>/dev/null || true
