#!/usr/bin/env bash
# Make the engine's CUDA JIT path work with pip's CUDA 13 wheels.
#
# cudarc 0.13.9 probes for libnvrtc under the names
#   libnvrtc.so, libnvrtc.so.{12,11,10,1}, libnvrtc64*.so
# but the `nvidia-*-cu13` wheels ship only `libnvrtc.so.13`. Without a shim the
# probe panics ("Unable to dynamically load the nvrtc shared library"), the GPU
# falls back to CPU, and the panic noise can interfere with background index
# builds.
#
# Usage:
#   source scripts/create_cuda_shims.sh          # exports HDB_LD_LIBRARY_PATH
#   LD_LIBRARY_PATH="$HDB_LD_LIBRARY_PATH" python scripts/prepare_demo.py ...
#
# Or symlink once and export the directory yourself:
#   export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.*/site-packages/nvidia/cu13/lib:$PWD/cuda-shims"
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
LIBNVRTC_DIR="$(ls -d "$REPO"/.venv*/lib/python3*/site-packages/nvidia/cu13/lib 2>/dev/null | head -1 || true)"
SHIM_DIR="$REPO/.venv-cuda-shims"

if [ -z "${LIBNVRTC_DIR:-}" ] || [ ! -e "$LIBNVRTC_DIR/libnvrtc.so.13" ]; then
  echo "create_cuda_shims: no libnvrtc.so.13 found under $REPO/.venv*/lib/python3*/site-packages/nvidia/cu13/lib" >&2
  echo "install a CUDA 13 wheel set first, e.g.:  pip install nvidia-cuda-nvrtc-cu13" >&2
  return 1 2>/dev/null || exit 1
fi

mkdir -p "$SHIM_DIR"
for name in libnvrtc.so libnvrtc.so.12 libnvrtc.so.11 libnvrtc.so.10 libnvrtc.so.1 \
            libnvrtc64.so libnvrtc64_120_0.so libnvrtc64_12.so; do
  ln -sf "$LIBNVRTC_DIR/libnvrtc.so.13" "$SHIM_DIR/$name"
done

# The driver/runtime libs (libcudart etc.) live in the same directory, so both
# must be on the loader path.
export HDB_LD_LIBRARY_PATH="$SHIM_DIR:$LIBNVRTC_DIR"
export LD_LIBRARY_PATH="${HDB_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "create_cuda_shims: shims -> $SHIM_DIR (libnvrtc.so.13 from $LIBNVRTC_DIR)"
echo "create_cuda_shims: LD_LIBRARY_PATH set for this shell"