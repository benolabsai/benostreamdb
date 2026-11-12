#!/bin/bash
# Convenience runner for executing MuSiQue GraphRAG inside a dedicated Docker container.
#
# Default envelope emulates Azure Standard_D8s_v5 (8 vCPU / 32 GiB RAM). The
# profile controls how the harness reserves memory; it does not change what the
# host actually has.
#
# Default configuration: bge-large encodes, MiniLM reranks (--preset poc, the
# implicit default). Release presets:
#   ./run_musique_docker.sh --preset storage   # engine-native rows only
#   ./run_musique_docker.sh --preset poc       # + one MiniLM rerank row
#   ./run_musique_docker.sh --preset research  # every strategy, incl. BGE
#
# GPU usage (e.g. dual RTX 5070 Ti):
#   GPU=1 ./run_musique_docker.sh --embed-model bge-large --encode-gpus cuda:0,cuda:1 --build-only
#   GPU=1 ./run_musique_docker.sh --embed-model bge-large --embed-device cpu
# GPU=1 selects the 'bsdb-musique:gpu' image (CUDA torch wheels), passes --gpus
# through, and adds --embed-device cuda when you have not chosen a device. The
# encode device is recorded in embed_meta.json and in the report, so a GPU-built
# corpus is never confused with a GPU-benchmarked run.
#
# Runtime requirements for GPU: NVIDIA driver + nvidia-container-toolkit on the
# host, and Docker >= 19.03 for --gpus. RTX 50-series (Blackwell, sm_120) needs
# CUDA 12.8+ wheels, which is what TORCH_INDEX_URL defaults to.
#
# Optional strongest embedder (nvidia/NV-Embed-v2, ~14.6 GiB): does not fit the
# d8lds 16 GiB profile, and 16 GiB cards need int8. Build once, then evaluate:
#   ./run_musique_docker.sh --embed-model nvembed --build-only --allow-slow-encode
#   GPU=1 ./run_musique_docker.sh --embed-model nvembed --embed-dtype int8 --embed-device cuda
set -euo pipefail

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$BENCH_DIR/.." && pwd)"

IMAGE_NAME="bsdb-musique:local"
GPU_IMAGE_NAME="bsdb-musique:gpu"
DATA_DIR="${DATA_DIR:-/home/ralbright/data/datasets/musique}"
SCRATCH_DIR="${SCRATCH_DIR:-$BENCH_DIR/scratch_musique}"
WHEELS_DIR="${WHEELS_DIR:-$ROOT_DIR/dist}"

# Envelope overrides. Defaults target Azure Standard_D8s_v5 (8 vCPU / 32 GiB).
CPUS="${CPUS:-8}"
MEMORY="${MEMORY:-32g}"
HARDWARE="${HARDWARE:-d8s}"
CACHE_GB="${CACHE_GB:-16}"

# GPU selection: GPU=1 enables passthrough, GPU_IDS picks devices (all / 0 / 0,1).
GPU="${GPU:-0}"
GPU_IDS="${GPU_IDS:-all}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

# Which harness the entrypoint should execute. Default is the passage-level
# GraphRAG harness; set BENCH_SCRIPT=run_musique_entity_rag.py to run the
# entity-level Vector Graph RAG harness instead.
BENCH_SCRIPT="${BENCH_SCRIPT:-run_musique_graphrag.py}"

DOCKER_GPU_ARGS=()
if [ "$GPU" = "1" ]; then
    # Preflight: the NVIDIA container runtime mounts /run/nvidia-persistenced/socket
    # when the persistence daemon is expected. If the socket is absent, container
    # creation fails with "failed to fulfil mount request" before Python ever runs.
    if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
        echo "[musique-docker] GPU=1 but nvidia-smi cannot initialize a device:" >&2
        nvidia-smi -L 2>&1 | sed 's/^/  /' >&2
        cat >&2 <<'EOF'
  "Driver/library version mismatch" means the loaded kernel module and the
  userspace driver are different builds, usually after a driver package upgrade.
  Reboot the host (a module reload fails while X/desktop holds the device), then
  re-check with: nvidia-smi -L

  CPU encode is checkpointed and resumable in the meantime:
    ./run_musique_docker.sh --preset poc --allow-slow-encode
EOF
        exit 1
    fi
    if [ ! -S /run/nvidia-persistenced/socket ]; then
        cat >&2 <<'EOF'
[musique-docker] GPU=1 but /run/nvidia-persistenced/socket is missing.
  The NVIDIA container runtime will fail with:
    failed to fulfil mount request: open /run/nvidia-persistenced/socket

  Fix one of:
    sudo systemctl enable --now nvidia-persistenced
    # or, if the daemon is not installed, tell the toolkit not to mount it:
    sudo sed -i 's/^#\?no-persistenced.*/no-persistenced = true/' \
      /etc/nvidia-container-runtime/config.toml
    sudo systemctl restart docker

  Meanwhile, CPU encode is checkpointed and resumable:
    ./run_musique_docker.sh --preset poc --allow-slow-encode
EOF
        exit 1
    fi
    # `--gpus 1` means "one GPU", not "device index 1". Translate a device list
    # into the device= form so GPU_IDS=1 really selects the second card.
    if [ "$GPU_IDS" = "all" ]; then
        GPU_SELECTOR="all"
    else
        GPU_SELECTOR="device=$GPU_IDS"
    fi
    IMAGE_NAME="$GPU_IMAGE_NAME"
    DOCKER_GPU_ARGS=(--gpus "$GPU_SELECTOR" -e NVIDIA_DRIVER_CAPABILITIES=compute,utility)
fi

# Only request a TTY when one is actually attached (CI logs are not a terminal).
DOCKER_TTY_ARGS=(-i)
if [ -t 0 ] && [ -t 1 ]; then
    DOCKER_TTY_ARGS=(-it)
fi

mkdir -p "$DATA_DIR" "$SCRATCH_DIR" "$SCRATCH_DIR/tmp"

# Optional external corpus/triplets for like-for-like comparisons with the
# published Zilliz/HippoRAG numbers. Mount them only when present so the
# default run stays self-contained.
EXTRA_MOUNTS=()
if [ -f "$BENCH_DIR/musique_corpus.json" ]; then
    EXTRA_MOUNTS+=(-v "$BENCH_DIR/musique_corpus.json":/opt/bench/musique_corpus.json:ro)
fi
if [ -f "$BENCH_DIR/hippo_openie.json" ]; then
    EXTRA_MOUNTS+=(-v "$BENCH_DIR/hippo_openie.json":/opt/bench/hippo_openie.json:ro)
fi
if [ -f "$BENCH_DIR/zilliz_manifest.json" ]; then
    EXTRA_MOUNTS+=(-v "$BENCH_DIR/zilliz_manifest.json":/opt/bench/zilliz_manifest.json:ro)
fi

build_image() {
    echo "[musique-docker] Building '$1' (torch index: ${2:-default PyPI})..."
    if [ -n "${2:-}" ]; then
        docker build -f "$BENCH_DIR/Dockerfile.musique" --build-arg TORCH_INDEX_URL="$2" -t "$1" "$ROOT_DIR"
    else
        docker build -f "$BENCH_DIR/Dockerfile.musique" -t "$1" "$ROOT_DIR"
    fi
}

if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    if [ "$IMAGE_NAME" = "$GPU_IMAGE_NAME" ]; then
        build_image "$IMAGE_NAME" "$TORCH_INDEX_URL"
    else
        build_image "$IMAGE_NAME" ""
    fi
fi

# If GPUs are enabled but the caller did not pick an embed device, default the
# embedder to CUDA so the accelerator is actually used. Announce it loudly: this
# changes the device the reported latencies were measured on.
EXTRA_ARGS=()
if [ "$GPU" = "1" ]; then
    if [[ " $* " != *" --embed-device "* ]]; then
        EXTRA_ARGS+=(--embed-device cuda)
    fi
    if [[ " $* " != *" --embed-device "* ]]; then
        echo "[musique-docker] GPU mode: defaulting --embed-device cuda (pass --embed-device cpu to override)."
    fi
fi

echo "[musique-docker] Launching MuSiQue GraphRAG under ${HARDWARE} cgroups (${CPUS} vCPU / ${MEMORY} RAM), GPUs: ${GPU} (${GPU_IDS})..."
exec docker run --rm "${DOCKER_TTY_ARGS[@]}" \
  --name bsdb-musique-graphrag \
  "${DOCKER_GPU_ARGS[@]+"${DOCKER_GPU_ARGS[@]}"}" \
  --cpus="$CPUS" \
  --memory="$MEMORY" \
  --memory-swap="$MEMORY" \
  --shm-size=2g \
  -v "$WHEELS_DIR":/wheels:ro \
  -v "$BENCH_DIR/run_musique_entrypoint.sh":/opt/bench/run_musique_entrypoint.sh:ro \
  -v "$BENCH_DIR/run_musique_graphrag.py":/opt/bench/run_musique_graphrag.py:ro \
  -v "$BENCH_DIR/run_musique_entity_rag.py":/opt/bench/run_musique_entity_rag.py:ro \
  "${EXTRA_MOUNTS[@]+"${EXTRA_MOUNTS[@]}"}" \
  -v "$DATA_DIR":/data/musique \
  -v "${HF_HOME:-$HOME/.cache/huggingface}":/root/.cache/huggingface:ro \
  -v "$SCRATCH_DIR":/work/scratch \
  -e TMPDIR=/work/scratch/tmp \
  -e BENCH_SCRIPT="$BENCH_SCRIPT" \
  "$IMAGE_NAME" \
    --data-dir /data/musique \
    --scratch-dir /work/scratch \
    --hardware "$HARDWARE" \
    --strict-hardware \
    --threads "$CPUS" \
    --cache-gb "$CACHE_GB" \
    --mode tq8 \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" \
    "$@"
