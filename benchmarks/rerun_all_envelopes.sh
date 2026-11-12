#!/usr/bin/env bash
# Re-run the three published vector/GraphRAG benchmarks under the SAME cgroup
# envelope and harness settings each dataset was originally measured with.
#
# Those envelopes are not a preference — they are recorded in the result
# artifacts, and a number produced under a different one is not comparable:
#
#   NeurIPS BigANN 10M  benchmarks/results/neurips_bigann/bigann_10m_10000k_*.json
#     meta.envelope = {hardware: d8lds, threads: 8, cache_gb: 8, preload_gb: 8.0}
#     meta.harness  = {queries: 10000, k: 10, m: 16, ef_construction: 200,
#                      flush_every: 5000000, modes: [sq8, pq], grid: 16 points}
#
#   Cohere 10M          benchmarks/results/neurips_bigann/bigann_cohere-10m_*.json
#     meta.envelope = {hardware: d8lds, threads: 8, cache_gb: 8, preload_gb: 8.0}
#     meta.harness  = {queries: 1000, k: 10, m: 16, ef_construction: 200,
#                      flush_every: 5000000, all_modes: false, mode: sq8}
#     meta.index_compaction = {enabled: true, k: 1024, m: 16, ef_construction: 200}
#
#   MuSiQue parity      benchmarks/MUSIQUE_PARITY.md + benchmarks/MUSIQUE_RESULTS.md
#     d8s envelope: 8 vCPU / 32 GiB, cache_gb 16, threads 8, k=5
#
# d8lds = Azure Standard_D8lds_v5: 8 vCPU / 16 GiB / local NVMe.
# d8s   = Azure Standard_D8s_v5:   8 vCPU / 32 GiB.
# `--hardware d8lds --strict-hardware` makes the harness abort if the container
# is not actually constrained, so an unconstrained run cannot silently produce a
# number that looks like a result.
#
# The runs are strictly sequential: two 8-vCPU containers on one host share
# memory bandwidth and page cache, which would quietly invalidate both.
#
# Usage:
#   ./benchmarks/rerun_all_envelopes.sh [neurips|cohere|musique|all]
set -euo pipefail

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$BENCH_DIR/.." && pwd)"
# Namespaced deliberately: a generic DATA_ROOT is often already exported by an
# unrelated project in the operator's shell, and `${BSDB_DATA_ROOT:-default}` would
# then silently mount the wrong directory (the harness reports the dataset as
# missing, which reads like a download problem rather than a wiring mistake).
BSDB_DATA_ROOT="${BSDB_DATA_ROOT:-/home/ralbright/data/datasets}"
SCRATCH="${SCRATCH:-$BENCH_DIR/scratch_neurips}"
WHEELS="${WHEELS:-$ROOT_DIR/dist}"
LOGS="${LOGS:-/tmp}"
WHAT="${1:-all}"

NEURIPS_IMAGE="bsdb-neurips:latest"
MUSIQUE_IMAGE="bsdb-musique:local"

require() {
    command -v "$1" >/dev/null 2>&1 || { echo "missing required tool: $1" >&2; exit 1; }
}

require_dataset() {
    local dir="$1"; shift
    [ -d "$dir" ] || { echo "dataset directory not found: $dir" >&2; exit 1; }
    local f
    for f in "$@"; do
        [ -s "$dir/$f" ] || {
            echo "dataset $dir is missing (or has an empty) $f" >&2
            echo "resolved from BSDB_DATA_ROOT=${BSDB_DATA_ROOT}" >&2
            exit 1
        }
    done
}

prepare() {
    require docker
    [ -n "$(ls -A "$WHEELS"/*.whl 2>/dev/null)" ] || {
        echo "no wheel in $WHEELS — build one first:" >&2
        echo "  .venv/bin/maturin build --release --out dist" >&2
        exit 1
    }
    docker image inspect "$NEURIPS_IMAGE" >/dev/null 2>&1 || {
        echo "building $NEURIPS_IMAGE" >&2
        docker build -f "$BENCH_DIR/Dockerfile.neurips" -t "$NEURIPS_IMAGE" "$ROOT_DIR"
    }
    mkdir -p "$SCRATCH/tmp" "$SCRATCH/results"
}

# $1 = flavour label, $2 = data dir, rest = harness args
run_neurips() {
    local flavour="$1" data="$2"
    shift 2
    echo "=== NeurIPS harness ($flavour) under d8lds: 8 vCPU / 16 GiB ==="
    docker run --rm --name "bsdb-neurips-$flavour" \
        --cpus=8 --memory=16g --memory-swap=16g \
        -v "$data:/data/vectors:ro" \
        -v "$SCRATCH:/work/scratch" \
        -v "$WHEELS:/wheels:ro" \
        -e TMPDIR=/work/scratch/tmp \
        "$NEURIPS_IMAGE" \
        --data-dir /data/vectors \
        --scratch-dir /work/scratch \
        --db-dir "/work/scratch/${flavour}_db" \
        --out /work/scratch/results \
        --hardware d8lds --strict-hardware \
        "$@"
}

neurips_10m() {
    require_dataset "$BSDB_DATA_ROOT/bigann" \
        base.10M.u8bin query.public.10K.u8bin GT.10M
    prepare
    # Matches bigann_10m_10000k_20261101T201843Z: modes sq8+pq, full sweep.
    run_neurips 10m "$BSDB_DATA_ROOT/bigann" \
        --dataset 10m --limit 0 --queries 10000 --stage both \
        --modes sq8,pq --flush-every 5000000 --sweep \
        2>&1 | tee "$LOGS/bench_neurips_10m.log"
}

cohere_10m() {
    require_dataset "$BSDB_DATA_ROOT/cohere" base.fbin queries.fbin gt_100.fbin
    prepare
    # Matches bigann_cohere-10m_10000k_20261104T130028Z: sq8 only, 1000 queries,
    # index compaction on with k=1024. Cohere ships `.fbin` files, not `.u8bin`,
    # and lives in its own directory (the harness derives the file names from the
    # dataset tag).
    run_neurips cohere "$BSDB_DATA_ROOT/cohere" \
        --dataset cohere-10m --limit 0 --queries 1000 --stage both \
        --mode sq8 --index-compaction --compaction-k 1024 --flush-every 5000000 --sweep \
        2>&1 | tee "$LOGS/bench_cohere_10m.log"
}

musique_parity() {
    require_dataset "$BSDB_DATA_ROOT/musique" musique_ans_v1.0_dev.jsonl
    prepare
    docker image inspect "$MUSIQUE_IMAGE" >/dev/null 2>&1 || {
        echo "building $MUSIQUE_IMAGE" >&2
        docker build -f "$BENCH_DIR/Dockerfile.musique" -t "$MUSIQUE_IMAGE" "$ROOT_DIR"
    }
    echo "=== MuSiQue parity under d8s: 8 vCPU / 32 GiB, cache 16 GiB ==="
    # f32 index + BGE cross-encoder is the published headline row; 1000 questions
    # and k=5 match MUSIQUE_PARITY.md. If the Zilliz sample manifest is present it
    # is passed through so the question slice is identical to the published run.
    # The Zilliz sample manifest ships in the repo (benchmarks/zilliz_manifest.json)
    # and is reachable in-container at /opt/bench, so parity is the default rather
    # than something that silently degrades to the dev set when an environment
    # variable is forgotten. A manifest outside those two mounts is an error, not a
    # fallback: it would measure a different question slice and read as parity.
    MUSIQUE_MANIFEST="${MUSIQUE_MANIFEST:-$BENCH_DIR/zilliz_manifest.json}"
    local manifest_args=()
    if [ -f "$MUSIQUE_MANIFEST" ]; then
        case "$MUSIQUE_MANIFEST" in
            "$BENCH_DIR"/*) manifest_args=(--sample-manifest "/opt/bench/${MUSIQUE_MANIFEST#"$BENCH_DIR"/}") ;;
            "$BSDB_DATA_ROOT"/musique/*) manifest_args=(--sample-manifest "/data/musique/${MUSIQUE_MANIFEST#"$BSDB_DATA_ROOT"/musique/}") ;;
            *) echo "$MUSIQUE_MANIFEST is not under \$BENCH_DIR or \$BSDB_DATA_ROOT/musique, so the container cannot see it" >&2; exit 1 ;;
        esac
        manifest_args+=(--manifest-dataset musique)
    else
        echo "note: no sample manifest at $MUSIQUE_MANIFEST — running the first 1000"
        echo "      questions of the dev set instead (not byte-identical to parity)."
        manifest_args=(--max-questions 1000)
    fi
    # The reranker and embedder are pulled from the Hugging Face hub at load time.
    # The benchmark container has no egress, and the parity baseline was measured
    # against the weights already in the operator's cache, so mount that cache and
    # pin the harness offline: a download would silently change the model revision
    # and make the comparison a different experiment.
    HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
    [ -d "$HF_CACHE" ] || {
        echo "Hugging Face cache not found at $HF_CACHE (set HF_CACHE)" >&2
        exit 1
    }
    docker run --rm --name bsdb-musique-parity \
        --cpus=8 --memory=32g --memory-swap=32g \
        -v "$BSDB_DATA_ROOT/musique:/data/musique" \
        -v "$BENCH_DIR:/opt/bench:ro" \
        -v "$SCRATCH:/work/scratch" \
        -v "$WHEELS:/wheels:ro" \
        -v "$HF_CACHE:/hf-cache" \
        -e TMPDIR=/work/scratch/tmp \
        -e HF_HOME=/hf-cache \
        -e HF_HUB_OFFLINE=1 \
        -e TRANSFORMERS_OFFLINE=1 \
        -e BENCH_SCRIPT=run_musique_parity.py \
        "$MUSIQUE_IMAGE" \
        --data-dir /data/musique \
        --scratch-dir /work/scratch/musique_parity \
        --corpus-json /opt/bench/musique_corpus.json \
        --triplets-json /opt/bench/hippo_openie.json \
        --mode f32 --reranker bge --top-k 5 \
        --hardware d8s --threads 8 --cache-gb 16 \
        "${manifest_args[@]}" \
        2>&1 | tee "$LOGS/bench_musique_parity.log"
}

case "$WHAT" in
    neurips) neurips_10m ;;
    cohere)  cohere_10m ;;
    musique) musique_parity ;;
    all)
        neurips_10m
        cohere_10m
        musique_parity
        ;;
    *)
        echo "usage: $0 [neurips|cohere|musique|all]" >&2
        exit 2
        ;;
esac
