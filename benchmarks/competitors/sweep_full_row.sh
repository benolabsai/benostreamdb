#!/usr/bin/env bash
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

DATASETS=(
  "sift-128-euclidean"
  "mnist-784-euclidean"
  "lastfm-64-dot"
  "glove-100-angular"
  "glove-200-angular"
  "nytimes-256-angular"
  "gist-960-euclidean"
)

ENGINES=(
  "faiss"
  "hnswlib"
  "lancedb"
  "pgvector"
  "weaviate"
  "qdrant"
  "opensearch"
)

for ds in "${DATASETS[@]}"; do
  echo "=========================================================="
  echo "Starting sweep for dataset: $ds"
  echo "=========================================================="
  for eng in "${ENGINES[@]}"; do
    echo "--- Running $eng on $ds ---"
    ENGINE="$eng" DATASET="$ds" LIMIT=20000 QUERIES=500 \
      docker compose -f docker-compose.bench.yml --profile run run --rm bench \
      || echo "FAILED: $eng on $ds"
  done
done

echo "Sweep complete. Regenerating summary report..."
python3 ../generate_summary.py

