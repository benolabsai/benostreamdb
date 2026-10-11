#!/usr/bin/env bash
# Run the competitor + BenoStreamDB benchmark matrix in Docker with a single
# shared hardware profile applied to every participant.
#
#   # CPU baseline (default):
#   benchmarks/competitors/docker_bench.sh --cpus 8 --mem 16g \
#       --dataset sift-128-euclidean --limit 20000 --queries 500
#
#   # GPU pass, using the same CPU envelope for the host-side work:
#   benchmarks/competitors/docker_bench.sh --gpu --cpus 8 --mem 16g
#
#   # Both passes, tagged device=cpu and device=gpu:
#   benchmarks/competitors/docker_bench.sh --both --cpus 8 --mem 16g
#
#   # BEIR lexical/hybrid (BenoStreamDB vs Tantivy vs OpenSearch), same envelope:
#   benchmarks/competitors/docker_bench.sh --workload beir --cpus 8 --mem 16g
#   benchmarks/competitors/docker_bench.sh --workload beir --beir-mode hybrid \
#       --beir-engines benostreamdb,lancedb --cpus 8 --mem 16g
#
# Every participant (server engines AND the runner that executes each client,
# including BenoStreamDB) is constrained to the same --cpus/--memory envelope.
# The envelope and the host hardware profile are written to results/.
#
# BenoStreamDB needs an abi3 wheel: build it first, or pass --build-wheel.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
COMPOSE=(-f "$HERE/docker-compose.bench.yml")

CPUS=8
MEM=16g
DATASET=sift-128-euclidean
# Space-separated list of datasets to run in one pass (defaults to $DATASET).
# Engines stay up across datasets, so the whole vector matrix runs in one go.
DATASETS=""
LIMIT=20000
QUERIES=500
K=10
M=16
EFC=200
EFS=200
# `lancedb` is LanceDB's default disk-ANN index (IVF_PQ); `lancedb_hnsw` is its
# scalar-quantized HNSW variant. Both are run so the comparison covers each
# family rather than only the HNSW-shaped one.
# `benostreamdb` is full-precision HNSW (parity with the other float HNSW
# engines); `benostreamdb_tq8`/`_tq4` are the TurboQuant-compressed variants, so
# the recall/size trade-off is visible rather than conflated.
CPU_ENGINES="faiss hnswlib lancedb lancedb_hnsw pgvector opensearch qdrant milvus weaviate benostreamdb benostreamdb_tq8 benostreamdb_tq4 benostreamdb_pq"
# The GPU pass runs only engines with a *working* GPU execution path. CPU-only
# competitors run on the CPU inside the GPU container and now correctly report
# device=cpu (per-engine backend), so re-measuring them adds nothing — the CPU
# pass already covers every engine once. FAISS is excluded because faiss-gpu-cu12
# has no Blackwell (sm_120) kernels on this class of GPU (`faiss.get_num_gpus()`
# returns 0), so it silently falls back to CPU. Override with --gpu-engines.
GPU_ENGINES="benostreamdb benostreamdb_tq8 benostreamdb_tq4 benostreamdb_pq"
UP_ENGINES="pgvector opensearch qdrant milvus weaviate neo4j memgraph clickhouse trino"
# Workload families. `vector` is the default; `graph` and `sql` reuse the same
# runner image and envelope but dispatch to the graph/SQL adapters.
WORKLOAD=vector
GRAPH_ENGINES="networkx neo4j memgraph kuzu cugraph benostreamdb"
SQL_ENGINES="duckdb datafusion clickhouse trino benostreamdb"
ALGORITHM=pagerank
GRAPH_NODES=10000
GRAPH_EDGES_COUNT=50000
# Optional host edge-list file (source\ttarget per line) to use instead of the
# synthetic graph — e.g. a SNAP dataset. Copied into the mounted data dir.
GRAPH_EDGES_HOST=""
SQL_ROWS=500000
# Optional host Parquet file + SQL query to use instead of the synthetic table
# (e.g. TPC-H or NYC TLC). Both are copied/passed into the runner.
SQL_PARQUET_HOST=""
SQL_QUERY=""
SQL_DATASET=""
# BEIR lexical/hybrid workload. One process runs every engine, so the whole
# comparison shares a single container envelope (see the `beir` dispatch below).
BEIR_DATASET=scifact
BEIR_ENGINES="benostreamdb,tantivy,opensearch"
BEIR_MODE=lexical
DO_CPU=1
DO_GPU=0
BUILD_WHEEL=0
ENGINES_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpus) CPUS="$2"; shift 2 ;;
    --mem) MEM="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --datasets) DATASETS="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --queries) QUERIES="$2"; shift 2 ;;
    --k) K="$2"; shift 2 ;;
    --m) M="$2"; shift 2 ;;
    --ef-construction) EFC="$2"; shift 2 ;;
    --ef-search) EFS="$2"; shift 2 ;;
    --engines) CPU_ENGINES="$2"; shift 2 ;;
    --gpu-engines) GPU_ENGINES="$2"; shift 2 ;;
    --up-engines) UP_ENGINES="$2"; shift 2 ;;
    --workload) WORKLOAD="$2"; shift 2 ;;
    --graph-engines) GRAPH_ENGINES="$2"; shift 2 ;;
    --sql-engines) SQL_ENGINES="$2"; shift 2 ;;
    --algorithm) ALGORITHM="$2"; shift 2 ;;
    --graph-edges-host) GRAPH_EDGES_HOST="$2"; shift 2 ;;
    --graph-nodes) GRAPH_NODES="$2"; shift 2 ;;
    --graph-edges-count) GRAPH_EDGES_COUNT="$2"; shift 2 ;;
    --sql-rows) SQL_ROWS="$2"; shift 2 ;;
    --sql-parquet-host) SQL_PARQUET_HOST="$2"; shift 2 ;;
    --sql-query) SQL_QUERY="$2"; shift 2 ;;
    --sql-dataset) SQL_DATASET="$2"; shift 2 ;;
    --beir-dataset) BEIR_DATASET="$2"; shift 2 ;;
    --beir-engines) BEIR_ENGINES="$2"; shift 2 ;;
    --beir-mode) BEIR_MODE="$2"; shift 2 ;;
    --gpu) DO_CPU=0; DO_GPU=1; shift ;;
    --both) DO_CPU=1; DO_GPU=1; shift ;;
    --build-wheel) BUILD_WHEEL=1; shift ;;
    --engines-only) ENGINES_ONLY=1; shift ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ "$DO_GPU" == "1" ]]; then
  COMPOSE+=(-f "$HERE/docker-compose.bench.gpu.yml")
fi

export BENCH_CPUS="$CPUS"
export BENCH_MEM="$MEM"
export BENCH_RAM_GB="${MEM%g}"
export DATASET LIMIT QUERIES K M EF_CONSTRUCTION="$EFC" EF_SEARCH="$EFS"
export WORKLOAD ALGORITHM
export GRAPH_EDGES="${GRAPH_EDGES:-}" SQL="${SQL:-}" PARQUET="${PARQUET:-}"
export BEIR_DATASET BEIR_ENGINES BEIR_MODE

mkdir -p "$HERE/results" "$HERE/data"

# Only results written during this run belong in the rollup; older JSON from
# previous datasets/devices would otherwise be double-counted.
RUN_START="$(date +%s)"

# --- Hardware profile ------------------------------------------------------- #
profile="$HERE/results/hardware_profile.txt"
{
  echo "# Hardware profile"
  echo "date: $(date -Is)"
  echo "envelope_cpus: $CPUS"
  echo "envelope_mem: $MEM"
  echo "cpu_model: $(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | sed 's/^ *//')"
  echo "host_cores: $(nproc 2>/dev/null || echo '?')"
  echo "host_ram_gb: $(awk '/MemTotal/{printf "%.1f", $2/1048576}' /proc/meminfo 2>/dev/null)"
  echo "docker_server: $(docker version --format '{{.Server.Version}}' 2>/dev/null)"
  echo "compose: $(docker compose version --short 2>/dev/null)"
  echo "dataset: $DATASET limit: $LIMIT queries: $QUERIES"
  echo "hnsw: m=$M ef_construction=$EFC ef_search=$EFS"
  echo "gpus:"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null | sed 's/^/  /' || echo "  (none)"
} > "$profile"
echo "wrote $profile"

if [[ "$BUILD_WHEEL" == "1" ]]; then
  # Build a manylinux_2_28 wheel so it loads on the slim runner image (whose
  # glibc is older than a bleeding-edge host's). Requires `maturin[zig]`.
  echo "building abi3 manylinux_2_28 wheel ..."
  ( cd "$REPO" && maturin build --release --zig --compatibility manylinux_2_28 -o dist )
fi

# --- Server engines under the shared envelope ------------------------------- #
if [[ "$ENGINES_ONLY" == "0" && -n "$UP_ENGINES" ]]; then
  echo "starting engines: $UP_ENGINES (cpus=$CPUS mem=$MEM)"
  docker compose "${COMPOSE[@]}" up -d $UP_ENGINES
  for svc in $UP_ENGINES; do
    cid="$(docker compose "${COMPOSE[@]}" ps -q "$svc" || true)"
    [[ -z "$cid" ]] && continue
    for _ in $(seq 1 60); do
      state="$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$cid" 2>/dev/null || echo unknown)"
      [[ "$state" == "healthy" || "$state" == "running" ]] && break
      sleep 5
    done
    echo "  $svc: $state"
  done
fi

# --- Runner images ---------------------------------------------------------- #
docker compose "${COMPOSE[@]}" --profile run build bench
if [[ "$DO_GPU" == "1" ]]; then
  docker compose "${COMPOSE[@]}" --profile run build bench-gpu
fi

run_pass() {
  local device="$1" service="$2" engines="$3"
  export DEVICE="$device"
  echo ""
  echo "############ $device pass (service=$service, cpus=$CPUS mem=$MEM) ############"
  for engine in $engines; do
    echo "=== $engine [$device] ==="
    ENGINE="$engine" docker compose "${COMPOSE[@]}" --profile run run --rm "$service" \
      || echo "  (engine $engine did not complete)"
  done
}

# --- Workload dispatch ------------------------------------------------------ #
if [[ "$WORKLOAD" == "graph" ]]; then
  mkdir -p "$HERE/data"
  if [[ -n "$GRAPH_EDGES_HOST" ]]; then
    # A real edge list (e.g. a SNAP dataset), copied into the mounted data dir.
    base="$(basename "$GRAPH_EDGES_HOST")"
    cp "$GRAPH_EDGES_HOST" "$HERE/data/$base"
    export GRAPH_EDGES="/opt/bench/data/$base"
    # The dataset name (extension stripped) flows into the result filename.
    export GRAPH_DATASET="${GRAPH_DATASET:-${base%.tsv}}"
    echo "using graph edge list: $base (dataset=$GRAPH_DATASET)"
  else
    # A synthetic edge list, mounted into the runner at /opt/bench/data.
    "$REPO/.venv/bin/python" - "$HERE/data/graph_edges.txt" "$GRAPH_NODES" "$GRAPH_EDGES_COUNT" <<'PY'
import random, sys
path, nodes, edges = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
random.seed(42)
with open(path, "w", encoding="utf-8") as f:
    for _ in range(edges):
        f.write(f"{random.randrange(nodes)}\t{random.randrange(nodes)}\n")
print(f"wrote {path} ({nodes} nodes, {edges} edges)")
PY
    export GRAPH_EDGES="/opt/bench/data/graph_edges.txt"
    export GRAPH_DATASET="synth_${GRAPH_NODES}_${GRAPH_EDGES_COUNT}"
  fi
  # cuGraph needs the GPU runner; the other graph engines run fine there too.
  if [[ " $GRAPH_ENGINES " == *" cugraph "* ]]; then
    COMPOSE+=(-f "$HERE/docker-compose.bench.gpu.yml")
    # Build the base runner first: `bench-gpu` is `FROM bsdb-bench-runner:latest`,
    # so a stale base image would ship an old harness into the GPU image.
    docker compose "${COMPOSE[@]}" --profile run build bench bench-gpu
    export DEVICE=gpu
    echo ""
    echo "############ gpu pass (service=bench-gpu, cpus=$CPUS mem=$MEM) ############"
    for engine in $GRAPH_ENGINES; do
      echo "=== $engine [gpu] ==="
      ENGINE="$engine" docker compose "${COMPOSE[@]}" --profile run run --rm bench-gpu \
        || echo "  (engine $engine did not complete)"
    done
  else
    run_pass cpu bench "$GRAPH_ENGINES"
  fi
elif [[ "$WORKLOAD" == "sql" ]]; then
  # The Parquet table is mounted into the runner at /opt/bench/data/sql_t/ so
  # Trino's Hive connector can register the directory as an external table
  # without picking up the other files in the shared data dir.
  mkdir -p "$HERE/data/sql_t"
  if [[ -n "$SQL_PARQUET_HOST" ]]; then
    base="$(basename "$SQL_PARQUET_HOST")"
    cp "$SQL_PARQUET_HOST" "$HERE/data/sql_t/$base"
    export PARQUET="/opt/bench/data/sql_t/$base"
    export SQL_DATASET="${SQL_DATASET:-${base%.parquet}}"
    echo "using sql parquet: $base (dataset=$SQL_DATASET)"
  else
    "$REPO/.venv/bin/python" - "$HERE/data/sql_t/data.parquet" "$SQL_ROWS" <<'PY'
import sys
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
path, n = sys.argv[1], int(sys.argv[2])
rng = np.random.default_rng(42)
t = pa.table({
    "id": pa.array(np.arange(n, dtype=np.int64)),
    "value": pa.array(rng.random(n).astype(np.float64)),
    "category": pa.array(rng.integers(0, 100, n).astype(np.int64)),
})
pq.write_table(t, path)
print(f"wrote {path} ({n} rows)")
PY
    export PARQUET="/opt/bench/data/sql_t/data.parquet"
    export SQL_DATASET="${SQL_DATASET:-synth_${SQL_ROWS}}"
  fi
  if [[ -n "$SQL_QUERY" ]]; then
    export SQL="$SQL_QUERY"
  else
    export SQL="SELECT category, count(*) AS n, avg(value) AS avg_value FROM t GROUP BY category ORDER BY n DESC LIMIT 10"
  fi
  run_pass cpu bench "$SQL_ENGINES"
elif [[ "$WORKLOAD" == "beir" ]]; then
  # BEIR lexical/hybrid: a single process runs every engine (embedded Tantivy +
  # BenoStreamDB, plus the server-backed OpenSearch), so the whole comparison
  # shares one container envelope. OpenSearch is the only server-backed engine,
  # so it is brought up under the same envelope as the runner.
  if [[ "$ENGINES_ONLY" == "0" ]]; then
    echo "starting engines: opensearch (cpus=$CPUS mem=$MEM)"
    docker compose "${COMPOSE[@]}" up -d opensearch
    cid="$(docker compose "${COMPOSE[@]}" ps -q opensearch || true)"
    for _ in $(seq 1 60); do
      state="$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$cid" 2>/dev/null || echo unknown)"
      [[ "$state" == "healthy" || "$state" == "running" ]] && break
      sleep 5
    done
    echo "  opensearch: $state"
  fi
  docker compose "${COMPOSE[@]}" --profile run build bench
  # BEIR is CPU-bound: Okapi BM25 has no GPU path, and the hybrid dense half is
  # an HNSW query (the GPU accelerates index *construction*, not query). So
  # there is no GPU pass — every engine's `backend` is `cpu`. The GPU comparison
  # lives in the vector workload, where the GPU is actually used.
  echo ""
  echo "############ beir pass (service=bench, cpus=$CPUS mem=$MEM) ############"
  DEVICE=cpu docker compose -f "$HERE/docker-compose.bench.yml" \
    --profile run run --rm bench \
    || echo "  (beir pass did not complete)"
else
  # Vector: loop over the dataset list so the engines stay up across datasets.
  for ds in ${DATASETS:-$DATASET}; do
    export DATASET="$ds"
    echo ""
    echo "######## dataset: $ds ########"
    [[ "$DO_CPU" == "1" ]] && run_pass cpu bench "$CPU_ENGINES"
    [[ "$DO_GPU" == "1" ]] && run_pass gpu bench-gpu "$GPU_ENGINES"
  done
fi

# --- Rollup ----------------------------------------------------------------- #
rollup="$HERE/results/rollup.md"
{
  echo "# Benchmark rollup ($(date -Is))"
  echo ""
  echo "Hardware profile: cpus=$CPUS mem=$MEM — see hardware_profile.txt"
  echo ""
  echo "| Engine | Device | Dataset | Recall@k | QPS | p50 (ms) | p99 (ms) | Build (s) | Index (MB) |"
  echo "|---|---|---|---|---|---|---|---|---|"
  "$REPO/.venv/bin/python" - "$HERE/results" "$RUN_START" <<'PY'
import glob, json, os, sys
rows = []
run_start = int(sys.argv[2])
for path in sorted(glob.glob(os.path.join(sys.argv[1], "*_*.json"))):
    if os.path.getmtime(path) < run_start:
        continue
    try:
        r = json.load(open(path))
    except Exception:
        continue
    if r.get("available") is False:
        rows.append(f"| {r.get('engine')} | - | - | unavailable: {r.get('error','')} | - | - | - | - | - |")
        continue
    # Compare like-for-like: the competitor adapters return ids only (no payload
    # fetch), so use BenoStreamDB's pure-index number when the harness reports it
    # rather than its full search+row-fetch number.
    qps = r.get("pure_index_qps", r.get("qps", "-"))
    p50 = r.get("pure_p50_ms", r.get("p50_ms", "-"))
    p99 = r.get("pure_p99_ms", r.get("p99_ms", "-"))
    rows.append("| {e} | {dev} | {d} | {rec} | {qps} | {p50} | {p99} | {b} | {sz} |".format(
        e=r.get("engine"), dev=r.get("device", "-"), d=r.get("dataset", "-"),
        rec=r.get("recall_at_k", "-"), qps=qps,
        p50=p50, p99=p99,
        b=r.get("build_s", "-"), sz=r.get("index_mb", "-")))
print("\n".join(rows) if rows else "| (no results) | | | | | | | | |")
PY
} > "$rollup"

echo ""
echo "rollup: $rollup"
cat "$rollup"
