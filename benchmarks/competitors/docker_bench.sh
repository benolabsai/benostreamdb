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
LIMIT=20000
QUERIES=500
K=10
M=16
EFC=200
EFS=200
CPU_ENGINES="faiss hnswlib lancedb duckdb pgvector elasticsearch benostreamdb"
GPU_ENGINES="faiss benostreamdb"
UP_ENGINES="pgvector opensearch neo4j"
DO_CPU=1
DO_GPU=0
BUILD_WHEEL=0
ENGINES_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpus) CPUS="$2"; shift 2 ;;
    --mem) MEM="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --queries) QUERIES="$2"; shift 2 ;;
    --k) K="$2"; shift 2 ;;
    --m) M="$2"; shift 2 ;;
    --ef-construction) EFC="$2"; shift 2 ;;
    --ef-search) EFS="$2"; shift 2 ;;
    --engines) CPU_ENGINES="$2"; shift 2 ;;
    --gpu-engines) GPU_ENGINES="$2"; shift 2 ;;
    --up-engines) UP_ENGINES="$2"; shift 2 ;;
    --gpu) DO_CPU=0; DO_GPU=1; shift ;;
    --both) DO_CPU=1; DO_GPU=1; shift ;;
    --build-wheel) BUILD_WHEEL=1; shift ;;
    --engines-only) ENGINES_ONLY=1; shift ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
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

mkdir -p "$HERE/results" "$HERE/data"

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
  echo "building abi3 wheel ..."
  ( cd "$REPO" && maturin build --release -o dist )
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

[[ "$DO_CPU" == "1" ]] && run_pass cpu bench "$CPU_ENGINES"
[[ "$DO_GPU" == "1" ]] && run_pass gpu bench-gpu "$GPU_ENGINES"

# --- Rollup ----------------------------------------------------------------- #
rollup="$HERE/results/rollup.md"
{
  echo "# Benchmark rollup ($(date -Is))"
  echo ""
  echo "Hardware profile: cpus=$CPUS mem=$MEM — see hardware_profile.txt"
  echo ""
  echo "| Engine | Device | Dataset | Recall@k | QPS | p50 (ms) | p99 (ms) | Build (s) | Index (MB) |"
  echo "|---|---|---|---|---|---|---|---|---|"
  "$REPO/.venv/bin/python" - "$HERE/results" <<'PY'
import glob, json, os, sys
rows = []
for path in sorted(glob.glob(os.path.join(sys.argv[1], "*_*.json"))):
    try:
        r = json.load(open(path))
    except Exception:
        continue
    if r.get("available") is False:
        rows.append(f"| {r.get('engine')} | - | - | unavailable: {r.get('error','')} | - | - | - | - | - |")
        continue
    rows.append("| {e} | {dev} | {d} | {rec} | {qps} | {p50} | {p99} | {b} | {sz} |".format(
        e=r.get("engine"), dev=r.get("device", "-"), d=r.get("dataset", "-"),
        rec=r.get("recall_at_k", "-"), qps=r.get("qps", "-"),
        p50=r.get("p50_ms", "-"), p99=r.get("p99_ms", "-"),
        b=r.get("build_s", "-"), sz=r.get("index_mb", "-")))
print("\n".join(rows) if rows else "| (no results) | | | | | | | | |")
PY
} > "$rollup"

echo ""
echo "rollup: $rollup"
cat "$rollup"
