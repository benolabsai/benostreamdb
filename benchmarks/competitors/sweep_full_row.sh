#!/usr/bin/env bash
#
# The benchmark sweep that `benchmarks/results/BENCHMARK_REPORT.md` summarizes.
#
# This is the single entry point. It measures every section the report reads,
# including the full competitor row (every engine the compose stack provides, on
# every vector dataset), then regenerates the report with the same interpreter it
# measured with.
#
# Three invariants are enforced per section, because a report that splices two
# machine states is worse than a report with a missing section:
#
#   1. artifacts older than this sweep are quarantined to
#      `results/stale/<stamp>-<section>/` before the section runs, so one section
#      can never mix a fresh number with a stale one;
#   2. a section that exits 0 without writing a fresh artifact is a failure, not a
#      pass — otherwise the previous run's files get attributed to this session;
#   3. each successful section is stamped into the session ledger via
#      `session_provenance.stamp`, and `generate_summary.py` refuses to build a
#      report whose sections span sessions.
#
# Usage:
#   ./sweep_full_row.sh                      # everything, keep going past failures
#   ONLY=vector_row ./sweep_full_row.sh      # one section
#   DRY_RUN=1 ./sweep_full_row.sh            # print the plan
#   CPUS=8 MEM=16g ONLY=vector_matrix ./sweep_full_row.sh
#
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(cd "$HERE/.." && pwd)"
ROOT_DIR="$(cd "$BENCH_DIR/.." && pwd)"

# Anchor every relative path at the repo root, exactly as docs/BENCHMARKING.md §3
# spells its commands. The host-side harnesses write to repo-relative paths
# (`benchmarks/results/...`), so invoking this script from any other directory made
# `production_workload.py` and `filters/run.py` do all their work and then fail on
# the write, which surfaced as "section FAILED" with no clue why.
cd "$ROOT_DIR"

# The interpreter that measures must be the interpreter that summarizes: the venv
# carries the abi3 wheel of the engine under test, and `generate_summary.py` quotes
# the engine version it is told. Override with SWEEP_PYTHON=... for a different env.
#
# Deliberately not `PYTHON`: a conda or pyenv install commonly exports that, and
# `${PYTHON:-default}` then runs the whole sweep under an interpreter that has no
# benostreamdb wheel — silently, since the first section would still pass.
PYTHON="${SWEEP_PYTHON:-$ROOT_DIR/.venv/bin/python}"
CPUS="${CPUS:-8}"
MEM="${MEM:-16g}"
ONLY="${ONLY:-}"
DRY_RUN="${DRY_RUN:-0}"
KEEP_GOING="${KEEP_GOING:-1}"

# The long-running services the containerised sections address by service name.
# A fresh clone has none of them, so the sweep starts them itself: that is what
# makes `sweep_full_row.sh` the only command someone needs after checking out the
# repository. `up -d` is idempotent, so an already-running stack is left alone.
ENGINE_SERVICES="${ENGINE_SERVICES:-pgvector opensearch qdrant milvus weaviate neo4j clickhouse trino}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SWEEP_START="$(date +%s)"

FAILED=()
SECTION_INDEX=0
SECTION_TOTAL=9
SECTION_NAME=""
SECTION_GLOBS=()
SECTION_START=0
SECTION_FAILED=0

want() {
    [ -z "$ONLY" ] && return 0
    case ",$ONLY," in *",$1,"*) return 0 ;; *) return 1 ;; esac
}

quarantine() {
    local glob rel mt dest moved=0
    for glob in "$@"; do
        for rel in "$BENCH_DIR"/$glob; do
            [ -e "$rel" ] || continue
            mt="$(stat -c %Y "$rel" 2>/dev/null)" || continue
            [ "$mt" -lt $((SWEEP_START - 5)) ] || continue
            dest="$BENCH_DIR/results/stale/$STAMP-$SECTION_NAME/$(realpath --relative-to="$BENCH_DIR" "$rel")"
            mkdir -p "$(dirname "$dest")"
            mv "$rel" "$dest"
            moved=$((moved + 1))
        done
    done
    [ "$moved" -eq 0 ] || echo "    quarantined $moved artifact(s) older than this sweep"
}

fresh_artifacts() {
    local glob rel mt
    for glob in "$@"; do
        for rel in "$BENCH_DIR"/$glob; do
            [ -e "$rel" ] || continue
            mt="$(stat -c %Y "$rel" 2>/dev/null)" || continue
            if [ "$mt" -ge $((SECTION_START - 5)) ]; then
                realpath --relative-to="$BENCH_DIR" "$rel"
            fi
        done
    done
}

section() {
    SECTION_NAME="$1"
    shift
    SECTION_GLOBS=("$@")
    SECTION_INDEX=$((SECTION_INDEX + 1))
    if ! want "$SECTION_NAME"; then
        return 1
    fi
    echo
    echo "=== [$SECTION_INDEX/$SECTION_TOTAL] $SECTION_NAME"
    SECTION_FAILED=0
    if [ "$DRY_RUN" = "1" ]; then
        return 0
    fi
    quarantine "${SECTION_GLOBS[@]}"
    SECTION_START="$(date +%s)"
    return 0
}

step() {
    echo "    \$ $*"
    [ "$DRY_RUN" = "1" ] && return 0
    "$@" && return 0
    # Report which step failed. A bare "section FAILED" cost a full re-run to
    # diagnose, because every step in the section is silent on success.
    #
    # The non-zero return is load-bearing: `step … || break` guards are how a
    # section notices that one dataset out of eight died. Returning 0 here made
    # every such guard dead code, so a section could measure 6 of 8 datasets and
    # still stamp itself with whatever it managed to write.
    echo "    step failed ($*)" >&2
    SECTION_FAILED=1
    return 1
}

# Some harnesses hardcode output paths relative to the repo root, so they have to
# run from there. Subshell so the cwd never leaks, and the failure still lands in
# the parent shell's SECTION_FAILED.
step_in_root() {
    echo "    \$ (cd $ROOT_DIR && $*)"
    [ "$DRY_RUN" = "1" ] && return 0
    ( cd "$ROOT_DIR" && "$@" ) && return 0
    echo "    step failed in $ROOT_DIR ($*)" >&2
    SECTION_FAILED=1
    return 1
}

section_done() {
    if [ "$DRY_RUN" = "1" ]; then
        return 0
    fi
    local elapsed=$(( $(date +%s) - SECTION_START ))
    if [ "$SECTION_FAILED" -ne 0 ]; then
        echo "    $SECTION_NAME: FAILED after ${elapsed}s" >&2
        FAILED+=("$SECTION_NAME")
        return 0
    fi

    local -a fresh=()
    local rel
    while IFS= read -r rel; do
        [ -n "$rel" ] && fresh+=("$rel")
    done < <(fresh_artifacts "${SECTION_GLOBS[@]}")

    if [ "${#fresh[@]}" -eq 0 ]; then
        # Exit 0 with nothing written means the command did nothing useful (wrong
        # flags, engine missing, output path the report does not read). Stamping it
        # would attribute the previous run's files to this session.
        echo "    $SECTION_NAME: NO FRESH ARTIFACTS after ${elapsed}s (looked for ${SECTION_GLOBS[*]}) — not stamped" >&2
        FAILED+=("$SECTION_NAME")
        return 0
    fi

    PYTHONPATH="$ROOT_DIR" "$PYTHON" - "$SECTION_NAME" "$elapsed" "${fresh[@]}" <<'PY'
import sys

from benchmarks import session_provenance

name, elapsed, artifacts = sys.argv[1], sys.argv[2], sys.argv[3:]
entry = session_provenance.stamp(name, artifacts)
print(f"    {name}: ok in {elapsed}s, stamped session {entry['session']} with {len(artifacts)} artifact(s)")
PY
}

VECTOR_DATASETS=(
    sift-128-euclidean
    glove-100-angular
    glove-200-angular
    nytimes-256-angular
    lastfm-64-dot
    fashion-mnist-784-euclidean
    mnist-784-euclidean
    gist-960-euclidean
)

# The full row: every engine the compose stack serves, on every dataset.
#
# How the services are reached: `docker compose -f docker-compose.bench.yml
# --profile run run --rm bench` starts the `bench` client (image
# `bsdb-bench-runner:latest`) on the `competitors_bench` network, where the
# long-running services are addressable by service name. Inside that container
# `run_bench.sh` dispatches: a BenoStreamDB index variant goes to
# `ann_benchmarks.py` (so pure-index latency is reported too), everything else
# goes to `run_competitor.py`, which is what implements pgvector / qdrant /
# weaviate / milvus / opensearch / faiss / hnswlib / lancedb. Results land in the
# bind-mounted `/results` as `<engine>_<dataset>_<device>.json`.

ROW_ENGINES=(
    benostreamdb
    benostreamdb_tq8
    benostreamdb_sq
    faiss
    hnswlib
    lancedb
    # lancedb_sq belongs here as well as in §3: it is the one competitor whose
    # refinement actually does something (it re-ranks from the scalar-quantized
    # codes), and leaving it out of this loop while listing it in REFINE_ENGINES
    # meant the Refine column was populated for four engines and empty for the
    # fifth it was written for.
    lancedb_sq
    pgvector
    qdrant
    weaviate
    milvus
    opensearch
)

# Engines that implement two-stage exact rescoring (retrieve k*factor, re-rank by
# exact distance). These get extra operating points from the same run, because the
# refined point is a *different* row of the report, not a different engine: it
# trades query throughput for recall, and the trade-off is only visible when both
# points are measured on the same machine in the same session.
#
# Milvus and Weaviate are deliberately absent: their quantization variants are not
# reachable through the adapter yet (Milvus needs HNSW_SQ/HNSW_PQ + `refine`,
# tracked as #125), and a `_rN` artifact that silently equals its unrefined twin
# is worse than a missing column.
REFINE_ENGINES=" benostreamdb benostreamdb_tq8 benostreamdb_sq lancedb lancedb_sq "
REFINE_FACTORS="${REFINE_FACTORS:-2 4}"

# --- start the services every containerised section talks to ----------------
if [ "$DRY_RUN" != "1" ] && [ -n "$ENGINE_SERVICES" ]; then
    echo
    echo "=== services: $ENGINE_SERVICES"
    docker compose -f "$HERE/docker-compose.bench.yml" --project-directory "$HERE" \
        up -d $ENGINE_SERVICES \
        || { echo "could not start the bench stack" >&2; exit 1; }
    unhealthy=""
    for _ in $(seq 1 72); do
        unhealthy="$(docker compose -f "$HERE/docker-compose.bench.yml" \
            --project-directory "$HERE" ps --format '{{.Service}} {{.Health}}' \
            | awk '$2 != "" && $2 != "healthy" {print $1}' | tr '\n' ' ')"
        [ -z "$unhealthy" ] && break
        sleep 5
    done
    if [ -n "$unhealthy" ]; then
        echo "    still starting after 6 min: $unhealthy" >&2
    else
        echo "    all services healthy"
    fi
fi

# --- [1/9] engine self-characterization, f32 and tq8 -------------------------
if section gpu_cdist results/gpu_cdist.json; then
    step "$PYTHON" benchmarks/gpu_cdist_bench.py
    section_done
fi

# --- [2/9] vector ANN, per dataset, full precision and tq8 -------------------
if section vector_ann "ann_benchmarks/results/*"; then
    for ds in "${VECTOR_DATASETS[@]}"; do
        for pair in "hnsw:" "hnsw_tq8:_tq8"; do
            idx="${pair%%:*}"
            suffix="${pair##*:}"
            step "$PYTHON" benchmarks/ann_benchmarks/run.py \
                --dataset "$ds" --index "$idx" --limit 20000 --queries 500 \
                --out "benchmarks/ann_benchmarks/results/${ds}_20k${suffix}.json" || break
        done
    done
    section_done
fi

# --- [3/9] competitor matrix: enveloped CPU+GPU pass, one run per dataset ----
# Every engine, BenoStreamDB included, is constrained to the same --cpus/--memory
# envelope here; this is the only pass in the sweep that is enveloped.
if section vector_matrix "competitors/results/*.json"; then
    # Build the manylinux wheel ONCE, before any measurement. Passing
    # `--build-wheel` inside the dataset loop recompiled the crate from the working
    # tree eight times, which both wasted the same three minutes eight times and
    # coupled the measurement to whatever was on disk at that moment: a
    # half-finished edit in the tree aborted two datasets mid-pass.
    step env PATH="$ROOT_DIR/.venv/bin:$PATH" maturin build --release --zig \
        --compatibility manylinux_2_28 -o dist
    for ds in "${VECTOR_DATASETS[@]}"; do
        step benchmarks/competitors/docker_bench.sh --both \
            --cpus "$CPUS" --mem "$MEM" --dataset "$ds" --limit 20000 --queries 500 || break
    done
    section_done
fi

# --- [4/9] the full competitor row, driven through the long-running stack ----
# `docker compose ... run --rm bench` is the client; the engines themselves are the
# long-running services (pgvector, weaviate, qdrant, opensearch, ...). This is the
# section that produces the per-engine, per-dataset row in the report.
if section vector_row "competitors/results/*.json"; then
    # Iterates VECTOR_DATASETS directly: a second hand-maintained list is how
    # fashion-mnist ended up with no BenoStreamDB row while every competitor had one.
    for ds in "${VECTOR_DATASETS[@]}"; do
        for eng in "${ROW_ENGINES[@]}"; do
            step env ENGINE="$eng" DATASET="$ds" LIMIT=20000 QUERIES=500 \
                docker compose -f "$HERE/docker-compose.bench.yml" \
                --project-directory "$HERE" --profile run run --rm bench
            case "$REFINE_ENGINES" in *" $eng "*)
                for rf in $REFINE_FACTORS; do
                    step env ENGINE="$eng" DATASET="$ds" LIMIT=20000 QUERIES=500 \
                        REFINE_FACTOR="$rf" \
                        docker compose -f "$HERE/docker-compose.bench.yml" \
                        --project-directory "$HERE" --profile run run --rm bench
                done
            ;; esac
        done
    done
    section_done
fi

# --- [5/9] graph: synthetic plus the SNAP datasets the report quotes ---------
if section graph "graph/results/*"; then
    step "$PYTHON" benchmarks/graph/run.py --nodes 10000 --edges 50000 \
        --engines benostreamdb,networkx,neo4j \
        --algorithms pagerank,connected_components,shortest_path \
        --out benchmarks/graph/results/graph_synth_10k.md
    step "$PYTHON" benchmarks/graph/run.py --dataset snap-web-google \
        --engines benostreamdb,networkx,neo4j \
        --algorithms pagerank,connected_components,shortest_path \
        --out benchmarks/graph/results/graph_snap_web_google.md
    step "$PYTHON" benchmarks/graph/run.py --dataset snap-roadnet-ca \
        --engines benostreamdb,networkx,neo4j \
        --algorithms pagerank,connected_components,shortest_path \
        --out benchmarks/graph/results/graph_snap_roadnet_ca.md
    section_done
fi

# --- [6/9] SQL: ClickBench 500k against DuckDB and DataFusion ----------------
if section sql "sql/results/*"; then
    step "$PYTHON" benchmarks/sql/run.py --rows 500000 \
        --engines benostreamdb,duckdb,datafusion \
        --out benchmarks/sql/results/clickbench_500k.md
    section_done
fi

# --- [7/9] BEIR lexical/hybrid ----------------------------------------------
# OpenSearch is the only server-backed participant, and it publishes no host
# port: the compose network is the only route to it. Running the host-side
# interpreter here (as this section used to) could therefore only ever measure
# the two embedded engines, so §4 of the report silently omitted the one
# competitor the stack exists to compare against. The container path
# (`WORKLOAD=beir` in run_bench.sh) reaches `$ES_URL` and writes the same
# competitor-schema JSON into the mounted `beir/results/`.
BEIR_ENGINES="${BEIR_ENGINES:-benostreamdb,tantivy,opensearch}"
if section beir "beir/results/*"; then
    # OpenSearch has to be up before the runner starts; the long-running stack
    # already provides it, so this only fails when the stack is not running.
    step bash -c 'test -n "$(docker compose -f "$1" --project-directory "$2" ps -q opensearch)"' \
        _ "$HERE/docker-compose.bench.yml" "$HERE"
    for ds in scifact nfcorpus arguana; do
        step env WORKLOAD=beir BEIR_DATASET="$ds" BEIR_ENGINES="$BEIR_ENGINES" \
            ENGINE=benostreamdb DEVICE=cpu \
            docker compose -f "$HERE/docker-compose.bench.yml" \
            --project-directory "$HERE" --profile run run --rm bench || break
    done
    section_done
fi

# --- [8/9] filters, production concurrency and maintenance -------------------
if section filters "results/production_filters.*"; then
    step "$PYTHON" benchmarks/filters/run.py --rows 200000
    section_done
fi

# Deliberately not "results/production_*": that pattern also matches
# production_filters.*, which the filters section writes, so running production
# afterwards quarantined the filters artifacts and left §6 empty.
if section production \
    "results/production_concurrent*" "results/production_concurrency*" \
    "results/production_maintenance*" "results/production_skew.*" \
    "results/production_soak.*"; then
    step "$PYTHON" benchmarks/production_workload.py --workload concurrent --sweep
    step "$PYTHON" benchmarks/production_workload.py --workload maintenance
    # §7 and §8 read exactly `results/production_skew.md` and
    # `results/production_soak.md`. They used to come from artifacts left by an
    # earlier session, so a sweep that quarantined `results/production_*` and then
    # did not rewrite them regenerated the report with two empty sections.
    step_in_root "$PYTHON" benchmarks/skew/run.py
    step env SOAK_REPORT="$BENCH_DIR/results/production_soak.md" \
        bash "$ROOT_DIR/scripts/pre_release_soak.sh" --quick
    section_done
fi

if [ "$DRY_RUN" = "1" ]; then
    exit 0
fi

if [ "${#FAILED[@]}" -ne 0 ]; then
    echo
    echo "failed sections: ${FAILED[*]}" >&2
    [ "$KEEP_GOING" = "1" ] || exit 1
fi

echo
echo "=== regenerating the report"
cd "$ROOT_DIR" || exit 1
"$PYTHON" benchmarks/generate_summary.py
