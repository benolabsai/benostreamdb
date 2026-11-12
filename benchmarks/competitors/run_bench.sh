#!/bin/sh
# Benchmark runner entrypoint (invoked by the image ENTRYPOINT `sh -lc`).
#
# Kept as a file rather than an inline compose `command:` because YAML folding
# mangled the multi-line `if/then` into an invalid one-liner.
#
# WORKLOAD selects the family: `vector` (default), `graph`, or `sql`.
set -e

WORKLOAD="${WORKLOAD:-vector}"

# BenoStreamDB index variants: the plain engine name is full-precision HNSW
# (parity with the other float HNSW engines); a `_tq8`/`_tq4`/`_pq` suffix
# selects the quantized/compressed index so the recall/size trade-off is
# reported explicitly instead of hiding behind a single number.
case "$ENGINE" in
    benostreamdb)     BSDB_INDEX=hnsw ;;
    benostreamdb_tq8) BSDB_INDEX=hnsw_tq8 ;;
    benostreamdb_sq)  BSDB_INDEX=hnsw_sq8 ;;
    benostreamdb_tq4) BSDB_INDEX=hnsw_tq4 ;;
    *)                BSDB_INDEX="" ;;
esac

# BenoStreamDB is installed from the mounted abi3 wheel for every workload that
# uses it (any index variant) and the BEIR comparison.
#
# `dist/` can hold several wheels of the same version at once — a local
# `maturin build` (cp310-abi3-linux_x86_64) alongside the manylinux wheel that
# `docker_bench.sh --build-wheel` produces. Passing the glob to pip then fails with
# ResolutionImpossible, because two candidates provide the same distribution, and
# the whole run dies before any engine is measured. Install exactly one: the newest.
if [ -n "$BSDB_INDEX" ] || [ "$WORKLOAD" = "beir" ]; then
    WHEEL="$(ls -t /wheels/*.whl 2>/dev/null | head -n 1)"
    if [ -z "$WHEEL" ]; then
        echo "[bench] no wheel found in /wheels; cannot run $ENGINE" >&2
        exit 1
    fi
    echo "[bench] installing $WHEEL"
    pip install --no-deps --force-reinstall "$WHEEL"
fi

# BenoStreamDB's vector workload uses the ANN-Benchmarks harness (it reports
# pure-index latency too); every other engine/workload uses run_competitor.py.
if [ -n "$BSDB_INDEX" ] && [ "$WORKLOAD" = "vector" ]; then
    # A refined point is a separate operating point of the same engine, so it needs
    # its own artifact name. Without the suffix the refined run overwrites the
    # unrefined one and the report can never show both (it would look like the
    # refinement had no effect, or like refinement was never measured).
    REFINE_TAG=""
    if [ "${REFINE_FACTOR:-1}" != "1" ]; then
        REFINE_TAG="_r${REFINE_FACTOR}"
    fi
    exec python /opt/bench/ann_benchmarks.py \
        --dataset "$DATASET" --limit "$LIMIT" --queries "$QUERIES" \
        --k "$K" --m "$M" --ef-construction "$EF_CONSTRUCTION" \
        --ef-search "$EF_SEARCH" --refine-factor "${REFINE_FACTOR:-1}" \
        --cores "$BENCH_CPUS" --ram-gb "${BENCH_MEM%g}" --device "$DEVICE" \
        --index "$BSDB_INDEX" --engine "$ENGINE" \
        --out "/results/${ENGINE}${REFINE_TAG}_${DATASET}_${DEVICE}.json"
fi

case "$WORKLOAD" in
    beir)
        # BEIR lexical/hybrid: one process runs every engine (embedded Tantivy +
        # BenoStreamDB, and the server-backed OpenSearch/Elasticsearch), so the
        # whole comparison shares this container's envelope. The dataset is
        # mounted read-only at /opt/bench/beir_data; reports land in
        # /results/beir/ (mounted from benchmarks/beir/results). The device is
        # part of the filename so cpu/gpu passes do not overwrite each other.
        mkdir -p /results/beir
        if [ "$BEIR_MODE" = "hybrid" ]; then
            exec python /opt/bench/beir_hybrid.py \
                --dataset "$BEIR_DATASET" --engines "$BEIR_ENGINES" \
                --data-dir /opt/bench/beir_data --device "${DEVICE:-cpu}" \
                --json-dir /results/beir \
                --out "/results/beir/${BEIR_DATASET}_hybrid_${DEVICE:-cpu}.md"
        fi
        exec python /opt/bench/beir_run.py \
            --dataset "$BEIR_DATASET" --engines "$BEIR_ENGINES" \
            --host "$ES_URL" --data-dir /opt/bench/beir_data --device "${DEVICE:-cpu}" \
            --json-dir /results/beir \
            --out "/results/beir/${BEIR_DATASET}_bm25_competitors_${DEVICE:-cpu}.md"
        ;;
    graph)
        exec python /opt/bench/run_competitor.py \
            --engine "$ENGINE" --workload graph --graph-edges "$GRAPH_EDGES" \
            --dataset "${GRAPH_DATASET:-graph}" \
            --algorithm "$ALGORITHM" --device "$DEVICE" \
            --cores "$BENCH_CPUS" --ram-gb "${BENCH_RAM_GB:-16}" \
            --out "/results/${ENGINE}_${GRAPH_DATASET:-graph}_${ALGORITHM}_${DEVICE}.json"
        ;;
    sql)
        exec python /opt/bench/run_competitor.py \
            --engine "$ENGINE" --sql "$SQL" --parquet "$PARQUET" \
            --dataset "${SQL_DATASET:-sql}" \
            --device "$DEVICE" --cores "$BENCH_CPUS" --ram-gb "${BENCH_RAM_GB:-16}" \
            --out "/results/${ENGINE}_${SQL_DATASET:-sql}_${DEVICE}.json"
        ;;
    *)
        # Same artifact-naming rule as the BenoStreamDB branch: a refined operating
        # point must not overwrite its unrefined sibling.
        REFINE_TAG_C=""
        if [ "${REFINE_FACTOR:-1}" != "1" ]; then
            REFINE_TAG_C="_r${REFINE_FACTOR}"
        fi
        exec python /opt/bench/run_competitor.py \
            --engine "$ENGINE" --dataset "$DATASET" --limit "$LIMIT" \
            --queries "$QUERIES" --k "$K" --m "$M" \
            --ef-construction "$EF_CONSTRUCTION" --ef-search "$EF_SEARCH" \
            --device "$DEVICE" --cores "$BENCH_CPUS" --ram-gb "${BENCH_RAM_GB:-16}" \
            --out "/results/${ENGINE}${REFINE_TAG_C}_${DATASET}_${DEVICE}.json"
        ;;
esac
