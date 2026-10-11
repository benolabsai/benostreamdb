#!/usr/bin/env python3
"""Unified Benchmark Suite Summary Generator.

Scans result artifacts across:
- benchmarks/ann_benchmarks/results/
- benchmarks/graph/results/
- benchmarks/sql/results/
- benchmarks/results/
And generates a comprehensive executive summary in benchmarks/results/BENCHMARK_REPORT.md.
"""

from __future__ import annotations

import glob
import json
import os
import platform
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Reference configuration for every participant, so the numbers in this report
# can be read against HOW each engine was set up and WHERE it was measured.
COMPETITOR_CONFIG = [
    ("vector", "benostreamdb",
     "Native HNSW/IVF via benchmarks/ann_benchmarks",
     "embedded (in-process Rust)"),
    ("vector", "faiss",
     "IndexHNSWFlat, M/efConstruction/efSearch; L2 or IP (L2-normalized for cosine)",
     "embedded (in-process C++)"),
    ("vector", "hnswlib",
     "HNSW, M/ef_construction/ef; space = l2 / ip / cosine per metric",
     "embedded (in-process C++)"),
    ("vector", "lancedb",
     "IVF_PQ (LanceDB default), distance l2/cosine/dot, nprobes = ef_search/10",
     "embedded (in-process Rust)"),
    ("vector", "lancedb_hnsw",
     "HnswSq (scalar-quantized HNSW), M/ef_construction, ef = ef_search",
     "embedded (in-process Rust)"),
    ("vector", "pgvector",
     "HNSW m/ef_construction; shared_buffers=4GB + maintenance_work_mem=2GB; "
     "hnsw.ef_search; op matches opclass (<-> L2 / <=> cosine / <#> IP)",
     "client → server (Postgres)"),
    ("vector", "opensearch",
     "knn_vector, Lucene HNSW, m/ef_construction, knn.algo_param.ef_search",
     "client → server (OpenSearch)"),
    ("vector", "qdrant",
     "collection HNSW, m/ef_construct, hnsw_ef; distance Euclid/Cosine/Dot",
     "client → server (Qdrant)"),
    ("vector", "milvus",
     "collection HNSW, M/efConstruction, ef; distance L2/IP/COSINE; Strong consistency",
     "client → server (Milvus)"),
    ("vector", "weaviate",
     "collection HNSW, max_connections/ef_construction, dynamic ef; "
     "distance cosine/dot/l2-squared",
     "client → server (Weaviate)"),
    ("graph", "benostreamdb",
     "CSR graph, in-process Rust (PageRank 30 iters / connected components / shortest path)",
     "embedded (in-process Rust)"),
    ("graph", "networkx",
     "in-memory Python (correctness oracle, not a perf baseline)",
     "embedded (in-process Python)"),
    ("graph", "neo4j",
     "Neo4j 5.26 + GDS, gds.graph.project + native gds.*.mutate; page cache 4G; "
     "latency = JVM compute (no per-node Bolt marshalling)",
     "native GDS (Neo4j JVM)"),
    ("graph", "memgraph",
     "native engine + MAGE query modules (pagerank.get / weakly_connected_components); "
     "aggregate-only result",
     "native engine (Memgraph)"),
    ("graph", "kuzu",
     "embedded columnar graph DB; page_rank / weakly_connected_components on a projected graph",
     "embedded (in-process C++)"),
    ("graph", "cugraph",
     "cuGraph on GPU (RMM managed memory, renumber=True)",
     "embedded (in-process GPU)"),
    ("sql", "benostreamdb",
     "in-process SQL (DataFusion-backed)",
     "embedded (in-process Rust)"),
    ("sql", "duckdb",
     "SET threads = BENCH_CPUS; SET memory_limit = BENCH_MEM",
     "embedded (in-process C++)"),
    ("sql", "datafusion",
     "target_partitions = BENCH_CPUS",
     "embedded (in-process Rust)"),
    ("sql", "clickhouse",
     "MergeTree ORDER BY tuple(); Parquet loaded via the client",
     "client → server (ClickHouse)"),
    ("sql", "trino",
     "Hive connector over the shared Parquet (external table)",
     "client → server (Trino)"),
]


def generate_summary():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_path = os.path.join(RESULTS_DIR, "BENCHMARK_REPORT.md")

    sections = [
        "# BenoStreamDB Comprehensive Benchmark Report",
        "",
        f"- **Generated At**: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        f"- **Platform**: {platform.platform()}",
        f"- **Python**: {platform.python_version()}",
        "",
        "---",
        "",
    ]

    # 1. Vector ANN Benchmarks
    #
    # These are BenoStreamDB's *own* index variants (HNSW / TurboQuant TQ8/TQ4 /
    # PQ) with no competitor, so they are not a head-to-head comparison — the
    # competitor matrix is §9. Keep only a pointer so the comparison report does
    # not duplicate raw self-benchmarks.
    sections.append("## 1. Vector ANN Performance (BenoStreamDB internal characterization)")
    sections.append("")
    sections.append(
        "Index-variant characterization for BenoStreamDB itself (HNSW, TurboQuant "
        "`hnsw_tq8`/`hnsw_tq4`, IVF-PQ) with **no competitor**. Head-to-head vector "
        "comparison is in **§9**; raw results are under "
        "`benchmarks/ann_benchmarks/results/`."
    )
    sections.append("")
    sections.append("### TurboQuant trade-off (pros and cons)")
    sections.append("")
    sections.append(
        "- **Pro — smaller index / less memory.** TurboQuant quantizes the stored "
        "vectors: `hnsw_tq8` (8-bit) is ≈4× smaller than float32 and `hnsw_tq4` "
        "(4-bit) ≈8× smaller, so more vectors fit per node and the working set "
        "(and on-disk sidecar) shrinks accordingly.\n"
        "- **Pro — cheaper distances / lower latency.** Comparing 1-byte codes is "
        "faster than 4-byte floats, so the quantized indexes typically serve lower "
        "p50 latency and higher QPS at a fixed `ef_search`.\n"
        "- **Pro — same HNSW graph / API.** The graph and query surface are "
        "unchanged; only the stored codes differ, so no schema/DDL difference.\n"
        "- **Con — lower recall.** Quantization is lossy: the stored codes no "
        "longer rank the exact neighbours, so recall@k drops (recovers only "
        "partially by raising `ef_search`, at a latency cost).\n"
        "- **Con — approximate distances for reranking.** Scores are approximate, "
        "so use the index for candidate discovery and rerank from the float "
        "payload where exactness matters.\n"
        "- **Default.** Full-precision `hnsw` is the default; `hnsw_tq8`/`hnsw_tq4`/"
        "`hnsw_pq` are opt-in for workloads that will trade recall for size/latency. "
        "§9 reports `benostreamdb` (float) beside `benostreamdb_tq8`/`_tq4` so the "
        "trade-off is explicit; holding several precisions on one column and "
        "selecting per query is on the roadmap (Theme 6)."
    )
    sections.append("")

    # 2. Graph Analytics Benchmarks
    #
    # Sourced from the same Docker competitor matrix as the vector/SQL results
    # (shared hardware envelope) so every engine appears — incl. Neo4j + GDS,
    # Memgraph and cuGraph. The `layer` column records HOW each engine is
    # measured: Neo4j/Memgraph run the algorithm server-side in a native engine
    # (timing `.stream` from Python measured Bolt/Python marshalling, not the
    # algorithm), while the others are in-process libraries.
    sections.append("## 2. Graph Analytics Performance (BenoStreamDB vs NetworkX vs Neo4j + GDS)")
    sections.append("")
    narrative = os.path.join(BASE_DIR, "graph", "results", "graph_competitors.md")
    if os.path.exists(narrative):
        with open(narrative, encoding="utf-8") as fh:
            sections.append(fh.read().strip())
        sections.append("")

    graph_records = []
    for f in sorted(glob.glob(os.path.join(BASE_DIR, "competitors", "results", "*.json"))):
        try:
            with open(f, encoding="utf-8") as fh:
                r = json.load(fh)
        except (OSError, ValueError):
            continue
        if r.get("workload") == "graph" and r.get("available") is not False and "seconds" in r:
            graph_records.append(r)

    if graph_records:
        groups: dict = {}
        for r in graph_records:
            key = (r.get("dataset") or "synthetic-10k", r.get("algorithm", "-"))
            groups.setdefault(key, []).append(r)
        for (ds, algo) in sorted(groups):
            sections.append(f"### Graph — `{ds}` ({algo})")
            sections.append("")
            sections.append("| Engine | Layer | Device | Load (s) | Execution (s) | Result size |")
            sections.append("|---|---|---|---|---|---|")
            for r in sorted(groups[(ds, algo)], key=lambda x: (x.get("seconds") or 0)):
                load = r.get("load_s", r.get("build_seconds", "-"))
                sections.append(
                    f"| {r.get('engine')} | {r.get('layer', '-')} | {r.get('device', 'cpu')} | "
                    f"{load} | {r.get('seconds')} | {r.get('result_size')} |"
                )
            sections.append("")
    else:
        sections.append("*No graph benchmark results found.*")
    sections.append("")

    # 3. SQL OLAP Benchmarks
    sections.append("## 3. SQL OLAP Performance (ClickBench Q0–Q9: BenoStreamDB vs DuckDB vs DataFusion)")
    sections.append("")
    sql_files = glob.glob(os.path.join(BASE_DIR, "sql", "results", "*.md"))
    if sql_files:
        for f in sorted(sql_files):
            sections.append(f"### Results from `{os.path.basename(f)}`")
            sections.append("")
            with open(f, encoding="utf-8") as fh:
                sections.append(fh.read().strip())
            sections.append("")
    else:
        sections.append("*No SQL benchmark results found.*")
    sections.append("")

    # 4. Lexical & Hybrid Search Benchmarks
    sections.append("## 4. Lexical & Hybrid Search Performance (BEIR SciFact: BM25 vs Tantivy & Hybrid RRF)")
    sections.append("")
    # Roll the per-engine JSON records (written by the BEIR harness) into one
    # table, so the lexical/hybrid numbers sit alongside the vector/graph/SQL
    # ones in the consolidated report.
    beir_json = glob.glob(os.path.join(BASE_DIR, "beir", "results", "*.json"))
    if beir_json:
        sections.append("### Rolled-up results (JSON)")
        sections.append("")
        sections.append(
            "| Engine | Backend | Workload | Dataset | Recall@k | nDCG@k | MRR@k | QPS | p50 (ms) | p99 (ms) | Build (s) | Index (MB) |"
        )
        sections.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for f in sorted(beir_json):
            try:
                with open(f, encoding="utf-8") as fh:
                    r = json.load(fh)
            except (OSError, ValueError):
                continue
            sections.append(
                f"| {r.get('engine')} | {r.get('backend', '-')} | {r.get('workload')} | "
                f"{r.get('dataset')} | "
                f"{r.get('recall_at_k')} | {r.get('ndcg_at_k')} | {r.get('mrr_at_k')} | "
                f"{r.get('qps')} | {r.get('p50_ms')} | {r.get('p99_ms')} | "
                f"{r.get('build_s')} | {r.get('index_mb')} |"
            )
        sections.append("")
    beir_files = glob.glob(os.path.join(BASE_DIR, "beir", "results", "*.md"))
    if beir_files:
        for f in sorted(beir_files):
            sections.append(f"### Results from `{os.path.basename(f)}`")
            sections.append("")
            with open(f, encoding="utf-8") as fh:
                sections.append(fh.read().strip())
            sections.append("")
    else:
        sections.append("*No BEIR benchmark results found.*")
    sections.append("")

    # 5. Production Workload & Concurrency
    sections.append("## 5. Production Workload & Concurrency Performance")
    sections.append("")

    conc_md = os.path.join(RESULTS_DIR, "production_concurrency.md")
    if os.path.exists(conc_md):
        with open(conc_md, encoding="utf-8") as fh:
            sections.append(fh.read().strip())
        sections.append("")

    recov_md = os.path.join(RESULTS_DIR, "production_recovery.md")
    if os.path.exists(recov_md):
        with open(recov_md, encoding="utf-8") as fh:
            sections.append(fh.read().strip())
        sections.append("")

    throttling_md = os.path.join(RESULTS_DIR, "production_throttling.md")
    if os.path.exists(throttling_md):
        with open(throttling_md, encoding="utf-8") as fh:
            sections.append(fh.read().strip())
        sections.append("")

    sections.append("### Tail Latency Under Background Maintenance (Compaction)")
    sections.append("")
    prod_json = os.path.join(RESULTS_DIR, "production_maintenance.json")
    if os.path.exists(prod_json):
        with open(prod_json, encoding="utf-8") as fh:
            data = json.load(fh)
        sections.append("| Metric | Value |")
        sections.append("|---|---|")
        for k, v in data.items():
            sections.append(f"| {k} | {v} |")
    else:
        sections.append("*No production maintenance results found.*")
    sections.append("")

    # Multi-Writer Concurrency Correctness (§7.7)
    mw_md = os.path.join(RESULTS_DIR, "production_concurrency_correctness.md")
    if os.path.exists(mw_md):
        with open(mw_md, encoding="utf-8") as fh:
            sections.append(fh.read().strip())
        sections.append("")

    # 6. Pathological Filters & Differential Oracle (§7.5)
    sections.append("## 6. Pathological Filters & Differential Oracle Benchmark")
    sections.append("")
    filters_md = os.path.join(RESULTS_DIR, "production_filters.md")
    if os.path.exists(filters_md):
        with open(filters_md, encoding="utf-8") as fh:
            sections.append(fh.read().strip())
        sections.append("")
    else:
        sections.append("*No pathological filter results found.*")
    sections.append("")

    # 7. Skewed Data & Power-Law Distribution Benchmark (§7.6)
    sections.append("## 7. Skewed Data & Power-Law Distribution Benchmark")
    sections.append("")
    skew_md = os.path.join(RESULTS_DIR, "production_skew.md")
    if os.path.exists(skew_md):
        with open(skew_md, encoding="utf-8") as fh:
            sections.append(fh.read().strip())
        sections.append("")
    else:
        sections.append("*No skewed data benchmark results found.*")
    sections.append("")

    # 8. Mixed Workload Soak & Endurance Benchmark (§7.8)
    sections.append("## 8. Mixed Workload Soak & Endurance Benchmark")
    sections.append("")
    soak_md = os.path.join(RESULTS_DIR, "production_soak.md")
    if os.path.exists(soak_md):
        with open(soak_md, encoding="utf-8") as fh:
            sections.append(fh.read().strip())
        sections.append("")
    else:
        sections.append("*No soak benchmark results found.*")
    sections.append("")

    # 9. Docker Competitor Matrix (single shared hardware envelope)
    #
    # Aggregate the per-engine JSON records (not the per-run `rollup.md`, which
    # only reflects the most recent invocation) so the report always shows the
    # full matrix — every engine, every dataset, and both CPU/GPU passes.
    sections.append("## 9. Docker Competitor Matrix (shared hardware envelope)")
    sections.append("")
    records = []
    for f in sorted(glob.glob(os.path.join(BASE_DIR, "competitors", "results", "*.json"))):
        try:
            with open(f, encoding="utf-8") as fh:
                r = json.load(fh)
        except (OSError, ValueError):
            continue
        if r.get("available") is False:
            continue
        records.append(r)

    vector = [r for r in records if "recall_at_k" in r]
    # Collapse accidental duplicates (the same engine/dataset/device measured by
    # more than one pass, e.g. a GPU-pass engine that fell back to CPU), keeping
    # the best run.
    _best: dict = {}
    for r in vector:
        key = (r.get("engine"), r.get("dataset"), r.get("device"))
        if key not in _best or (r.get("qps") or 0) > (_best[key].get("qps") or 0):
            _best[key] = r
    vector = list(_best.values())
    graph = [r for r in records if r.get("workload") == "graph" and "seconds" in r]
    sql = [r for r in records if r.get("workload") == "sql" and "seconds" in r]

    if vector:
        # One table per dataset with CPU and GPU rows mixed; the Backend column
        # says which the algorithm actually ran on.
        groups = {}
        for r in vector:
            groups.setdefault(r.get("dataset", "-"), []).append(r)
        for ds in sorted(groups):
            sections.append(f"### Vector — `{ds}`")
            sections.append("")
            sections.append(
                "| Engine | Index | Backend | Recall@k | QPS | p50 (ms) | p99 (ms) | Build (s) | Index (MB) |"
            )
            sections.append("|---|---|---|---|---|---|---|---|---|")
            for r in sorted(groups[ds], key=lambda x: -(x.get("qps") or 0)):
                iv = r.get("index")
                idx = r.get("index_type") or (iv.get("type") if isinstance(iv, dict) else iv) or "-"
                sections.append(
                    f"| {r.get('engine')} | {idx} | {r.get('device', 'cpu')} | "
                    f"{r.get('recall_at_k')} | {r.get('qps')} | "
                    f"{r.get('p50_ms')} | {r.get('p99_ms')} | {r.get('build_s')} | {r.get('index_mb')} |"
                )
            sections.append("")

    if graph:
        # Graph is covered in full (all engines, incl. Neo4j + GDS) by §2.
        sections.append("### Graph")
        sections.append("")
        sections.append("Graph results for every engine (incl. Neo4j + GDS) are in **§2**.")
        sections.append("")

    if sql:
        sections.append("### SQL")
        sections.append("")
        sections.append("| Engine | Dataset | Device | Seconds | Rows |")
        sections.append("|---|---|---|---|---|")
        for r in sorted(sql, key=lambda x: (x.get("dataset", ""), x.get("seconds") or 0)):
            sections.append(
                f"| {r.get('engine')} | {r.get('dataset', '-')} | {r.get('device', 'cpu')} | "
                f"{r.get('seconds')} | {r.get('rows')} |"
            )
        sections.append("")

    if not (vector or graph or sql):
        sections.append("*No Docker competitor matrix results found.*")
        sections.append("")

    # 10. Competitor configurations (frame of reference)
    sections.append("## 10. Competitor Configurations (frame of reference)")
    sections.append("")
    sections.append(
        "Every engine runs inside the same Docker envelope (`BENCH_CPUS`/`BENCH_MEM`, "
        "recorded in `hardware_profile.txt`). The table documents the index/engine "
        "setup and the measurement layer behind each number."
    )
    sections.append("")
    sections.append("| Workload | Engine | Configuration | Measurement layer |")
    sections.append("|---|---|---|---|")
    for wl, engine, config, layer in COMPETITOR_CONFIG:
        sections.append(f"| {wl} | {engine} | {config} | {layer} |")
    sections.append("")

    content = "\n".join(sections) + "\n"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Generated comprehensive report at {report_path}")


if __name__ == "__main__":
    generate_summary()
