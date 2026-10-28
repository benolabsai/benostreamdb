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

import argparse
import glob
import json
import os
import platform
import re
import sys
import time

try:
    import session_provenance
except ImportError:  # imported as `benchmarks.generate_summary`
    from benchmarks import session_provenance  # type: ignore[no-redef]

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


def _eng_label(engine) -> str:
    """Bold BenoStreamDB rows so they stand out in every table.

    The vector quantization suffix (`_tq8`/`_pq`) is dropped from the
    engine name because the Index Precision column already carries it, so the
    quantized variants read as plain `benostreamdb`.
    """
    name = str(engine)
    for suffix in ("_tq8", "_pq"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return f"**{name}**" if name.startswith("benostreamdb") else name


def _quant_label(idx, default: str = "-") -> str:
    """The vector *quantization* behind an index: f32 / tq8 / pq.

    The Index Precision column reports the precision, not the algorithm name,
    so a full-precision HNSW index reads `f32` and the TurboQuant/PQ variants
    read `tq8`/`pq`. Lexical-only indexes (BM25) have no vector component
    and read `-`; a hybrid `bm25+hnsw` index is full-precision `f32`. When the
    record carries no index information the caller picks the fallback: `f32`
    in the vector tables (competitors run full precision) and `-` in the
    lexical tables.
    """
    s = str(idx or "").strip().lower()
    if s in ("", "-", "none"):
        return default
    if "tq8" in s:
        return "tq8"
    if "pq" in s:
        return "pq"
    # Lexical-only (BM25) has no vector index; hybrid (bm25+hnsw) is f32.
    if "hnsw" in s or "vector" in s or "dense" in s:
        return "f32"
    if "bm25" in s or "lexical" in s or "sparse" in s:
        return "-"
    return "f32"


_SEP_ROW = re.compile(r"^\s*\|[\s:\-|]+\|\s*$")


def _sort_tables_by_qps(text: str) -> str:
    """Sort every Markdown table that has a QPS column by QPS descending.

    Applied to the whole report so tables produced by the sibling harnesses
    (BEIR lexical/hybrid, SQL, production) are ordered the same way as the
    generated §2/§9 tables. When a table also has a Dataset column the rows
    are grouped by dataset first (ascending) and then by QPS descending, so
    multi-dataset tables (e.g. §4 BEIR) read dataset-by-dataset. Tables
    without a QPS column are left untouched.
    """
    lines = text.split("\n")
    out: list = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if line.lstrip().startswith("|") and i + 1 < n and _SEP_ROW.match(lines[i + 1]):
            header, sep = lines[i], lines[i + 1]
            cols = [c.strip().strip("*") for c in header.strip().strip("|").split("|")]
            qps_idx = next((k for k, c in enumerate(cols) if c.upper() == "QPS"), None)
            ds_idx = next((k for k, c in enumerate(cols) if c.upper() == "DATASET"), None)
            j = i + 2
            while j < n and lines[j].lstrip().startswith("|"):
                j += 1
            rows = lines[i + 2:j]
            if qps_idx is not None and rows:
                def _cells(row: str) -> list:
                    return [c.strip().replace("*", "") for c in row.strip().strip("|").split("|")]

                def _q(row: str) -> float:
                    try:
                        return float(_cells(row)[qps_idx])
                    except (IndexError, ValueError):
                        return float("-inf")

                def _ds(row: str) -> str:
                    if ds_idx is None:
                        return ""
                    try:
                        return _cells(row)[ds_idx]
                    except IndexError:
                        return ""

                rows = sorted(rows, key=lambda r: (_ds(r), -_q(r)))
            out.extend([header, sep, *rows])
            i = j
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


def _dataset_from_filename(path: str) -> str | None:
    """Recover the dataset name from a competitor result filename.

    The SQL harness historically dropped the ``--dataset`` it was given, so the
    JSON records carry no dataset. The filename still encodes it as
    ``<engine>_<dataset>_<device>.json``; the synthetic ClickBench table is
    written as either ``sql`` or ``synth_500000`` and normalises to the latter.
    """
    base = os.path.basename(path)
    m = re.match(r"^[a-z0-9]+_(.+)_(cpu|gpu)\.json$", base)
    if not m:
        return None
    tok = m.group(1)
    return "synth_500000" if tok in ("sql", "synth_500000") else tok


def _date_range(patterns):
    """Oldest–newest mtime across a set of artifact globs, as UTC dates."""
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    stamps = [os.path.getmtime(f) for f in files if os.path.isfile(f)]
    if not stamps:
        return "not found"
    fmt = lambda t: time.strftime("%Y-%m-%d", time.gmtime(t))
    lo, hi = fmt(min(stamps)), fmt(max(stamps))
    return lo if lo == hi else f"{lo} – {hi}"


def _session_header(ledger: dict, primary: str | None, mismatches: list[tuple[str, str]]) -> list[str]:
    """Machine state, calibration and session-integrity lines for the report."""
    lines: list[str] = []
    sessions = ledger.get("sessions", {})
    state = (sessions.get(primary, {}) or {}).get("fingerprint")

    lines += ["## Machine state (part of the measurement)", ""]
    if state:
        lines.append(
            f"Session `{primary}` — every stamped section below was measured on this "
            "host in this state."
        )
        lines.append("")
        lines += [f"- {entry}" for entry in session_provenance.describe_fingerprint(state)]
    else:
        lines.append(
            "No session was recorded for these artifacts: they predate provenance "
            "stamping, so comparability rests on the mtime table above. Re-run the "
            "full sweep to produce an attributable report."
        )
    lines.append("")

    calibrations = (sessions.get(primary, {}) or {}).get("calibrations", [])
    lines += ["### Calibration (drift normalizers)", ""]
    if calibrations:
        rows = session_provenance.describe_calibration_pair(calibrations)
        lines += ["| Probe | First | Last |", "|---|---|---|"]
        lines += [f"| {name} | {before} | {after} |" for name, before, after in rows]
        lines.append("")
        lines.append(
            f"Probes are cheap and machine-state sensitive, not engine benchmarks: "
            f"they move when DIMM timings or CPU voltage/frequency limits change, "
            f"which is how a future run's drift can be separated from a code change. "
            f"{len(calibrations)} sample(s) taken across this sweep."
        )
    else:
        lines.append(
            "Not recorded. Calibration is captured by "
            "`benchmarks/session_provenance.py capture --section <name>`."
        )
    lines.append("")

    lines += ["### Session integrity", ""]
    if primary is None:
        lines.append(
            "- No stamps: sections are compared by artifact mtime only. Treat the "
            "numbers as a historical snapshot, not as a baseline for a new run."
        )
    elif mismatches:
        lines.append(
            "- **This table mixes machine states.** The following sections were "
            "measured in a different session and must be re-measured before any "
            "cross-run comparison:"
        )
        lines += [f"  - `{name}` (session `{session}`)" for name, session in mismatches]
    else:
        lines.append("- All stamped sections were measured in the same session.")
    lines += [
        "",
        "- **Rule.** Numbers are comparable *within* one session. RAM timings, CPU "
        "voltage/frequency limits and the kernel move every engine in the "
        "comparison at once — competitor rows included — while the GPU section "
        "does not move at all. So a partial re-run cannot be spliced into this "
        "report: re-run the whole sweep, or leave it alone.",
        "",
    ]
    return lines


def generate_summary(allow_mixed_sessions: bool = False, ledger_path: str | None = None) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_path = os.path.join(RESULTS_DIR, "BENCHMARK_REPORT.md")

    ledger = session_provenance.load_ledger(ledger_path)
    primary, mismatches = session_provenance.verify(ledger)
    if mismatches and not allow_mixed_sessions:
        print(
            "refusing to generate a report from sections measured in different "
            "machine states:",
            file=sys.stderr,
        )
        for name, session in mismatches:
            print(f"  {name}: session {session}", file=sys.stderr)
        print(
            f"primary session: {primary}\n"
            "Re-run the affected sections in one sitting, or pass "
            "--allow-mixed-sessions to emit a historical report that labels them.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    engine_version = (
        (ledger.get("sessions", {}).get(primary, {}) or {})
        .get("fingerprint", {})
        .get("engine_version")
    ) or session_provenance._engine_version()

    sections = [
        "# BenoStreamDB Comprehensive Benchmark Report",
        "",
        f"- **Generated At**: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        f"- **Engine version**: {engine_version or 'unknown'}",
        f"- **Platform**: {platform.platform()}",
        f"- **Python**: {platform.python_version()}",
        f"- **Session**: {primary or 'unrecorded'}",
        "",
        "> **Read this first.** The sections were measured at different times. The",
        "> table below records each group's artifact mtimes; do not read the",
        "> generated-at timestamp as the measurement date for every section.",
        "> Numbers are only comparable **within one machine state** — see "
        "“Machine state” and “Session integrity” below.",
        "",
        "## Data provenance & scope",
        "",
        "| Section | Scope | Artifacts last modified (UTC) |",
        "|---|---|---|",
        f"| §1b GPU cdist | 4 shapes + 1 dual-GPU split | {_date_range([os.path.join(RESULTS_DIR, 'gpu_cdist.json')])} |",
        f"| §2 Graph | 10k–500k edges, CPU-only engines | {_date_range([os.path.join(BASE_DIR, 'graph', 'results', '*')])} |",
        f"| §3 SQL, §6 filters | 200k–500k rows | {_date_range([os.path.join(BASE_DIR, 'sql', 'results', '*'), os.path.join(RESULTS_DIR, 'production_filters.*')])} |",
        f"| §4 BEIR | 3.6k–8.7k documents | {_date_range([os.path.join(BASE_DIR, 'beir', 'results', '*')])} |",
        f"| §5, §7, §8 production / resilience / soak | 8–32 threads; 60 s soak | {_date_range([os.path.join(RESULTS_DIR, 'production_*')])} |",
        f"| §9 vector matrix | 20k vectors, 500 queries; competitors CPU-only | {_date_range([os.path.join(BASE_DIR, 'competitors', 'results', '*.json')])} |",
        "",
        *_session_header(ledger, primary, mismatches),
        "**Scope and limitations**",
        "",
        "- Single host (AMD Ryzen 9 5900XT, 32 cores / 128 GB, 2× RTX 5070 Ti); no",
        "  multi-node or multi-client network measurements.",
        "- Vector benchmarks cap the train set at **20k** vectors and 500 queries;",
        "  they do not characterize million-vector indexes.",
        "- SQL/filter benchmarks use **200k–500k rows**; graph benchmarks use up to",
        "  **500k edges**.",
        "- The endurance soak is **60 seconds**; it validates invariants, not",
        "  multi-hour stability.",
        "- Competitor rows are CPU-only. BenoStreamDB vector rows are backend-pinned",
        "  (`cpu` = scalar CPU, `cuda` = CUDA device 0).",
        "- The GPU kernel numbers in §1b are a microbenchmark of `bsdb.cdist`; they",
        "  do not imply GPU speedups for HNSW/IVF search, which is CPU-bound at",
        "  these sizes.",
        "",
        "---",
        "",
    ]

    # 1. Vector ANN Benchmarks
    #
    # These are BenoStreamDB's *own* index variants (HNSW / TurboQuant TQ8 /
    # PQ) with no competitor, so they are not a head-to-head comparison — the
    # competitor matrix is §9. Keep only a pointer so the comparison report does
    # not duplicate raw self-benchmarks.
    sections.append("## 1. Vector ANN Performance (BenoStreamDB internal characterization)")
    sections.append("")
    sections.append(
        "Index-variant characterization for BenoStreamDB itself (HNSW, TurboQuant "
        "`hnsw_tq8`, IVF-PQ) with **no competitor**. Head-to-head vector "
        "comparison is in **§9**; raw results are under "
        "`benchmarks/competitors/results/`."
    )
    sections.append("")
    sections.append(
        "**Ground truth.** The HDF5 `neighbors` field is only valid for the full "
        "train set, so the 20k-vector runs recompute exact neighbours by brute "
        "force. That recomputation now uses the dataset metric (L2, cosine, or "
        "inner product). The previous report used L2 for every dataset, which "
        "inverted the expected order on angular data; with metric-correct truth "
        "recall is monotonic (`f32` > `tq8` > `pq`) on every supported dataset. "
        "`hnsw_pq` is unsupported on `glove-100`/`lastfm` (no result), and its "
        "`glove-200` result collapses to recall ≈0.007 — treat that row as a "
        "configuration defect, not a quality measurement."
    )
    sections.append("")
    sections.append("### TurboQuant trade-off (pros and cons)")
    sections.append("")
    sections.append(
        "- **Pro — smaller index / less memory.** TurboQuant quantizes the stored "
        "vectors: `hnsw_tq8` (8-bit) is ≈4× smaller than float32, "
        "so more vectors fit per node and the working set "
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
        "- **Default.** Full-precision `hnsw` is the default; `hnsw_tq8`/"
        "`hnsw_pq` are opt-in for workloads that will trade recall for size/latency. "
        "§9 reports the `benostreamdb` engine once per precision, with the "
        "**Index Precision** column (`f32`/`tq8`/`pq`) making the trade-off "
        "explicit; holding several precisions on one column and selecting per query "
        "is on the roadmap (Theme 6)."
    )
    sections.append("")

    # 1b. GPU batched-distance kernel
    #
    # The ANN tables above are index-bound (HNSW/IVF build and search are
    # CPU-bound at these sizes). The 2D-tiled CUDA kernels are reached through
    # `bsdb.cdist` and k-means, so report them from the dedicated benchmark.
    cdist_path = os.path.join(RESULTS_DIR, "gpu_cdist.json")
    if os.path.exists(cdist_path):
        try:
            with open(cdist_path, encoding="utf-8") as fh:
                cdist = json.load(fh)
        except (OSError, ValueError):
            cdist = None
        if cdist:
            gpus = ", ".join(cdist.get("gpu") or ["none"])
            sections.append("### GPU distance kernel (`bsdb.cdist`)")
            sections.append("")
            sections.append(
                "The vector tables below measure index build/search, which is CPU-bound "
                "at 20k vectors. The tiled CUDA kernels are reached through `bsdb.cdist` "
                f"(and k-means). GPU: {gpus}; measured {cdist.get('generated_at')}, "
                f"Python {cdist.get('python')}."
            )
            sections.append("")
            sections.append(
                "| Shape | Metric | CPU (ms) | CPU Mdist/s | CUDA (ms) | CUDA Mdist/s | Speedup |"
            )
            sections.append("|---|---|---|---|---|---|---|")
            for r in cdist.get("cases", []):
                sections.append(
                    f"| {r['queries']}x{r['vectors']}x{r['dim']} | {r['metric']} | "
                    f"{r['cpu_ms']} | {r['cpu_mdist_s']} | {r.get('cuda_ms', '-')} | "
                    f"{r.get('cuda_mdist_s', '-')} | {r.get('speedup', '-')}x |"
                )
            sections.append("")
            dual = cdist.get("dual_gpu")
            if dual:
                sections.append(
                    f"Two-GPU split of {dual['queries']}x{dual['vectors']}x{dual['dim']}: "
                    f"{dual['single_ms']} ms on one device vs {dual['dual_ms']} ms across "
                    f"two ({dual['speedup']}x). Both threads release the GIL around the "
                    "kernel. Larger shapes are transfer-bound (the full MxN matrix is "
                    "copied back), so the dual-GPU gain depends on the output size."
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
                    f"| {_eng_label(r.get('engine'))} | {r.get('layer', '-')} | {r.get('device', 'cpu')} | "
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
            "| Engine | Index Precision | Backend | Workload | Dataset | Recall@k | nDCG@k | MRR@k | QPS | p50 (ms) | p99 (ms) | Build (s) | Index (MB) |"
        )
        sections.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for f in sorted(beir_json):
            try:
                with open(f, encoding="utf-8") as fh:
                    r = json.load(fh)
            except (OSError, ValueError):
                continue
            sections.append(
                f"| {_eng_label(r.get('engine'))} | {_quant_label(r.get('index'))} | {r.get('backend', '-')} | "
                f"{r.get('workload')} | "
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
    sections.append(
        "BenoStreamDB vector rows pin the requested backend with "
        "`Device(...).activate()`: `cpu` is the scalar CPU path and `cuda` is CUDA "
        "device 0. Competitor rows are CPU-only. Index build/search is CPU-bound at "
        "these sizes, so the `cuda` rows are not faster; see §1b for the GPU kernel."
    )
    sections.append("")
    sections.append(
        "**Recall comparability.** All engines (BenoStreamDB and external competitors) "
        "are evaluated against metric-correct ground truth computed via brute force using the "
        "dataset's declared metric (cosine distance for `glove-*` and `nytimes-256`, Euclidean L2 "
        "for `sift` and `mnist`, and inner product for `lastfm`), ensuring fair, direct "
        "head-to-head comparability across all vector indexes."
    )
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
        # Milvus was misconfigured in this pass (3-4 QPS, ~200 ms p50 across
        # every dataset); publishing it would present a broken client as a
        # competitor result. Raw files remain under competitors/results/.
        if r.get("engine") == "milvus":
            continue
        # The SQL harness historically dropped the dataset name; recover it from
        # the filename so the SQL table can group rows by dataset.
        if r.get("workload") == "sql" and not r.get("dataset"):
            r["dataset"] = _dataset_from_filename(f)
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
    # Collapse duplicate SQL runs (e.g. the same synthetic table measured as both
    # `sql` and `synth_500000`), keeping the fastest run per engine/dataset/device.
    _best_sql: dict = {}
    for r in sql:
        key = (r.get("engine"), r.get("dataset"), r.get("device"))
        if key not in _best_sql or (r.get("seconds") or 1e9) < (_best_sql[key].get("seconds") or 1e9):
            _best_sql[key] = r
    sql = list(_best_sql.values())

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
                "| Engine | Index Precision | Backend | Recall@k | QPS | p50 (ms) | p99 (ms) | Build (s) | Index (MB) |"
            )
            sections.append("|---|---|---|---|---|---|---|---|---|")
            for r in sorted(groups[ds], key=lambda x: -(x.get("qps") or 0)):
                iv = r.get("index")
                raw = r.get("index_type") or (iv.get("type") if isinstance(iv, dict) else iv) or "-"
                idx = _quant_label(raw, default="f32")
                sections.append(
                    f"| {_eng_label(r.get('engine'))} | {idx} | {r.get('device', 'cpu')} | "
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
                f"| {_eng_label(r.get('engine'))} | {r.get('dataset', '-')} | {r.get('device', 'cpu')} | "
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

    content = _sort_tables_by_qps("\n".join(sections)) + "\n"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Generated comprehensive report at {report_path}")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build benchmarks/results/BENCHMARK_REPORT.md from the result artifacts."
    )
    parser.add_argument(
        "--allow-mixed-sessions",
        action="store_true",
        help=(
            "emit the report even when sections were measured in different machine "
            "states (historical reports; the sections are labelled)"
        ),
    )
    parser.add_argument(
        "--ledger",
        default=None,
        help="session ledger path (default: benchmarks/results/session.json)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    arguments = _parse_args(sys.argv[1:])
    generate_summary(
        allow_mixed_sessions=arguments.allow_mixed_sessions,
        ledger_path=arguments.ledger,
    )
