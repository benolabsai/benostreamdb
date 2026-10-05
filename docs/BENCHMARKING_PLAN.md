# BenoStreamDB Benchmarking Plan

Status: working plan (exportable). Owner: BenoStreamDB core.
Companion docs: [`BENCHMARKING.md`](BENCHMARKING.md) (how to run the existing suite),
[`RESOURCE_LIMITS.md`](RESOURCE_LIMITS.md) (memory/CPU knobs),
[`CONCURRENCY.md`](CONCURRENCY.md) (writer/reader model).

This document is the **complete plan** for producing defensible, reproducible,
competitor-comparable benchmark numbers for a public announcement. It covers the
dataset suite, the competitor set, the harnesses, the resource-envelope rules,
the workload classes, the metrics captured, the reporting format, and the
methodology required to keep the numbers honest.

---

## 1. Goals and non-goals

**Goals**

1. Report an apples-to-apples **recall@k / QPS / p50 / p99** curve for vector
   search against the standard ANN-Benchmarks datasets, versus reference
   implementations run under the same core/RAM envelope.
2. Report lexical (BM25), hybrid (RRF), graph, and SQL/analytical performance
   against established baselines.
3. Report **production behaviour**, not just steady-state throughput: tail
   latency under concurrent maintenance (compaction/vacuum), recovery time after
   a crash, provider throttling resilience, and pathological filter/skew cases.
4. Make every number reproducible from a committed script + a recorded
   environment, so a third party can regenerate it.

**Non-goals**

- Winning every axis. Overlay indexes trade build cost and storage for query
  speed and reconstructibility; the report must state where we lose.
- Tuned-vs-untuned comparisons. A tuned competitor compared to an untuned
  BenoStreamDB (or vice versa) is not a benchmark.

---

## 2. Datasets (external, standard, citable)

All datasets are public and used by existing competitors, so results map onto
published numbers.

### 2.1 Vector (ANN-Benchmarks)

Source: `http://ann-benchmarks.com` (HDF5: `train`, `test`, `neighbors`).

| Dataset | Dim | Metric | Notes |
|---|---|---|---|
| `sift-128-euclidean` | 128 | L2 | The default; 1M train |
| `gist-960-euclidean` | 960 | L2 | High-dim, larger vectors |
| `fashion-mnist-784-euclidean` | 784 | L2 | Mid-size |
| `mnist-784-euclidean` | 784 | L2 | Mid-size |
| `glove-100-angular` | 100 | Cosine | NLP embeddings |
| `glove-200-angular` | 200 | Cosine | NLP embeddings |
| `nytimes-256-angular` | 256 | Cosine | Text |
| `lastfm-64-dot` | 64 | Inner product | Recommendation |

The harness downloads and caches these under `benchmarks/ann_benchmarks/data/`.
`--limit N` subsets the train set for smoke runs; exact ground truth is then
recomputed by brute force.

### 2.2 Lexical / hybrid (IR)

| Dataset | Use | Notes |
|---|---|---|
| BEIR subset (e.g. `scifact`, `nfcorpus`, `arguana`) | BM25 + hybrid retrieval | Standard IR eval; nDCG@10 / recall@k |
| MS MARCO passage (dev) | BM25 + hybrid at scale | Large; use a fixed passage slice |
| A synthetic keyword corpus | pathological filters, high-cardinality terms | Determinism |

### 2.3 Graph

| Dataset | Use | Notes |
|---|---|---|
| LDBC SNB (SF1/SF10) | traversal, shortest path, PageRank, communities | Standard graph benchmark |
| SNAP (e.g. `com-LiveJournal`, `roadNet-CA`) | scale + skew | Public, citable |
| Synthetic RMAT / power-law graphs | adversarial degree skew | Deterministic generator |

### 2.4 SQL / analytical

| Dataset | Use | Notes |
|---|---|---|
| ClickBench (hits) | scan/aggregate/filter throughput | Industry-standard |
| TPC-H (SF1/SF10) | joins, group-by, ordering | via DataFusion / Flight SQL |
| NYC TLC parquet | wide-table scans, partition pruning | Real-world |

---

## 3. Competitor set

Each competitor is run under the **same `--cores` / `--ram-gb` envelope** as the
BenoStreamDB run (see §5). Adapters live in `benchmarks/competitors/`.

### Tier 1 — vector (must-have for launch)

| Engine | Role | Notes |
|---|---|---|
| FAISS (HNSW/IVF/PQ) | the reference ANN library | `IndexHNSWFlat`, tuned `M`, `efConstruction`, `efSearch` |
| hnswlib | the reference HNSW implementation | same `M` / `ef_construction` / `ef` |
| pgvector | SQL-integrated ANN | HNSW + IVFFlat; `m`, `ef_construction`, `ef_search` |
| LanceDB | embedded / columnar ANN | HNSW/IVF_PQ |

### Tier 1 — lexical / hybrid

| Engine | Role |
|---|---|
| Elasticsearch / OpenSearch | BM25 + kNN + RRF hybrid |
| Tantivy | embedded BM25 reference |

### Tier 1 — graph

| Engine | Role |
|---|---|
| Neo4j (GDS) | traversal + centrality reference |
| NetworkX | single-threaded reference (correctness + slow baseline) |
| GraphX / cuGraph | distributed / GPU scale reference |

### Tier 1 — SQL

| Engine | Role |
|---|---|
| DuckDB | embedded OLAP reference |
| Trino | distributed SQL |
| ClickHouse | columnar aggregation |
| DataFusion | same engine family (isolates our overhead) |

### Tier 2 — storage / table format

| Engine | Role |
|---|---|
| Iceberg + Spark/Trino | table-format interop baseline |
| Delta Lake | format-cost comparison |

**Rule:** every competitor row in a report must name the version, the exact
config, and the command used.

---

## 4. Harnesses

### 4.1 Vector: `benchmarks/ann_benchmarks/run.py`

- Loads an ANN-Benchmarks HDF5 dataset into a BenoStreamDB table with a vector
  index, runs `--queries` test queries, and reports standard metrics.
- Parameter mapping (so the competitor config is matched exactly):

  | Harness flag | Meaning | Maps to |
  |---|---|---|
  | `--m` | max neighbours per node | HNSW `M` (`complexity`) |
  | `--ef-construction` | build beam width | HNSW `ef_construction` (`quality`) |
  | `--ef-search` | query beam width | HNSW `efSearch` |
  | `--metric` | distance | `l2` / `cosine` / `inner_product` |
  | `--index` | algorithm | `hnsw`, `hnsw_tq8`, `hnsw_tq4`, `hnsw_pq` |

- Two latency views are reported:
  - **End-to-end** (`to_arrow(vector_filter=...)`): index search + row fetch.
  - **Pure index** (`vector_search_scored`): index only, no Parquet payload.
    This is the number directly comparable to ANN-Benchmarks.

Example:

```bash
python benchmarks/ann_benchmarks/run.py \
  --dataset sift-128-euclidean \
  --cores 8 --ram-gb 16 \
  --m 32 --ef-construction 200 --ef-search 200 \
  --queries 1000 --out benchmarks/ann_benchmarks/results/sift_tq8.md
```

### 4.2 Competitor adapters: `benchmarks/competitors/`

[`run_competitor.py`](../benchmarks/competitors/run_competitor.py) is the single
runner; each engine is an adapter accepting the same `--cores`, `--ram-gb`,
`--m`, `--ef-construction`, `--ef-search`, `--metric`, `--dataset`, `--queries`
arguments and emitting the JSON schema of §6. Implemented so far:

- **Vector:** `faiss`, `hnswlib` (embedded); `pgvector` (`--dsn`), `lancedb`
  (`--path`), `opensearch` (`--host`).
- **Graph:** `networkx` (reference), `neo4j` GDS (`--uri`), `cugraph` (GPU),
  `benostreamdb`.
- **SQL:** `duckdb`, `datafusion` (embedded); `clickhouse`, `trino` (server);
  `benostreamdb`.
- **Storage / table format:** Iceberg round-trip (Spark/Trino) + Delta Lake —
  `benchmarks/iceberg_roundtrip/run.py`.

Adapters report `{"available": false, "error": ...}` when the client library or
server is missing, so a partial environment still produces a clear report. See
[`benchmarks/competitors/README.md`](../benchmarks/competitors/README.md).

### 4.3 Production workload harness (to build)

Drives the classes in §7 against the Flight SQL gateway and the Python client,
with the Prometheus endpoint (`/metrics`) scraped for the duration.

---

## 5. Resource-envelope rules

The engine and the competitor must be given the **same cores and RAM**. The
harness sets the BenoStreamDB knobs from `--cores` / `--ram-gb`:

| Env var | Value | Purpose |
|---|---|---|
| `RAYON_NUM_THREADS` | `cores` | data-parallel CPU |
| `BSDB_MAX_CONCURRENCY` | `cores` | query concurrency cap |
| `BSDB_INDEX_BUILD_CONCURRENCY` | `cores` | index build fan-out |
| `BSDB_MAX_INGEST_RAM_GB` | `0.8 * ram_gb` | ingest memory ceiling |
| `BSDB_INGEST_MEMORY_BUDGET_GB` | `0.8 * ram_gb` | ingest working set |
| `BSDB_DATAFUSION_MEMORY_GB` | `0.5 * ram_gb` | SQL operator memory |

For the competitor, enforce the same envelope with the process's cgroup, or
`docker run --cpus=<cores> --memory=<ram_gb>g`, or the library's own pool size.

**Must be recorded per run:** CPU model, core count, RAM, OS, disk type, and
whether the data is warm (page cache) or cold.

**Containerized runs (one profile for everyone).**
[`benchmarks/competitors/docker_bench.sh`](../benchmarks/competitors/docker_bench.sh)
applies a single envelope (`--cpus` / `--mem`) to **every** participant — the
server engines *and* the runner that executes each client, including
BenoStreamDB — so no one gets more resources than another. It records the host
hardware profile (CPU model, cores, RAM, Docker/Compose versions, GPU list) to
`results/hardware_profile.txt`, and each result's `env` carries the
container-visible cores/RAM.

- **CPU is the baseline.** All engines run on CPU under the CPU envelope.
- **GPU is opt-in per engine.** `--gpu` (or `--both`) adds a GPU pass
  (`docker-compose.bench.gpu.yml`, `gpus: all`) restricted to GPU-capable
  engines (FAISS, BenoStreamDB, and cuGraph once installed). CPU-only engines
  (pgvector, LanceDB, OpenSearch, Neo4j CPU GDS, DuckDB) are excluded.
- Results are tagged `{engine}_{dataset}_{device}.json` with `device=cpu|gpu`,
  and `rollup.md` has a Device column.

---

## 6. Metrics and output schema

Every run emits a machine-readable record:

```json
{
  "engine": "benostreamdb",
  "engine_version": "0.12.0",
  "dataset": "sift-128-euclidean",
  "workload": "vector_ann",
  "index": { "type": "hnsw_tq8", "m": 32, "ef_construction": 200, "ef_search": 200, "metric": "l2" },
  "envelope": { "cores": 8, "ram_gb": 16, "warm_cache": true },
  "build": { "seconds": 123.4, "index_mb": 512.0, "peak_rss_mb": 3100 },
  "query": {
    "count": 1000,
    "recall_at_k": 0.982,
    "k": 10,
    "qps": 812.3,
    "latency_ms": { "p50": 1.10, "p90": 1.60, "p99": 3.40, "max": 9.80 }
  },
  "env": { "cpu": "...", "os": "...", "disk": "nvme" },
  "command": "python benchmarks/ann_benchmarks/run.py ...",
  "notes": ""
}
```

Standard metrics:

- **Quality:** recall@k (ANN), nDCG@10 / recall@k (IR), result-set equality for
  graph/SQL correctness checks.
- **Throughput:** QPS (single-thread and at a fixed concurrency).
- **Latency:** p50/p90/p99/max.
- **Cost:** build seconds, index size on disk, peak RSS.
- **Stability:** RSS growth over the run, error count, recovery time.

Reports are emitted as Markdown into `benchmarks/**/results/` (and
`benchmarks/competitive/benchmark_results/`) plus the JSON above.

---

## 7. Production workload classes

Steady-state QPS is not enough for a launch. Each class below is a separate,
scripted scenario with its own pass/fail criteria.

1. **Concurrent queries** — N clients at fixed concurrency; report QPS and
   p99/p999; assert no errors and bounded RSS.
2. **p99 under maintenance** — saturate readers while compaction and vacuum run;
   assert p99 stays within a stated multiple of the quiescent p99 and that no
   committed row is lost or duplicated (see the multi-writer isolation probes).
3. **Recovery time** — SIGKILL the process at injected crash points
   (`src/core/fault_injection.rs`), restart, and measure time-to-first-correct-read
   and any data loss. Must be zero loss (WAL + atomic manifest commit).
4. **Object-store throttling** — run against a store that fails/slows a fraction
   of requests (`FaultyStore` pattern in `tests/test_multi_writer_concurrency.rs`);
   assert no corruption and bounded latency inflation.
5. **Pathological filters** — selective filters that prune almost nothing,
   high-cardinality `IN` lists, `NOT`/wildcard abuse; assert index-vs-full-scan
   agreement (differential oracle) and report the slowdown.
6. **Skewed data** — power-law degree distributions (graph) and hot-key
   filter distributions; report tail latency.
7. **Concurrency correctness** — two+ writers on a shared store; assert exact
   row counts (no lost updates, no duplicate rows) under insert/delete/compact.
8. **Mixed workload soak** — long-running, memory-bounded CPU churn with an RSS
   ceiling; emits `[soak-stats]` and `soak-report.md`.

---

## 8. Methodology and reproducibility

1. **Warmup.** Discard a warmup phase before measuring; ANN-Benchmarks and the
   harness must report steady-state numbers, and state whether the index is
   resident (warm) or being re-read (cold).
2. **Build the extension you measure.** The Python numbers come from the native
   extension (`python/benostreamdb/benostreamdb.abi3.so`). Rebuild it with
   `maturin develop --release` before any benchmark run; a stale `.so` invalidates
   the result. (This bit us once — see §11.)
3. **Single source of truth for versions.** Core, flight, search, dbt, Spark, and
   Trino versions derive from the core `Cargo.toml`; record the resolved version.
4. **Determinism.** Fixed RNG seeds; fixed query subsets; identical `k`.
5. **Isolation.** Close other heavy processes; pin CPU affinity where possible;
   state the cgroup/limits.
6. **Repeat and report variance.** Run each configuration at least 3 times and
   report the median plus spread.
7. **Commit the exact command** with every result.

---

## 9. Reporting format

- One Markdown file per (engine, dataset, workload) in a `results/` directory.
- A roll-up table per workload family:

  | Engine | Version | Recall@10 | QPS | p50 (ms) | p99 (ms) | Build (s) | Index (MB) | RSS (MB) |
  |---|---|---|---|---|---|---|---|---|

- A short "method + caveats" section on every report: envelope, warm/cold, and
  anything that could bias the comparison.
- Production classes get a separate report with pass/fail per criterion.

---

## 10. Automation and CI

- **Local pre-release soak:** `scripts/pre_release_soak.sh <minutes>` (default
  30; `--quick` = 1 min) runs differential-oracle, crash-injection, a long
  memory-bounded soak, and the multi-writer soak, and writes `soak-report.md`.
- **CI gate:** `.github/workflows/release.yml` `soak-gate` job runs the same
  correctness gates before a release; artifacts include `soak-report.md`.
- **Coverage:** `.github/workflows/coverage.yml` enforces a line-coverage floor.
- **Benchmark jobs** (to add): scheduled vector/lexical/graph/SQL runs on a
  fixed runner spec, publishing JSON + Markdown to `results/` and to the docs site.

---

## 11. Known findings and current status

### Resolved: the ~250 ms → ~205 ms → ~20 ms → sub-ms ANN latency saga

The reported per-query latency was **not** the search algorithm, the manifest, or
Parquet I/O. Three compounding defects, found and fixed:

1. **Stale native extension.** The first measurements used a
   `benostreamdb.abi3.so` built ~1 hour before the sources. Always rebuild with
   `maturin develop --release` before a run (§8.2). This alone took ~205 ms → ~20 ms.
2. **The declared vector index was never built.** `add_index` on an empty table
   registered the config, but the incoming column was a **`list<double>`**
   (`float64`, e.g. from `pandas`). `build_vector_index`
   (`src/core/index/build_vector.rs`) silently `return`ed for any inner type
   other than `Float32`, so **no index files were written** and every query fell
   back to a brute-force scan. Fixes:
   - `build_vector_index` now accepts `Float64` and narrows it to the engine's
     canonical `Float32` (the engine indexes in `f32` throughout — HNSW/IVF/PQ/
     TurboQuant, distances, and the temp vector file), with a warning.
   - The write path (`src/core/table/write.rs`) coerces any column with a
     **declared vector index** to `Float32`, and the schema-reconcile step now
     casts mismatched columns to the target type, so data flows into an index
     declared before any data existed. Non-vector (e.g. string/BM25) indexes are
     unaffected — the coercion is dtype-guarded to float lists only.
3. **The harness never waited for the background build.**
   `benchmarks/ann_benchmarks/run.py` now calls `table.wait_for_background_tasks()`
   after `commit()` (counted in `build_s`) and **fails loudly** if zero index
   files were produced, so a silent full-scan fallback can never be published.

Also applied: `Table::execute_vector_search_as_scored` sets
`.with_columns(Some(vec![]))` for the explicit no-payload path.

### Measured result (SIFT 20k subset, `hnsw_tq8`, M=16 / efc=200 / ef=200)

| Metric | Value |
|---|---|
| Index files | 57 |
| Index size | 26.1 MB |
| Build time | 1.2 s |
| recall@10 | 0.950 |
| QPS (end-to-end) | 1097.8 |
| p50 / p99 | 0.70 ms / 1.17 ms |
| QPS (pure index) | 1146.0 |
| pure p50 / p99 | 0.87 ms / 1.06 ms |

Artifact: `benchmarks/ann_benchmarks/results/sift_20k_tq8.md`. Correctness is
still guaranteed by the differential oracle (indexed == full scan), which passes
for HNSW/IVF/PQ/TurboQuant.

### Open

- The full unfiltered scan (`read_async(None,None,None)`) is ~16 ms for the 20k
  table; verify against release builds and decide whether Parquet decode is the
  floor.
- The `float64` column is now indexed correctly, but the stored Parquet keeps
  `list<double>`; declaring vector columns as `float32` at the source avoids the
  per-build narrowing (and future read-path casts).
- A pre-existing unbalanced-delimiter bug in
  `src/core/index/hnsw_rs/arrow_hnsw.rs` (unclosed closure in the search loop)
  was fixed to unblock the release build.

---

## 12. Roadmap

| # | Item | State |
|---|---|---|
| 1 | ANN-Benchmarks harness (`benchmarks/ann_benchmarks/run.py`) | done |
| 2 | Resource-envelope `--cores` / `--ram-gb` | done |
| 3 | Score-only fast path | done |
| 4 | Fix vector-index-never-built (float64) + harness wait/verify | done |
| 5 | `benchmarks/competitors/` Tier-1 adapters (FAISS, hnswlib, pgvector, LanceDB) | done |
| 6 | Lexical/hybrid harnesses (BEIR, Elasticsearch/OpenSearch, Tantivy) | done |
| 7 | Graph harnesses (LDBC, Neo4j GDS, NetworkX, cuGraph, BenoStreamDB) | done (`benchmarks/graph/run.py`) |
| 8 | SQL harnesses (ClickBench, DuckDB, DataFusion, ClickHouse, BenoStreamDB) | done (`benchmarks/sql/run.py`) |
| 9 | Production workload harness (concurrent, maintenance, recovery, throttling, filters, skew, concurrency correctness, soak) | done |
| 10 | Scheduled CI benchmark jobs + results site | done (`.github/workflows/benchmarks.yml`) |
| 11 | Trino SQL competitor (`_sql_trino` + `trino` compose service) | done |
| 12 | Iceberg round-trip (Spark/Trino) + Delta Lake (`benchmarks/iceberg_roundtrip/run.py`) | done |
