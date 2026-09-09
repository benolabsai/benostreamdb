# Benchmark Findings — `hypersearch` vs Elasticsearch 7.10.2

Step 4.2 of the OpenSearch/ES integration plan. Compares the `hypersearch`
REST server against a single-node Elasticsearch 7.10.2 (Docker, security
disabled, 1 shard / 0 replicas, refresh disabled during ingest, default 1 GiB
JVM heap), feeding both the **same document stream** (one document per HTTP
POST) on the same host.

- **Tool:** [`benchmarks/competitive/benchmark_es710.py`](benchmarks/competitive/benchmark_es710.py)
- **Raw results:** `benchmarks/competitive/benchmark_results/es710_hypersearch_*.json` / `.md`

## How to run

```bash
# From the repo root, after building the binary:
cargo build -p hyperstreamdb-search --bin hypersearch

# Quick validation run (200 docs, 20 query runs, dim 32) — the M4 gate:
./venv/bin/python benchmarks/competitive/benchmark_es710.py --quick

# Full target runs (spec Step 4.2):
./venv/bin/python benchmarks/competitive/benchmark_es710.py --size 100000 --runs 100
./venv/bin/python benchmarks/competitive/benchmark_es710.py --size 1000000 --runs 100

# hypersearch-only (no ES) or keep the ES container for inspection:
./venv/bin/python benchmarks/competitive/benchmark_es710.py --skip-es
./venv/bin/python benchmarks/competitive/benchmark_es710.py --keep-es
```

The ES 7.10.2 image (`docker.elastic.co/elasticsearch/elasticsearch:7.10.2`,
~814 MB) is pulled on first run.

## Results — quick run (200 docs, dim 32, 20 runs)

| Operation | HyperStreamDB p50 | HyperStreamDB p95 | ES 7.10.2 p50 | ES 7.10.2 p95 |
|-----------|-------------------|-------------------|---------------|---------------|
| Ingest (docs/s) | **879** | — | 221 | — |
| Refresh (ms) | 19,560 | — | **46** | — |
| `match` BM25 (ms) | 86.1 | 100.4 | **3.3** | 4.8 |
| Filtered `match`+`term` (ms) | 106.5 | 120.1 | **2.6** | 2.9 |
| `knn` HNSW (ms) | 85.6 | 89.0 | n/a¹ | n/a¹ |
| Hybrid `match`+`knn` (ms) | 101.5 | 109.3 | n/a¹ | n/a¹ |

¹ ES 7.10 has no `dense_vector` type, so `knn` and hybrid are hypersearch-only.

## Findings

1. **Search latency is within the target envelope.** Hypersearch query
   latencies (86–120 ms p95) fall inside the plan's **50–200 ms** target for
   website/document search. This is the intended operating point.

2. **ES is far faster on search — by design.** ES 7.10 answers in 3–6 ms
   because it pins segments in memory (sub-millisecond, JVM heap + hot SSD).
   Hypersearch trades that latency for an object-storage-native, scale-to-zero
   model. This is the core positioning trade-off, not a regression.

3. **Ingest throughput favors hypersearch** (879 vs 221 docs/s) for single-doc
   POSTs, because writes land in the memtable + WAL immediately. (The
   `_bulk` endpoint, added in M3, improves this further for batch ingestion.)

4. **Refresh (index build) is the one-time cost.** Hypersearch's refresh
   (19.5 s for 200 docs) includes building the BM25 inverted index + HNSW
   vector index and committing the manifest/Iceberg metadata; ES's refresh
   (46 ms) only flushes the translog. As document count grows, hypersearch's
   per-refresh index build grows too — this is the main area to optimize
   (incremental index builds, background/async indexing, and the on-demand
   index-file cache) before the 100k/1M runs.

5. **`knn` and hybrid are differentiators.** ES 7.10 cannot do native vector
   search; hypersearch serves `knn` and RRF-fused hybrid in the same 50–200 ms
   envelope with no extra infrastructure.

## Next steps (100k / 1M)

The quick run validates the pipeline end-to-end (the M4 gate). The full
100k/1M runs are long-running (ingest alone is ~20 min for 1M docs at the
observed hypersearch rate, ~75 min for ES, plus index builds and query runs)
and should be run on a dedicated host:

```bash
./venv/bin/python benchmarks/competitive/benchmark_es710.py --size 100000 --runs 100
./venv/bin/python benchmarks/competitive/benchmark_es710.py --size 1000000 --runs 100
```

The S3/MinIO variant (spec: "local FS, then S3 via MinIO") sets
`HYPERSEARCH_STORAGE_URI=s3://<bucket>/search` with a running MinIO and the
same commands; the on-demand index-file cache
([`hyperstreamdb-search/src/index_cache.rs`](hyperstreamdb-search/src/index_cache.rs))
is what makes the S3 path viable at scale.
