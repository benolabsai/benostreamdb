# OpenSearch 2.11 vs BenoStreamDB — REST API Benchmark

**Generated:** 2026-09-12T14:28:24  
**Host:** AMD Ryzen 9 5900XT 16-Core Processor (Linux 7.0.0-31-generic)  
**ES:** skipped (build `n/a`, Docker, single-node, 1 shard, no replicas, 1 GiB JVM)  
**bsdb-search:** 7.10.2 (release build, storage: `local`, in-process HNSW/BM25)  
**Dataset:** 100,000 docs × 64-dim embeddings, 100 query runs, k=10

## Ingest (bulk POST)

| System | docs/s | total | mean/doc | p95/doc | refresh (until searchable) |
|---|---|---|---|---|---|
| BenoStreamDB | 12,872 | 7.8s | 0.044ms | 0.058ms | 2980ms |

## Total Time to Searchable State (Ingest + Refresh)

| System | Ingest Time | Refresh Time | Total Time |
|---|---|---|---|
| BenoStreamDB | 7.8s | 3.0s | **10.7s** |

## Query latency (p50 / p95 / p99, 100 runs each)

| System | Operation | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---|---|---|---|
| BenoStreamDB | match_bm25 | 4.632 | 4.769 | 5.073 |
| BenoStreamDB | filtered | 5.257 | 6.473 | 6.746 |
| BenoStreamDB | knn | 299.323 | 310.387 | 319.039 |
| BenoStreamDB | hybrid_rrf | 294.283 | 302.98 | 305.005 |

## Verdict vs plan Step 4.2 envelope (50–200 ms)

| System | Operation | p95 | verdict |
|---|---|---|---|
| BenoStreamDB | match_bm25 | 4.769ms | below envelope (<50ms) |
| BenoStreamDB | filtered | 6.473ms | below envelope (<50ms) |
| BenoStreamDB | knn | 310.387ms | ABOVE envelope (> 200ms) |
| BenoStreamDB | hybrid_rrf | 302.98ms | ABOVE envelope (> 200ms) |

## Storage Footprint (True Cost of Data Lake Architecture)

**Raw document payload:** `289.17 MB` (uncompressed JSON over HTTP)

| System | Primary Data (Lake) | Secondary Indexes (Search) | Data Duplication? | Total True Footprint |
|---|---|---|---|---|
| BenoStreamDB (local) | 182.43 MB (Parquet) | 217.33 MB (HNSW+BM25) | No | **399.78 MB** |

## Memory Usage

| System | Memory Ingest (RSS/Heap) | Memory Post-Search |
|---|---|---|
| BenoStreamDB (local) | 877.5 MB RSS | 13573.7 MB RSS |

## Fairness caveats

- Single-doc POST on **both** systems (bsdb-search has no `_bulk` in this test); ES index pre-created with refresh disabled, bsdb-search creates the index on first write.
- The `embedding` field is sent to bsdb-search only: ES 7.10 has no `dense_vector` type.
- `knn` and `hybrid_rrf` are bsdb-search-only (no vector search in ES 7.10).
- ES refresh does little work (segments are indexed during ingest); bsdb-search refresh includes BM25/HNSW index build, so refresh times are not like-for-like.
- bsdb-search is a **release** build; ES uses the stock Docker image.
- Both systems run on the same host; ES JVM heap is the image default (1 GiB).

## Notes

- OpenSearch skipped (--skip-es)

Raw results: `opensearch_bsdb-search_local_100000_20260912_142811.json`
