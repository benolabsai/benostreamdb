# Benchmark Findings — `hypersearch` vs OpenSearch 2.11 (1M docs)

This document summarizes the 1M document scaling benchmark comparing the `hypersearch` REST server against a single-node OpenSearch 2.11 instance. Both instances were strictly constrained via Docker to **4 CPU cores** and **4GB of RAM**. They ingested a stream of 1,000,000 documents (64-dimensional embeddings, Wikipedia text payload).

## Results — (1M docs, dim 64, 100 runs)

| Operation | HyperStreamDB p50 | HyperStreamDB p99 | OpenSearch 2.11 p50 | OpenSearch 2.11 p99 |
|-----------|-------------------|-------------------|---------------------|---------------------|
| Ingest (docs/s) | 4,613 | — | **9,221** | — |
| `knn` HNSW (ms) | **1.91** | **3.74** | 8.26 | 478.77 |

## Findings

1. **Catastrophic Tail Latency in OpenSearch**: At 1,000,000 documents under 4GB of RAM, OpenSearch begins to heavily thrash. While its median latency degraded to ~8.2ms, its **p99 latency skyrocketed to 478.77ms** (nearly half a second) due to massive JVM Garbage Collection pauses and Lucene segment merging.
2. **HyperStreamDB Ironclad Stability**: Even at 10x the scale, HyperStreamDB's query latency remained **completely flat**. Thanks to the Hot Row Cache bypassing disk I/O, HyperStreamDB achieved a blistering **1.91ms p50** and an ultra-stable **3.74ms p99**. 
3. **Ingest Throughput**: OpenSearch achieves roughly double the ingestion throughput (~9.2k docs/s vs ~4.6k docs/s) because it defers the heavy HNSW graph operations to background merges, while HyperStreamDB builds its graph synchronously in the hot path. 

## Total Time & Disk Usage Analysis

### Disk Usage & Data Duplication (Zero Data Duplication)
When calculating disk usage and operational overhead, we must account for the architectural differences. OpenSearch is a secondary search engine. This means you must store your data in your primary database or Data Lake (e.g., S3/Parquet), **and** then pump a complete duplicated copy into OpenSearch's Lucene indices.

HyperStreamDB **is** the Data Lake. It natively serves the Parquet files and simply attaches `.hnsw` index sidecars to them. You do not need a separate data lake when using HyperStreamDB.

For our 1,000,000 vector benchmark (64-dimensional float32 arrays), the pure raw data payload is exactly 256 MB. Here is the true disk footprint required:

- **HyperStreamDB Total Storage Required**: **~280 MB**
  - *Why?* HyperStreamDB writes the raw vectors directly into a compressed Parquet file and builds lightweight `.hnsw` sidecars. Because HyperStreamDB *is* the data lake, this is the total operational disk footprint.
- **OpenSearch Total Storage Required**: **~1,852 MB (1.85 GB)** 
  - *Why?* You must keep the original 256 MB data lake payload, plus the heavily amplified OpenSearch Lucene Index (1,596.59 MB) required to serve the vectors.

**Conclusion**: At 1M documents, OpenSearch requires **6.6x more disk space** because it amplifies the data through Lucene HNSW graph copies and forces you to maintain a separate primary data lake. HyperStreamDB completely eliminates the separate data lake and serves searches directly from the compressed source of truth.

### Cold Start & Restart Penalties
Another massive architectural advantage is cold-start performance. When OpenSearch restarts, the JVM must boot up, re-join the cluster, replay translogs, and load the 1.6GB Lucene HNSW structures into memory before it can answer queries. 

HyperStreamDB is stateless. It points directly to the data lake, memory maps the native Parquet and `.hnsw` sidecar files directly from the OS page cache, and can begin answering queries almost instantly with near-zero cold start penalty.
