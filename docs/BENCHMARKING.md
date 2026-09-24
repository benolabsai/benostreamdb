# Benchmarking Strategy and Performance Analysis

This document provides a comprehensive overview of BenoStreamDB's benchmarking methodology, competitive evaluation against **OpenSearch 2.11** at 100K and 1M document scales, comparison against LanceDB, and Rust-native micro-benchmark test suites.

---

## 1. Competitive Benchmarks: `bsdb-search` vs. OpenSearch 2.11 (100K & 1M Docs)

This section consolidates the empirical findings from our long-running competitive scaling benchmarks comparing BenoStreamDB's `bsdb-search` REST server against a single-node OpenSearch 2.11 instance.

### Benchmark Setup & Test Environment
Both engines were executed under identical, strictly constrained container environments via Docker:
- **CPU Constraint**: 4 CPU cores (`cpus: "4.0"`)
- **RAM Constraint**: 4 GB RAM hard limit (`memory: "4g"`)
- **Dataset**: Wikipedia text payloads paired with 64-dimensional float32 vector embeddings.
- **Query Load**: 100 repeated query iterations after warm-up to derive statistically robust p50 and p99 latencies.
- **Tooling**: `bench_100k.py`, `bench_1m.py`, driven by `run_comparison.sh` and `run_1m_comparison.sh` via `docker-compose-bench.yml`.

---

### Results Summary Table

| Scale | Operation / Metric | BenoStreamDB (p50) | BenoStreamDB (p99) | OpenSearch 2.11 (p50) | OpenSearch 2.11 (p99) | Performance Ratio |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **100K** | Ingest Throughput (docs/s) | 4,419 | — | **7,510** | — | 0.59x *(deferred index build in OS)* |
| **100K** | `match` BM25 Search (ms) | 3.96 | 4.75 | **2.91** | **4.33** | Parity (<5ms) |
| **100K** | Filtered `match` Search (ms) | 5.17 | 5.35 | **3.23** | **4.45** | Parity (<6ms) |
| **100K** | `knn` HNSW Vector Search (ms) | **1.94** | **4.26** | 4.29 | 62.58 | **BenoStreamDB 14.7x faster p99** |
| **100K** | Operational Storage Footprint | **~26.0 MB** | — | ~185.4 MB | — | **BenoStreamDB 7.1x less disk** |
| **1M** | Ingest Throughput (docs/s) | 4,613 | — | **9,221** | — | 0.50x *(deferred index build in OS)* |
| **1M** | `knn` HNSW Vector Search (ms) | **1.91** | **3.74** | 8.26 | 478.77 | **BenoStreamDB 128x faster p99** |
| **1M** | Operational Storage Footprint | **~280 MB** | — | ~1,852 MB (1.85 GB) | — | **BenoStreamDB 6.6x less disk** |

---

### Detailed Findings & Architectural Analysis

#### 1. Vector Query Latency & Tail Stability (Hot Row Cache)
- **100K Scale**: BenoStreamDB achieves a p50 of **1.94 ms** and p99 of **4.26 ms**, outperforming OpenSearch's 4.29 ms p50 and 62.58 ms p99.
- **1M Scale**: BenoStreamDB's query latency remains **completely flat** at **1.91 ms p50** and **3.74 ms p99**. Conversely, OpenSearch suffers catastrophic tail latency collapse: its p50 degrades to **8.26 ms**, while its p99 explodes to **478.77 ms** (nearly half a second).
- **The Architectural Cause**: Under a tight 4GB container RAM constraint, OpenSearch triggers heavy JVM garbage collection pauses, buffer swapping, and Lucene background segment merging. BenoStreamDB utilizes an asynchronous, thread-safe **Hot Row Cache** (`BLOCK_CACHE`) that serves scattered kNN vector candidate row fetches directly from memory, eliminating disk I/O bottlenecks and guaranteeing flat latency profiles regardless of dataset scale.

#### 2. Total Storage Footprint & Zero Data Duplication
When evaluating total infrastructure and operational costs, architectural differences between secondary search engines and native lakehouses become critical:
- **OpenSearch (Secondary Index Architecture)**:
  - Requires storing primary data in a persistent Data Lake or database (e.g. S3 Parquet), *plus* pumping a complete duplicate copy into OpenSearch's Lucene indices.
  - Lucene's graph and document storage amplify vector footprints significantly.
  - At 100K vectors (25.6 MB raw payload), OpenSearch requires **185.4 MB** total storage (25.6 MB raw data + 159.84 MB Lucene index).
  - At 1M vectors (256 MB raw payload), OpenSearch requires **1,852 MB (1.85 GB)** total storage (256 MB raw data + 1,596 MB Lucene index).
- **BenoStreamDB (Native Lakehouse Overlay Architecture)**:
  - BenoStreamDB **is** the Data Lake. It directly queries native Parquet data and creates compact `.hnsw` sidecar index files alongside data files.
  - At 100K vectors, BenoStreamDB requires **~26.0 MB** total storage (**7.1x storage savings**).
  - At 1M vectors, BenoStreamDB requires **~280 MB** total storage (**6.6x storage savings**).
  - Zero data duplication: single source of truth for analytical SQL (Spark, Trino, DataFusion) and low-latency search.

#### 3. Cold Start & Restart Overhead
- **OpenSearch**: When restarted, the JVM must boot, initialize thread pools, join or elect cluster coordinators, replay transaction logs (translogs), and page Lucene HNSW graph segments into memory before serving vector traffic.
- **BenoStreamDB**: Built as a stateless Rust binary. It attaches directly to the Parquet and `.hnsw` sidecars via memory mapping (`mmap`) backed by the OS page cache. Upon process restart or scale-to-zero wake-up, queries are served almost instantly with zero warm-up penalty.

#### 4. Ingestion Throughput & Asynchronous Indexing
- OpenSearch achieves higher raw ingestion rates (~7.5k to ~9.2k docs/s) during active insertion because it defers heavy HNSW graph operations to periodic background merges (`refresh_interval`).
- BenoStreamDB's baseline synchronous indexing builds graph structures in the hot path (~4.4k to ~4.6k docs/s).
- With the introduction of the Async WAL and Memtable batch coalescing pipeline, BenoStreamDB decouples physical ingestion from background HNSW construction.

#### 5. Memory Safety
- Both 100,000 and 1,000,000 document benchmark runs completed within the 4GB hard container limit without any Out-Of-Memory (OOM) aborts, confirming deterministic memory management in Rust.

---

### How to Reproduce the Competitive Benchmarks

1. **Spin up the isolated benchmark environment**:
   ```bash
   # Launch Docker Compose with 4 CPUs / 4GB RAM limits
   docker compose -f docker-compose-bench.yml up -d
   ```

2. **Run the 100K Document Comparison**:
   ```bash
   ./run_comparison.sh
   # Or directly invoke the Python benchmark runner:
   python bench_100k.py --es-url http://localhost:9201 --hs-url http://localhost:9200
   ```

3. **Run the 1M Document Comparison**:
   ```bash
   ./run_1m_comparison.sh
   # Or directly invoke:
   python bench_1m.py --es-url http://localhost:9201 --hs-url http://localhost:9200
   ```

---

## 2. Ingestion & High-Dimensional Vector Performance (vs. LanceDB)

BenoStreamDB features high-throughput vector ingestion that rivals dedicated local-first vector formats like LanceDB, while preserving full Apache Iceberg table format compatibility.

### Key Architectural Optimizations
1. **Delayed Indexing (Async Worker Pool)**: Ingestion is non-blocking. Vectors are committed to Parquet immediately, while indexing proceeds asynchronously in the background across available CPU cores.
2. **Mini-Batch K-Means Centroid Training**: IVF centroid training is $O(\text{Sample})$ instead of $O(N)$, speeding up centroid calculation by over 10x.
3. **Parallel PQ Training**: Product Quantization subspaces are trained concurrently, fully saturating multi-core CPUs.
4. **Runtime SIMD Dispatch**: Automatic AVX2/AVX-512/FMA detection ensures peak throughput across hardware configurations.

### 768-Dimensional Vector Throughput (Intel i5-8350U @ 1.70GHz)

| Metric | Baseline | Optimized Engine | Speedup |
| :--- | :---: | :---: | :---: |
| **Ingestion Throughput (10k rows)** | 360 rows/sec | **6,834 rows/sec** | **19.0x** |
| **Ingestion Throughput (100k rows)** | — | **22,999 rows/sec** | **—** |
| **Indexing Latency (10k rows)** | 27.8s | **1.46s** | **19.0x** |
| **Write Availability** | Blocking | **Instant (Async)** | ∞ |

### Competitive Ingest Comparison: BenoStreamDB vs. LanceDB (100k 768D Vectors)
- **BenoStreamDB (768D, 100k)**: **22,999 rows/sec**
- **LanceDB (768D, 100k)**: **45,427 rows/sec**

---

## 3. Benchmarking Strategy & Competitive Matrix

Our benchmarking methodology categorizes storage and search engines into three tiers:

### Tier 1: Direct Architectural Competitors
- **OpenSearch / Elasticsearch**: Primary targets for the REST Search API (`bsdb-search`). Evaluated on search latency (BM25, kNN, hybrid RRF), index footprint, and cold starts.
- **Deep Lake**: Architectural peer (data lake + vector search). Focus: Hybrid SQL queries and ACID transaction isolation.
- **Apache Iceberg / Delta Lake**: Standard lakehouse formats. Focus: Point lookup speedups (100–1000x) via index sidecars and integrated vector search.

### Tier 2: Vector Search & Specialized Engines
- **Qdrant**: Direct vector database competitor. Focus: Filtered vector search pre-pruning performance and Qdrant REST protocol compatibility.
- **LanceDB**: Disk-based vector lakehouse. Focus: Query latency, index build times, and Arrow integration.
- **Milvus**: Distributed vector database. Focus: Serverless scale-to-zero costs vs. always-on infrastructure overhead.

### Tier 3: Secondary Comparisons
- **pgvector**: Relational database vector extension.
- **Apache Hudi**: Alternative lakehouse format.

---

## 4. Measurement Rigor & Accuracy Standards

To guarantee high statistical fidelity, our benchmark harnesses enforce strict measurement controls:

1. **Index Verification**: Benchmarks poll for complete index readiness (`_cat/indices` or manifest validation) before timing queries, avoiding cold full scans.
2. **Query Warm-Up**: Initial queries are discarded from metrics to eliminate one-off thread pool initialization and OS page faults.
3. **Statistical Sample Size**: 100+ query executions per test case to calculate accurate p50, p90, and p99 percentiles.
4. **Isolated Resource Constraints**: Hard CPU and memory enforcement using Docker container limits (`--cpus` and `-m`) to simulate production cloud environments.
5. **Combined Hybrid Testing**: Hybrid queries test actual pre-filter pruning combined with vector distance computation.

---

## 5. Rust-Native Micro-Benchmarks (`cargo bench`)

For low-level execution without Python runtime overhead, BenoStreamDB includes a comprehensive suite of native micro-benchmarks implemented with [Criterion](https://github.com/bheisler/criterion.rs):

### Benchmark Suites
- **`benches/bench_table.rs`**: Evaluates 100K+ batch ingestion, scalar index point lookups, HNSW-IVF vector distance calculations across dimensions (128D–1536D), and background segment compaction.

### Running Micro-Benchmarks
```bash
# Run all Criterion micro-benchmarks
cargo bench

# Run specific table benchmarks with statistical output
cargo bench --bench bench_table
```

All Criterion runs output statistical summaries with outlier detection and regression analysis in `target/criterion/`.
