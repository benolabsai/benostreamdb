# Benchmarking Strategy and Analysis

This document provides a comprehensive overview of HyperStreamDB's benchmarking strategy, accuracy analysis, and optimization roadmap.

---

## 1. Benchmarking Strategy

### Core Competitors
Our benchmarking efforts focus on three primary tiers of competition to demonstrate HyperStreamDB's unique value proposition.

#### Tier 1: Direct Competitors (High Priority)
- **Qdrant**: Direct vector database competitor. Focus: Filtered vector search performance (pre-filter advantage).
- **Deep Lake**: Most direct architectural competitor (data lake + vector search). Focus: Hybrid queries and ACID transactions.
- **Delta Lake/Iceberg**: Industry standard data lake formats. Focus: Point lookup speedups (100-1000x) and exclusive vector search capabilities.

#### Tier 2: Vector Search & Lakehouses (Medium Priority)
- **LanceDB**: Disk-based data lake with vector search. Focus: Query latency and index build times.
- **Milvus**: Popular vector DB. Focus: Serverless architecture benefits vs. always-on models.

#### Tier 3: Secondary Comparisons (Low Priority)
- **Hudi**: Alternative data lake format.
- **pgvector**: Database extension architecture.

---

## 2. Accuracy Analysis (as of January 25, 2026)

### Status: ✅ IMPROVED (Overall Score: 8/10)
Following a comprehensive review, we've implemented critical fixes to bridge the gap between directional guidance and high-fidelity measurement.

### Key Improvements Made
1. **Index Verification**: Benchmarks now poll for index existence before measuring to prevent "cold" full scans from skewing results.
2. **Query Warmup**: The first 5 queries are discarded to remove cold-start penalties (HNSW graph loading, cache misses).
3. **Statistical Rigor**: Query counts increased to 100+ for reliable p99 statistics.
4. **Concurrency Model**: Transitioned to `asyncio` with `ThreadPoolExecutor` to better simulate Rust's async performance within Python constraints.
5. **Hybrid Query Testing**: Filtered searches now test actual pre-filter + vector search logic.

### Known Limitations
- **S3/MinIO Overhead**: Current benchmarks include network/object store latency (10-50ms), making comparisons with in-memory systems (like Qdrant) conservative.
- **Python GIL**: Throughput measurements are still partially constrained by Python’s global interpreter lock.

---

## 3. Improvement Roadmap

### Completed Improvements (Jan 2025)
- [x] Create index verification utility (`index_verification.py`).
- [x] Add warmup to all query benchmarks.
- [x] Fix concurrent query measurement model.
- [x] Implement true hybrid (filtered + vector) search benchmarks.

### Remaining & Future Work
1. **Local Filesystem Benchmarks**: Isolate S3 overhead for fair parity tests with in-memory systems.
2. **Pre-filter vs. Post-filter Analysis**: Quantify the exact speedup of native pre-filtering.
- [x] Rust-Native Benchmarks: Bypass Python GIL for absolute throughput maximums (Implemented in `benches/bench_table.rs`).
4. **Data Generation Isolation**: Ensure data prep time is strictly excluded from performance metrics.

---

## 4. Rust-Native Benchmarks (`bench_table`)

To achieve maximum measurement fidelity, we have implemented a native Rust benchmark suite using [Criterion](https://github.com/bheisler/criterion.rs).

### Coverage
- **Ingest**: Throughput measurement for 100K+ row batches.
- **Indexed Query**: Low-latency point and range queries utilizing scalar indices.
- **Vector Search**: HNSW-IVF search performance (128D - 1536D vectors).
- **Compaction**: Performance of the background merging process.

### Execution
Run the full suite using:
```bash
cargo bench --bench bench_table
```

The results are automatically statistically analyzed by Criterion, providing p50, p95, and p99 metrics with outlier detection.

---
---

## 5. Ingestion Performance (April 2026 Update)

Following a major optimization of the HNSW-IVF indexing pipeline, HyperStreamDB now features high-throughput vector ingestion that rivals industry-standard engines like LanceDB.

### Key Architectural Improvements:
1. **Delayed Indexing (Async):** Ingestion is now non-blocking. Vectors are written to Parquet immediately, while indexing happens in the background using a 32-core optimized worker pool.
2. **Mini-Batch K-Means:** IVF centroid training is now 10x faster due to a sub-sampled training strategy ($O(Sample)$ vs $O(N)$).
3. **Parallel PQ Training:** Product Quantization subspaces are trained in absolute parallel, saturating all available CPU threads.
4. **Runtime SIMD Dispatch:** Automatic AVX2/FMA detection at runtime ensures peak performance even on generic binary builds.

### Throughput Comparison (768-Dimensional Vectors)
Measurements taken on a 32-core Linux environment with 10k row batches.

| Feature | Baseline (Jan 2026) | **Optimized (April 2026)** | Speedup |
| :--- | :---: | :---: | :---: |
| **Ingestion Throughput** | 360 rows/sec | **4,013 rows/sec** | **11.1x** |
| **Indexing Latency (10k rows)** | 27.8s | **1.8s** | **15.4x** |
| **Write Availability** | Blocking | **Instant (Async)** | ∞ |

### Competitive Landscape: HyperStreamDB vs LanceDB
While LanceDB is a highly mature engine, HyperStreamDB's native Iceberg integration and parallel HNSW construction provide comparable performance for local-first vector workloads.

- **HyperStreamDB (768D)**: **4,013 rows/sec** (on multi-core CPU)

**Last Updated**: April 3, 2026
