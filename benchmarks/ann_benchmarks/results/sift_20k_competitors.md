# ANN-Benchmarks Competitor Comparison: sift-128-euclidean (20k vectors)

- **Dataset**: `sift-128-euclidean` (128-dim, L2 distance)
- **Train Vectors**: 20,000
- **Test Queries**: 1,000
- **Top-K**: 10
- **Parameters**: `M=16, ef_construction=200, ef_search=200`
- **Device**: CPU (all engines)
- **Host**: AMD Ryzen 9 5900XT 16-Core Processor (Linux)

### Competitor Comparison Table

| Engine | Storage / Engine Architecture | Build Time | Index Size | Recall@10 | Throughput (QPS) | p50 Latency | p99 Latency |
|---|---|---|---|---|---|---|---|
| **hnswlib** | In-Memory C++ HNSW | 0.15s | RAM | **0.9999** | **12,308.2** | **0.08 ms** | 0.15 ms |
| **faiss** | In-Memory IndexHNSWFlat | 0.68s | RAM | **0.9999** | **9,219.4** | **0.10 ms** | 0.19 ms |
| **benostreamdb (hnsw_tq8)** | Iceberg + TQ8 Index | 0.90s | 26.2 MB | **0.9568** | **2,071.0** | **0.44 ms** | 0.64 ms |
| **benostreamdb (hnsw)** | Iceberg + HNSW Index | 1.00s | 33.9 MB | **0.9838** | **1,876.2** | **0.48 ms** | 0.67 ms |
| **lancedb (HNSW-SQ)** | Columnar Lance Table + HNSW (scalar-quantized) | 0.43s | 16.2 MB | 0.9319 | 601.9 | 1.61 ms | 2.09 ms |

*(Note: BenoStreamDB pure-index search without the Parquet row-payload fetch achieves **2,719.1 QPS** / **0.37 ms** p50 for TQ8 and **2,487.4 QPS** / **0.40 ms** p50 for HNSW).*

### Key Findings
1. **vs Columnar Embedded Competitor (LanceDB, HNSW-SQ)**: BenoStreamDB is **3.4x faster** (2,071.0 vs 601.9 QPS) with higher recall (0.9568/0.9838 vs 0.9319) under the same HNSW parameters. LanceDB has no plain-float HNSW, so its closest index (scalar-quantized HNSW) is used; its default IVF-PQ index scores far lower (recall 0.6704) and is not an apples-to-apples comparison.
2. **vs Dedicated In-Memory C++ Libraries (FAISS / Hnswlib)**: These hold raw float arrays entirely in unmanaged RAM and reach ~0.9999 recall at 9–12k QPS. BenoStreamDB maintains full transactional Iceberg tables with aligned overlay indexes while sustaining sub-millisecond query latencies (0.37–0.48 ms p50) and 0.96–0.98 recall.
3. **Quantization trade-off**: TQ8 cuts index size 23% (26.2 vs 33.9 MB) and raises QPS ~10% at a ~2.7-point recall cost versus full HNSW.

### Methodology
- Competitor numbers: `benchmarks/competitors/run_competitor.py` (faiss/hnswlib/lancedb adapters), same `M=16, ef_construction=200, ef_search=200`, 1,000 queries, k=10.
- BenoStreamDB numbers: `benchmarks/ann_benchmarks/run.py --limit 20000`, same parameters.
- All engines ran on CPU under the same host; recall is measured against exact L2 ground truth.
- LanceDB is configured with `HnswSq` (its closest HNSW-family index) rather than the default `IvfPq`, so the comparison is HNSW-vs-HNSW.
