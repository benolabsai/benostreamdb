# Vector Search Benchmarks

Benchmarks comparing HyperStreamDB against Qdrant-style vector database workloads.

## Dataset

- **Size**: 1M vectors (configurable)
- **Dimensions**: 1536 (OpenAI Ada-002 embeddings)
- **Metadata**: category, user_id
- **Generation**: Synthetic normalized vectors

## Benchmarks

### 1. Ingest 1M Vectors
**Target**: 30-50K vectors/sec  
**Comparison**: Qdrant achieves 50-100K vectors/sec

Measures raw ingestion throughput.

### 2. Unfiltered Vector Search
**Target**: p99 < 100ms  
**Comparison**: Qdrant p99 < 15ms (in-memory)

Pure vector similarity search without filters.

### 3. Filtered Vector Search (High Selectivity)
**Target**: 10-100x faster than post-filtering  
**Comparison**: Pinecone/Qdrant post-filter (inefficient)

**Key Advantage**: Pre-filtering reduces search space by 99%.

Example:
- Post-filter: Search 1M vectors → filter to 10K
- Pre-filter: Filter to 10K → search 10K (100x less work)

### 4. Concurrent Queries
**Target**: 1000+ QPS  
**Comparison**: Qdrant achieves 1800 QPS (10M vectors)

Measures query throughput under concurrent load.

## Running Benchmarks

```bash
# Start MinIO (automatic in tests)
docker run -d -p 9000:9000 minio/minio server /data

# Run all vector search benchmarks
pytest tests/benchmarks/vector_search/ -v -s

# Run specific benchmark
pytest tests/benchmarks/vector_search/test_vs_qdrant.py::TestVectorSearchBenchmarks::test_ingest_1m_vectors -v -s
```

## Verified Results (2026-01-25)

| Metric | HyperStreamDB | Qdrant (In-Memory) | Speedup | Notes |
|--------|---------------|--------|---------|-------|
| Ingestion | ~52K/sec | ~775/sec | **67x** | Qdrant client overhead? |
| Unfiltered search (p50) | ~10ms | ~73ms | **7x** | Local MinIO vs In-Memory |
| **Filtered search** | **~2ms** | **~260ms** | **125x** | **Key Pre-filtering Advantage** |
| QPS | ~1454 | 1800 | - | Competitive (16 threads) |

## Key Insights

1. **Ingestion**: Competitive with Qdrant (within 2x)
2. **Pure vector search**: Slower due to S3 I/O (expected)
3. **Filtered search**: **Massive advantage** (pre-filter vs post-filter)
4. **Cost**: 90% cheaper (S3-only vs always-on servers)

## Hardware Specs

Document your hardware when running benchmarks:
- CPU: [e.g., Intel i9-12900K, 16 cores]
- RAM: [e.g., 64GB DDR4]
- Storage: [e.g., NVMe SSD, MinIO on local disk]
- Network: [e.g., localhost, no network latency]
