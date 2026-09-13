# HyperStreamDB Comprehensive Guide

**Version:** 0.7.0+  
**Last Updated:** September 2026

HyperStreamDB is a serverless, indexed streaming lakehouse database combining the transactional guarantees of Apache Iceberg with reconstructible persistent index overlays (scalar roaring bitmaps, BM25 Okapi, and HNSW vector search) for blazing-fast queries directly on object storage (S3, GCS, Azure, Local).

---

## 1. Architecture Overview

HyperStreamDB decouples compute from storage, allowing for infinite scaling and zero-copy integration with data lakes.

*   **Storage-Native**: Authoritative open table format (Apache Iceberg V2 & V3) storing raw data in standard Parquet files.
*   **Advisory Index Overlays**: Reconstructible sidecar index files attached directly to Parquet data:
    *   **HNSW-IVF**: Approximate nearest neighbor vector search with dynamic metric support (`L2`, `Cosine`, `InnerProduct`, `L1`, `Hamming`, `Jaccard`).
    *   **TurboQuant™ (TQ4 & TQ8)**: Built-in scalar quantization using Fast Walsh-Hadamard Transforms for up to 8x memory compression.
    *   **Inverted Index & BM25 Okapi**: Full-text and keyword search sidecars with term frequency and document length normalization.
    *   **Roaring Bitmaps**: For boolean, categorical, and composite multi-column filtering.
    *   **Hot Row Cache**: Sub-millisecond scattered row fetches directly bypassing disk I/O on vector candidate lookups.
*   **Dual REST Search API (`hypersearch`)**: Concurrently exposes OpenSearch/Elasticsearch 7.10 (port 9200) and Qdrant Vector API (port 6333) from a single shared engine.
*   **Engine**: Built in Rust with Apache Arrow and DataFusion for vectorized query execution.

---

## 2. Getting Started

### Installation
Prerequisites: Rust toolchain (latest stable).

```bash
# Build engine and search server from source
cargo build --release

# Install Python bindings
pip install . 
```

### Basic Usage (Python)

```python
import hyperstreamdb as hdb
import pyarrow as pa
import pandas as pd
import numpy as np

# 1. Create a Table with Schema
schema = pa.schema([
    ('id', pa.int32()), 
    ('label', pa.int32()),   # 1:World, 2:Sports, 3:Business, 4:Sci/Tech
    ('title', pa.string()),
    ('description', pa.string()),
    ('embedding', pa.list_(pa.float32(), 384)) # SBERT/all-MiniLM-L6-v2 size
])

table = hdb.Table.create("file:///tmp/news_db", schema)

# 2. Ingest Real Data
df = pd.DataFrame({
    'id': [1, 2],
    'label': [3, 4],
    'title': ["Wall St. Bears Claw Back", "SpaceX Launches New Falcon"],
    'description': ["Stocks fell today as inflation concerns...", "The private space company successfully..."],
    'embedding': [np.random.rand(384).tolist() for _ in range(2)]
})

table.write(df)
table.commit()

# 3. Hybrid Search (Scalar + Vector)
query_vec = np.random.rand(384).tolist()
results = table.search(
    vector_column="embedding",
    query_vector=query_vec,
    k=5,
    filter="label = 4 AND description LIKE '%Space%'"
)
print(results.to_pandas()[['title', 'description']])
```

---

## 3. Key Features

### 3.1 SQL Support & pgvector Operators
HyperStreamDB integrates with Apache DataFusion to support full SQL queries with pgvector-compatible syntax.

```python
session = hdb.Session()
session.register_table("my_table", table)

df = session.sql("""
    SELECT id, content,
           embedding <-> '[0.1, 0.2, 0.3]'::vector AS l2_distance,
           embedding <=> '[0.1, 0.2, 0.3]'::vector AS cosine_distance
    FROM my_table 
    WHERE category = 'science' 
    ORDER BY l2_distance 
    LIMIT 10
""")
```

**Supported Distance Operators**:
- `<->` L2 (Euclidean) distance
- `<=>` Cosine distance
- `<#>` Inner product
- `<+>` L1 (Manhattan) distance
- `<~>` Hamming distance
- `<%>` Jaccard distance

### 3.2 Multi-Vector Search & Reciprocal Rank Fusion (RRF)
Search and rank across multiple embedding columns simultaneously (e.g. text embedding + image embedding) using Reciprocal Rank Fusion:

```python
# Multi-vector search combined via RRF
results = table.multi_vector_search(
    queries=[
        {"column": "text_emb", "query": text_vec, "k": 20, "metric": "cosine"},
        {"column": "image_emb", "query": image_vec, "k": 20, "metric": "l2"}
    ],
    top_k=10,
    rrf_k=60
)
```

### 3.3 Composite Scalar Roaring Bitmap Indexes
Accelerate multi-column point and range queries (e.g. `(tenant_id, status)`):

```python
# Create composite index
table.create_composite_index(
    columns=["tenant_id", "status"],
    algorithm="composite_bitmap"
)
```

### 3.4 Dual REST Search API (`hypersearch`)
Run standard OpenSearch / Elasticsearch 7.10 clients or Qdrant vector clients directly against HyperStreamDB:

```bash
# Start dual REST gateway (OpenSearch on :9200, Qdrant on :6333)
cargo run -p hyperstreamdb-search --bin hypersearch
```

- **OpenSearch / Elasticsearch (Port 9200)**: Supports `_bulk`, `_search` (BM25, kNN, hybrid RRF), `_mapping`, `_cat/indices`, and index management.
- **Qdrant Vector API (Port 6333)**: Supports collection management, point upsert, and vector similarity search.

### 3.5 TurboQuant™ (TQ4 / TQ8) Quantization
Built-in scalar quantization reduces HNSW memory consumption by up to 8x:

```python
# Register 8-bit TurboQuant index (4x memory reduction)
table.add_index("embedding", "hnsw_tq8")

# Register 4-bit TurboQuant index (8x memory reduction)
table.add_index("embedding", "hnsw_tq4")
```

### 3.6 Hardware Acceleration
The indexing engine supports hardware acceleration across multiple backends:
*   **CUDA**: NVIDIA GPUs (`cudarc`)
*   **Metal**: Apple Silicon (MPS via WGPU)
*   **ROCm**: AMD GPUs
*   **Intel**: AVX-512 and XPU runtime SIMD dispatch

---

## 4. Multi-Catalog & Enterprise Governance

HyperStreamDB integrates seamlessly with standard data catalogs for table discovery and atomic commits:

*   **Apache Polaris & Lakekeeper**: Full OAuth2 client credentials grant flow (`/v1/oauth/tokens`) with automatic background token refresh.
*   **Project Nessie**: Git-like branching, merging, and versioning for lakehouse tables.
*   **AWS Glue**: Managed serverless metadata and optimistic locking.
*   **Hive Metastore (HMS)**: Thrift-based catalog integration for Hadoop/Spark environments.
*   **Unity Catalog**: REST catalog for Databricks ecosystems.

Example connecting to Apache Polaris REST catalog with OAuth2:
```python
table = hdb.Table.from_rest(
    url="https://polaris.example.com/api/catalog/v1",
    namespace="production",
    table="events",
    credential="CLIENT_ID:CLIENT_SECRET",
    scope="PRINCIPAL_ROLE:ALL"
)
```

---

## 5. Operational Tooling & Observability

### CLI Commands (`hyperstream`)
```bash
# Inspect table metadata and partitions
hyperstream table inspect --uri s3://bucket/table

# Compaction (Merge small Parquet files)
hyperstream table compact --uri s3://bucket/table

# Clean up unreferenced snapshots
hyperstream table expire-snapshots --uri s3://bucket/table --older-than-days 7
```

### Prometheus Metrics & Tracing
- **Prometheus**: Exposed on `GET /metrics` on the search server or client metrics.
- **Tracing**: OpenTelemetry (OTLP) export via `JAEGER_ENABLED=true`.

---

## 6. Performance Best Practices

1. **Leverage Hot Row Cache**: `HYPERSTREAM_BLOCK_CACHE_GB` defaults to 1 GB (proven rock-solid under 4GB container constraints during 1M document scaling tests), caching candidate record batches in memory for sub-2ms kNN search.
2. **Column Projection**: Always specify `columns=[...]` in scalar reads to skip reading large high-dimensional vector embeddings.
3. **Use TurboQuant for Large Vector Sets**: TQ8 and TQ4 provide 4x–8x memory savings with negligible recall loss.
4. **Regular Compaction**: Run `table.compact()` to merge fragmented segments and maintain optimal HNSW graph structures.
