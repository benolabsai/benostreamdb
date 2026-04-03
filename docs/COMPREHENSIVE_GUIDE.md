# HyperStreamDB Comprehensive Guide

**Version:** 0.1.0 (Alpha)  
**Last Updated:** 2026-01-27

HyperStreamDB is a serverless, hybrid-search database optimized for high-performance vector and scalar queries directly on data lakes (S3, GCS, Azure, Local).

---

## 1. Architecture Overview

HyperStreamDB decouples compute from storage, allowing for infinite scaling and zero-copy integration with data lakes.

*   **Storage-Native**: Data and indices are stored as Parquet and custom index files in your object store.
*   **Serverless**: No long-running daemons required. Operations spin up, execute, and spin down.
*   **Hybrid Indices**:
    *   **HNSW-IVF**: For approximate nearest neighbor (vector) search.
    *   **Inverted Index**: For ultra-fast scalar filtering on high-cardinality columns.
    *   **Roaring Bitmaps**: For boolean and categorical filtering.
*   **Engine**: Built on Rust with Apache Arrow and DataFusion for vectorized query execution.

---

## 2. Getting Started

### Installation
Prerequisites: Rust toolchain (latest stable).

```bash
# Build from source
cargo build --release
pip install . 
```

### Basic Usage (Python)

```python
import hyperstreamdb as hdb
import pyarrow as pa
import pandas as pd
import numpy as np

# 1. Create a Table with AG News Schema
schema = pa.schema([
    ('id', pa.int32()), 
    ('label', pa.int32()),   # 1:World, 2:Sports, 3:Business, 4:Sci/Tech
    ('title', pa.string()),
    ('description', pa.string()),
    ('embedding', pa.list_(pa.float32(), 384)) # SBERT/all-MiniLM-L6-v2 size
])

table = hdb.Table.create("file:///tmp/ag_news", schema)

# 2. Ingest Real Data (Example: AG News Sample)
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
# Search for "Space" related news in "Sci/Tech" category (label=4)
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

### 3.1 SQL Support
HyperStreamDB integrates with Apache DataFusion to support full SQL queries with pgvector-compatible syntax.

#### Basic SQL Queries

```python
session = hdb.Session()
session.register_table("my_table", table)

df = session.sql("""
    SELECT id, content 
    FROM my_table 
    WHERE id > 500 
    ORDER BY id DESC 
    LIMIT 10
""")
```

#### pgvector SQL Operators

HyperStreamDB provides full pgvector compatibility for vector operations:

```python
# Vector similarity search with distance operators
results = session.sql("""
    SELECT id, content,
           embedding <-> '[0.1, 0.2, 0.3]'::vector AS l2_distance,
           embedding <=> '[0.1, 0.2, 0.3]'::vector AS cosine_distance
    FROM documents
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

**Vector Aggregations**:
```python
# Compute category centroids
results = session.sql("""
    SELECT category, 
           vector_avg(embedding) AS centroid,
           COUNT(*) AS doc_count
    FROM documents
    GROUP BY category
""")
```

**Configuration Parameters**:
```python
# Tune search accuracy/speed tradeoff
session.set_config("hnsw.ef_search", 128)  # Higher = more accurate
session.set_config("ivf.probes", 20)       # Higher = more accurate
```

See [pgvector SQL Guide](PGVECTOR_SQL_GUIDE.md) for complete documentation.

### 3.2 Hardware Acceleration
The indexing engine supports hardware acceleration for multiple backends:
*   **CUDA**: NVIDIA GPUs (Linux/Windows)
*   **Metal**: Apple Silicon (MPS)
*   **ROCm**: AMD GPUs
*   **Intel**: AVX-512 optimizations

Enable via `Cargo.toml` features or environment detection.

### 3.3 Multi-Catalog Support
HyperStreamDB supports enterprise catalog integrations:
*   **Nessie**: Git-like versioning for data.
*   **Unity Catalog**: Databricks integration.
*   **AWS Glue**: Native AWS metadata.
*   **Hive Metastore**: Legacy Hadoop compatibility.
*   **REST**: Iceberg-compatible REST catalog.

---

## 4. Operational Tooling

A dedicated CLI `hdb` is provided for management.

### CLI Commands
```bash
# Inspect table metadata/stats
hdb table inspect --uri s3://bucket/table

# Compaction (Merge small files)
hdb table compact --uri s3://bucket/table

# Vacuum (Cleanup old files)
hdb table vacuum --uri s3://bucket/table --older-than-days 7
```

### Metrics & Tracing
*   **Metrics**: Prometheus endpoint compatible. Tracks ingestion rates, query latency, and compaction times.
*   **Tracing**: OpenTelemetry (OTLP) support.
    *   Enable: `export JAEGER_ENABLED=true`
    *   Sends traces to `localhost:4317` (OTLP/gRPC).

---

## 5. Performance Tips

1.  **Index Everything**: Ensure vector columns have HNSW indices and scalar filter columns have Inverted indices.
    *   `table.create_index("col", index_type="hnsw")`
2.  **Projection**: Always specify `columns=[...]` in `read` operations to avoid reading unused large embeddings.
3.  **Compaction**: Run compaction regularly to keep file count low and query performance high.

---

## 6. Roadmap Status

| Feature | Status |
|---------|--------|
| Core Vector Search | ✅ |
| Hybrid Filtering | ✅ |
| Native Partitioning | ✅ |
| Hardware Accel (GPU) | ✅ |
| SQL Engine | ✅ |
| Catalogs (Glue/Unity/etc) | ✅ |
| CLI & Observability | ✅ |
| Spark/Trino Connectors | 🚧 (APIs ready) |

---

## 7. Partitioning Strategy

HyperStreamDB supports coarse-grained pruning via table partitioning. This allows the query planner to skip entire directories of data without reading file headers.

### Creating a Partitioned Table
```python
spec = {
    "fields": [
        {"name": "category", "transform": "identity"}
    ]
}
table = hdb.Table.create_partitioned("s3://bucket/table", schema, spec)
```

### Benefits
*   **Massive Scale**: Scan millions of files in milliseconds by pruning based on high-level keys.
*   **S3 Lifecycle**: Easily move old partitions (e.g., `date=2023`) to colder storage classes.
*   **Isolation**: Writes to different partitions never conflict, enabling high-concurrency ingestion.

