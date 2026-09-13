# API Reference

This page provides an overview of HyperStreamDB APIs across different languages and interfaces.

## Python API

### Table Operations

```python
import hyperstreamdb as hdb

# Create/open table
table = hdb.Table("s3://bucket/my-table")

# Write data
table.write_pandas(df)
table.write_arrow(arrow_table)

# Read data
df = table.to_pandas()
df = table.to_pandas(filter="id > 100")

# Vector search with default L2 distance
df = table.to_pandas(
    vector_filter={
        "column": "embedding",
        "query": query_vec,
        "k": 10
    }
)

# Vector search with custom metric and index parameters
df = table.to_pandas(
    vector_filter={
        "column": "embedding",
        "query": query_vec,
        "k": 10,
        "metric": "cosine",      # l2, cosine, inner_product, l1, hamming, jaccard
        "ef_search": 200,        # HNSW parameter (higher = more accurate, slower)
        "probes": 10             # IVF parameter (higher = more accurate, slower)
    }
)

# Hybrid query (scalar + vector)
df = table.to_pandas(
    filter="category = 'science'",
    vector_filter={
        "column": "embedding",
        "query": query_vec,
        "k": 10,
        "metric": "cosine"
    }
)

# Multi-vector search with Reciprocal Rank Fusion (RRF)
results = table.multi_vector_search(
    queries=[
        {"column": "text_embedding", "query": text_vec, "k": 20, "metric": "cosine"},
        {"column": "image_embedding", "query": image_vec, "k": 20, "metric": "l2"}
    ],
    top_k=10,
    rrf_k=60
)

# Composite scalar index (multi-column roaring bitmap)
table.create_composite_index(
    columns=["tenant_id", "status"],
    algorithm="composite_bitmap"
)

# Maintenance
table.compact()
table.expire_snapshots(retain_last=10)
```

### Vector Distance API

See [Python Vector API Documentation](PYTHON_VECTOR_API.md) for complete reference.

```python
import hyperstreamdb as hdb
import numpy as np

# Single-pair distance
distance = hdb.l2_distance(vec1, vec2)
distance = hdb.cosine_distance(vec1, vec2)

# Batch operations with GPU acceleration
ctx = hdb.GPUContext.auto_detect()
distances = hdb.l2_distance_batch(query, database, context=ctx)

# Sparse vectors
sparse = hdb.SparseVector(indices, values, dim)
distance = hdb.l2_distance_sparse(sparse1, sparse2)

# Binary vectors
distance = hdb.hamming_distance_packed(binary1, binary2)
```

**Supported Distance Metrics:**
- `l2_distance()` - Euclidean distance
- `cosine_distance()` - Cosine distance
- `inner_product()` - Inner product
- `l1_distance()` - Manhattan distance
- `hamming_distance()` - Hamming distance
- `jaccard_distance()` - Jaccard distance

**GPU Backends:**
- CUDA (NVIDIA)
- ROCm (AMD)
- Metal/MPS (Apple Silicon)
- Intel XPU (via WGPU)
- CPU (fallback)

### SQL API

```python
import hyperstreamdb as hdb

# Create session
session = hdb.Session()
session.register("my_table", table)

# Execute SQL with pgvector operators
results = session.sql("""
    SELECT id, content,
           embedding <-> '[0.1, 0.2, 0.3]'::vector AS distance
    FROM my_table
    WHERE category = 'science'
    ORDER BY distance
    LIMIT 10
""")

# Enable GPU acceleration for SQL
ctx = hdb.GPUContext.auto_detect()
hdb.set_thread_gpu_context(ctx)
```

See [pgvector SQL Guide](PGVECTOR_SQL_GUIDE.md) for SQL syntax reference.

### Iceberg V2/V3 API

```python
import hyperstreamdb as hdb

table = hdb.Table("s3://bucket/table")

# Sort orders (V2)
table.set_sort_order(["timestamp", "user_id"], ascending=[False, True])

# Partition evolution (V2)
table.set_partition_spec([
    {"source_id": 1, "field_id": 1000, "name": "date", "transform": "day"}
])

# Row lineage (V3) - automatic when format_version >= 3
# Adds _row_id and _last_updated_sequence_number columns

# Standard Iceberg operations
table.update_spec(new_spec)
table.replace_sort_order(sort_order)
table.rewrite_data_files(filter_expr)
table.rollback_to_snapshot(snapshot_id)
```

See [Iceberg V2/V3 API Guide](ICEBERG_V2_V3_API.md) for complete reference.

## Spark Connector

```scala
// Read
val df = spark.read
  .format("hyperstream")
  .option("path", "s3://bucket/table")
  .load()

// Write
df.write
  .format("hyperstream")
  .option("path", "s3://bucket/table")
  .save()

// Vector search
df.createOrReplaceTempView("documents")
spark.sql("""
  SELECT id, content,
         embedding <-> array(0.1, 0.2, 0.3) AS distance
  FROM documents
  ORDER BY distance
  LIMIT 10
""")
```

## Trino Connector

```sql
-- Query table
SELECT * FROM hyperstream.default.my_table
WHERE id > 100;

-- Vector search with pgvector operators
SELECT id, content,
       embedding <-> ARRAY[0.1, 0.2, 0.3] AS distance
FROM hyperstream.default.documents
WHERE category = 'science'
ORDER BY distance
LIMIT 10;
```

## Configuration

### GPU Context Configuration

HyperStreamDB supports GPU acceleration across NVIDIA CUDA, AMD ROCm, Apple Metal (MPS), and Intel XPU. You can configure execution devices using PyTorch-style strings (`"cuda:0"`) or explicit parameters (`device_id=0` / `index=0`).

```python
import hyperstreamdb as hdb

# 1. Auto-detect best available GPU backend
device = hdb.Device("auto")
print(f"Auto-selected: {device.backend}, device_id: {device.index}")

# 2. Specify backend and device index explicitly
device = hdb.Device("cuda:0")              # First NVIDIA GPU
device = hdb.Device("cuda", index=1)       # Second NVIDIA GPU
device = hdb.Device("rocm:0")              # AMD ROCm GPU
device = hdb.Device("mps")                 # Apple Silicon (always index 0)
device = hdb.Device("xpu:0")               # Intel discrete / integrated GPU
device = hdb.Device("cpu")                 # Force CPU execution

# Or using GPUContext API
ctx = hdb.GPUContext("cuda", device_id=0)

# 3. Query system availability & performance stats
print("Available backends:", hdb.Device.list_available_backends())

stats = device.get_stats()
print(f"GPU compute time: {stats['total_gpu_time_ms']}ms")
print(f"Kernel launches: {stats['total_kernel_launches']}")
device.reset_stats()
```

#### How to Find the Proper `device_id` Manually

To determine the exact integer `device_id` or index on your machine, run the hardware tool matching your GPU:

| Hardware Vendor | CLI Command to Inspect Device IDs | Output Column / Identification |
|:---|:---|:---|
| **NVIDIA (CUDA)** | `nvidia-smi` | The leftmost **`GPU`** column (`0`, `1`, `2`...) corresponds directly to `device_id`.<br>Scriptable: `nvidia-smi --query-gpu=index,name,memory.total --format=csv` |
| **AMD (ROCm)** | `rocm-smi` or `rocminfo` | Look for **`GPU[0]`**, **`GPU[1]`** in `rocm-smi` or **`Device [Node 1]`** in `rocminfo`. |
| **Intel (XPU / Level Zero)** | `clinfo -l` or `sycl-ls` | Lists OpenCL/OneAPI devices indexed starting from `0`. |
| **Apple Silicon (MPS)** | `system_profiler SPDisplaysDataType` | Apple Silicon uses unified memory with a single integrated GPU; `device_id` is always **`0`** (`"mps"`). |

See [GPU Setup Guide](GPU_SETUP_GUIDE.md) for full driver prerequisites and kernel benchmarks.

### Vector Index Configuration

```python
import hyperstreamdb as hdb

table = hdb.Table("s3://bucket/table")

# TurboQuant 8-bit (4x compression)
table.add_index("embedding", "hnsw_tq8")

# TurboQuant 4-bit (8x compression)
table.add_index("embedding", "hnsw_tq4")

# Custom HNSW configuration
table.add_index(
    column="embedding",
    index_config={
        "type": "hnsw",
        "complexity": 16, # connections per node
        "quality": 200     # construction search width
    }
)
```

See [Vector Configuration Guide](VECTOR_CONFIGURATION.md) for tuning parameters.

## REST Search APIs (`hyperstreamdb-search`)

HyperStreamDB provides a dual-protocol HTTP gateway exposing OpenSearch / Elasticsearch 7.10 compatibility alongside the Qdrant Vector API from a single server:

### Starting the Server
```bash
cargo run -p hyperstreamdb-search --bin hypersearch
```

### OpenSearch / Elasticsearch 7.10 Endpoints (Port 9200)

| Method | Endpoint | Description |
|:---|:---|:---|
| `GET` | `/` | Cluster status, version string, and hardware acceleration metadata. |
| `GET` | `/_cluster/health`, `/_health` | Cluster and store health status. |
| `GET` | `/_cat/indices` | Tabular index metadata (doc count, store size, health). |
| `PUT` | `/{index}` | Create an index with optional schema mapping and index type registration. |
| `GET` | `/{index}/_mapping` | Retrieve Arrow-to-Elasticsearch mapping. |
| `POST` | `/{index}/_doc` | Ingest single document with automatic schema evolution. |
| `POST` | `/_bulk`, `/{index}/_bulk` | High-throughput NDJSON bulk write (`index`, `create`, `delete`). |
| `POST` | `/{index}/_search` | Query endpoint supporting `match` (BM25), `knn` (HNSW), hybrid RRF, and filters. |
| `POST` | `/{index}/_refresh` | Trigger immediate memtable flush and Iceberg snapshot commit. |
| `GET` | `/metrics` | Prometheus metrics scrape endpoint. |

> [!NOTE]
> **Data Lake Catalog Synchronization**: Every search index created via `PUT /{index}` or auto-created on ingest is stored as an Apache Iceberg (v2) table. When an external catalog is configured (e.g. **Snowflake / Apache Polaris**, Project Nessie, AWS Glue, Hive Metastore, Unity Catalog), document ingestion via `_bulk` and `_doc` automatically commits snapshots to the catalog via Iceberg Atomic Swap, keeping Spark, Trino, and Snowflake in sync with real-time updates.


### Qdrant Vector Endpoints (Port 6333)

| Method | Endpoint | Description |
|:---|:---|:---|
| `GET` | `/collections` | List available vector collections. |
| `PUT` | `/collections/{name}` | Create a vector collection with vector parameters (dimension, distance). |
| `PUT` | `/collections/{name}/points` | Upsert vector points with payload metadata. |
| `POST` | `/collections/{name}/points/search` | Approximate nearest neighbor vector search with optional payload filters. |

## Error Handling

### Python Exceptions

```python
import hyperstreamdb as hdb

try:
    distance = hdb.l2_distance(vec1, vec2)
except ValueError as e:
    # Dimension mismatch, NaN/inf values, invalid input
    print(f"Invalid input: {e}")
except TypeError as e:
    # Invalid input types
    print(f"Type error: {e}")
except RuntimeError as e:
    # GPU errors, backend not available
    print(f"Runtime error: {e}")
except MemoryError as e:
    # GPU out of memory
    print(f"Memory error: {e}")
```

### GPU Fallback

```python
import hyperstreamdb as hdb

# GPU operations automatically fall back to CPU on error
ctx = hdb.GPUContext.auto_detect()
try:
    distances = hdb.l2_distance_batch(query, database, context=ctx)
except RuntimeError as e:
    print(f"GPU error, falling back to CPU: {e}")
    distances = hdb.l2_distance_batch(query, database)  # No context = CPU
```

## Performance Tips

1. **Use GPU for large batches**: 10,000+ vectors for best speedup
2. **Reuse GPU context**: Don't create new context for each operation
3. **Use float32**: Better GPU performance than float64
4. **Batch operations**: Process multiple queries together
5. **Use sparse vectors**: For high-dimensional sparse data
6. **Use binary vectors**: For binary features (SimHash, LSH)
7. **Profile workload**: Use `ctx.get_stats()` to monitor GPU usage

## See Also

- [Python Vector API](PYTHON_VECTOR_API.md) - Complete Python API reference
- [GPU Setup Guide](GPU_SETUP_GUIDE.md) - GPU installation and configuration
- [pgvector SQL Guide](PGVECTOR_SQL_GUIDE.md) - SQL syntax for vector operations
- [Vector Configuration](VECTOR_CONFIGURATION.md) - Index tuning and optimization
- [Benchmarking Guide](BENCHMARKING.md) - Performance testing
