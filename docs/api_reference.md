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
hdb.set_global_gpu_context(ctx)
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

```python
import hyperstreamdb as hdb

# Auto-detect best backend
ctx = hdb.GPUContext.auto_detect()

# Specify backend explicitly
ctx = hdb.GPUContext("cuda", device_id=0)
ctx = hdb.GPUContext("rocm", device_id=0)
ctx = hdb.GPUContext("mps")
ctx = hdb.GPUContext("intel")

# List available backends
backends = ctx.list_available_backends()

# Performance monitoring
stats = ctx.get_stats()
print(f"GPU time: {stats['total_gpu_time_ms']}ms")
print(f"Kernel launches: {stats['kernel_launches']}")
ctx.reset_stats()
```

See [GPU Setup Guide](GPU_SETUP_GUIDE.md) for installation and configuration.

### Vector Index Configuration

```python
import hyperstreamdb as hdb

# Configure HNSW index parameters
table = hdb.Table("s3://bucket/table")
table.create_vector_index(
    column="embedding",
    metric="cosine",
    m=16,              # Number of connections per layer
    ef_construction=200 # Size of dynamic candidate list
)
```

See [Vector Configuration Guide](VECTOR_CONFIGURATION.md) for tuning parameters.

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
