# Python Vector Distance API Reference

## Overview

BenoStreamDB provides a comprehensive Python API for vector distance computations with GPU acceleration support across multiple hardware backends. This API allows you to compute distances between vectors directly from Python without writing SQL queries, with optional GPU acceleration for high-performance batch operations.

> [!NOTE]
> This guide covers **standalone** distance functions for CPU/GPU. For persistent vector storage and search with TurboQuant indexing (TQ8/TQ4), please see [Vector Configuration](CONFIGURATION.md).

## Supported Distance Metrics

The API supports six distance metrics:

| Metric | Function | Description | Use Case |
|--------|----------|-------------|----------|
| **L2 (Euclidean)** | `l2_distance()` | √Σ(a-b)² | General-purpose similarity |
| **Cosine** | `cosine_distance()` | 1 - (a·b)/(‖a‖‖b‖) | Text embeddings, normalized vectors |
| **Inner Product** | `inner_product()` | -a·b | Maximum similarity search |
| **L1 (Manhattan)** | `l1_distance()` | Σ\|a-b\| | Robust to outliers |
| **Hamming** | `hamming_distance()` | Count of differing bits | Binary vectors, hashing |
| **Jaccard** | `jaccard_distance()` | 1 - \|A∩B\|/\|A∪B\| | Set similarity |

## GPU Backend Support

### Supported Hardware

| Backend | Hardware | Platform | Status |
|---------|----------|----------|--------|
| **CUDA** | NVIDIA GPUs | Linux, WSL2 | ✅ Supported |
| **ROCm** | AMD GPUs | Linux | ✅ Supported |
| **Metal (MPS)** | Apple Silicon | macOS | ✅ Supported |
| **Intel XPU** | Intel GPUs | Linux | ✅ Supported |
| **CPU** | All platforms | Fallback | ✅ Always available |

### Backend Priority

When using `GPUContext.auto_detect()`, backends are selected in this priority order:
1. CUDA (NVIDIA)
2. Metal/MPS (Apple Silicon)
3. ROCm (AMD)
4. Intel XPU (via WGPU)
5. CPU (fallback)

## Installation

### Basic Installation

```bash
pip install benostreamdb
```

### GPU Backend Requirements

#### NVIDIA CUDA

**Requirements:**
- NVIDIA GPU with compute capability 6.0+ (Pascal or newer)
- CUDA Toolkit 11.0 or later
- NVIDIA driver 450.80.02 or later

**Installation (Linux):**
```bash
# Ubuntu/Debian
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-12-3

# Verify installation
nvidia-smi
nvcc --version
```

**Installation (Windows via WSL2):**
1. Install WSL2 and a Linux distribution (e.g., Ubuntu 22.04)
2. Install NVIDIA drivers on the Windows host
3. Install NVIDIA Container Toolkit or the CUDA Toolkit within WSL2
4. Follow the Linux installation instructions above within your WSL2 environment

#### AMD ROCm

**Requirements:**
- AMD GPU (Radeon RX 5000 series or newer, or Radeon Instinct)
- ROCm 5.0 or later
- Linux only (Ubuntu 20.04/22.04, RHEL 8/9)

**Installation (Ubuntu):**
```bash
# Add ROCm repository
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_5.7.50700-1_all.deb
sudo apt-get install ./amdgpu-install_5.7.50700-1_all.deb

# Install ROCm
sudo amdgpu-install --usecase=rocm

# Add user to video and render groups
sudo usermod -a -G video,render $USER

# Verify installation
rocm-smi
```

#### Apple Metal (MPS)

**Requirements:**
- Apple Silicon Mac (M1, M2, M3, M4, M5, or newer)
- macOS 12.3 or later
- No additional installation required

**Verification:**
```python
import benostreamdb as bsdb
ctx = bsdb.GPUContext.auto_detect()
print(ctx.backend)  # Should show "mps" on Apple Silicon
```

#### Intel XPU (WGPU)

**Requirements:**
- Intel GPU (Iris Xe or newer recommended)
**Installation (Linux / WSL2):**
Intel hardware is supported natively via WGPU. Ensure you have modern graphics drivers and the `vulkan-loader` installed.
```bash
# Ubuntu/Debian
sudo apt-get install vulkan-tools
vulkaninfo | grep vendor
```

## Quick Start

### Basic Distance Computation

```python
import benostreamdb as bsdb
import numpy as np

# Create two vectors
vec1 = np.array([1.0, 2.0, 3.0])
vec2 = np.array([4.0, 5.0, 6.0])

# Compute L2 distance
distance = bsdb.l2_distance(vec1, vec2)
print(f"L2 distance: {distance}")

# Compute cosine distance
distance = bsdb.cosine_distance(vec1, vec2)
print(f"Cosine distance: {distance}")
```

### GPU-Accelerated Batch Operations

```python
import benostreamdb as bsdb
import numpy as np

# Create GPU context (auto-detect best backend)
ctx = bsdb.GPUContext.auto_detect()
print(f"Using backend: {ctx.backend}")

# Create query vector and database
query = np.random.randn(768).astype(np.float32)
database = np.random.randn(100000, 768).astype(np.float32)

# Compute distances on GPU (10x+ faster for large databases)
distances = bsdb.l2_distance_batch(query, database, context=ctx)

# Find top-k nearest neighbors
k = 10
top_k_indices = np.argsort(distances)[:k]
top_k_distances = distances[top_k_indices]

print(f"Top {k} nearest neighbors:")
for idx, dist in zip(top_k_indices, top_k_distances):
    print(f"  Index {idx}: distance {dist:.4f}")
```

### Sparse Vector Operations

```python
import benostreamdb as bsdb
import numpy as np

# Create sparse vectors (only store non-zero elements)
# Useful for high-dimensional sparse data (e.g., TF-IDF, bag-of-words)
sparse1 = bsdb.SparseVector(
    indices=np.array([0, 5, 100, 500], dtype=np.int32),
    values=np.array([1.0, 2.5, 0.8, 3.2], dtype=np.float32),
    dim=1000
)

sparse2 = bsdb.SparseVector(
    indices=np.array([5, 50, 100, 600], dtype=np.int32),
    values=np.array([2.0, 1.5, 0.9, 2.1], dtype=np.float32),
    dim=1000
)

# Compute sparse distance (only processes non-zero elements)
distance = bsdb.l2_distance_sparse(sparse1, sparse2)
print(f"Sparse L2 distance: {distance}")

# Convert to dense if needed
dense1 = sparse1.to_dense()
```

### Binary Vector Operations

```python
import benostreamdb as bsdb
import numpy as np

# Binary vectors for efficient similarity search
# Each bit represents a feature (e.g., SimHash, LSH)

# Create bit-packed binary vectors (8 bits per byte)
binary1 = np.packbits(np.random.randint(0, 2, 128))  # 128 bits = 16 bytes
binary2 = np.packbits(np.random.randint(0, 2, 128))

# Compute Hamming distance (counts differing bits)
distance = bsdb.hamming_distance_packed(binary1, binary2)
print(f"Hamming distance: {distance} bits differ")

# Compute Jaccard distance for binary vectors
distance = bsdb.jaccard_distance_packed(binary1, binary2)
print(f"Jaccard distance: {distance}")

# Auto-packing: provide unpacked binary vectors (0/1 values)
# The API will automatically pack them for efficiency
unpacked1 = np.random.randint(0, 2, 128, dtype=np.uint8)
unpacked2 = np.random.randint(0, 2, 128, dtype=np.uint8)
distance = bsdb.hamming_distance(unpacked1, unpacked2)  # Auto-packed internally
```

## API Reference

### Distance Functions

#### Single-Pair Distance Functions

All single-pair functions accept two vectors and return a scalar distance:

```python
l2_distance(vec1, vec2, context=None) -> float
cosine_distance(vec1, vec2, context=None) -> float
inner_product(vec1, vec2, context=None) -> float
l1_distance(vec1, vec2, context=None) -> float
hamming_distance(vec1, vec2, context=None) -> float
jaccard_distance(vec1, vec2, context=None) -> float
```

**Parameters:**
- `vec1`, `vec2`: NumPy arrays, Python lists, or any array-like objects
- `context` (optional): `GPUContext` for GPU acceleration

**Returns:** `float` - The computed distance

**Raises:**
- `ValueError`: If vectors have different dimensions
- `ValueError`: If vectors contain NaN or infinite values
- `TypeError`: If input types are invalid

#### Batch Distance Functions

Compute distances between one query vector and multiple database vectors:

```python
l2_distance_batch(query, database, context=None) -> np.ndarray
cosine_distance_batch(query, database, context=None) -> np.ndarray
inner_product_batch(query, database, context=None) -> np.ndarray
l1_distance_batch(query, database, context=None) -> np.ndarray
hamming_distance_batch(query, database, context=None) -> np.ndarray
jaccard_distance_batch(query, database, context=None) -> np.ndarray
```

**Parameters:**
- `query`: 1D NumPy array (shape: `[dim]`)
- `database`: 2D NumPy array (shape: `[n_vectors, dim]`)
- `context` (optional): `GPUContext` for GPU acceleration

**Returns:** `np.ndarray` - 1D array of distances (shape: `[n_vectors]`)

**Performance:** GPU acceleration provides 10x+ speedup for databases with 100,000+ vectors

#### Sparse Distance Functions

Efficient distance computation for sparse vectors:

```python
l2_distance_sparse(sparse1, sparse2, context=None) -> float
cosine_distance_sparse(sparse1, sparse2, context=None) -> float
inner_product_sparse(sparse1, sparse2, context=None) -> float
```

**Parameters:**
- `sparse1`, `sparse2`: `SparseVector` objects
- `context` (optional): `GPUContext` for GPU acceleration

**Returns:** `float` - The computed distance

#### Binary Distance Functions

Efficient distance computation for bit-packed binary vectors:

```python
hamming_distance_packed(binary1, binary2, context=None) -> int
jaccard_distance_packed(binary1, binary2, context=None) -> float
```

**Parameters:**
- `binary1`, `binary2`: NumPy uint8 arrays (bit-packed)
- `context` (optional): `GPUContext` for GPU acceleration

**Returns:** Distance value (int for Hamming, float for Jaccard)

### GPU Context Management

#### GPUContext Class

```python
class GPUContext:
    """Manages GPU backend selection and device configuration."""
    
    @staticmethod
    def auto_detect() -> GPUContext:
        """Detect and return the highest-priority available GPU backend."""
    
    def __init__(self, backend: str, device_id: int = 0):
        """
        Create GPU context with specific backend.
        
        Args:
            backend: Backend name ("cuda", "rocm", "mps", "intel", "cpu")
            device_id: GPU device ID for multi-GPU systems (default: 0)
        
        Raises:
            RuntimeError: If backend is not available
        """
    
    @property
    def backend(self) -> str:
        """Get current backend name."""
    
    @property
    def device_id(self) -> int:
        """Get current device ID."""
    
    def list_available_backends(self) -> list[str]:
        """List all detected GPU backends."""
    
    def get_stats(self) -> dict:
        """
        Get performance statistics.
        
        Returns:
            dict with keys:
                - total_gpu_time_ms: Total GPU computation time
                - kernel_launches: Number of GPU kernel launches
                - vectors_processed: Total vectors processed
        """
    
    def reset_stats(self):
        """Clear all accumulated performance metrics."""
```

#### Usage Examples

```python
# Auto-detect best backend
ctx = bsdb.GPUContext.auto_detect()

# Create specific backend
ctx = bsdb.GPUContext("cuda", device_id=0)

# List available backends
backends = ctx.list_available_backends()
print(f"Available backends: {backends}")

# Monitor performance
distances = bsdb.l2_distance_batch(query, database, context=ctx)
stats = ctx.get_stats()
print(f"GPU time: {stats['total_gpu_time_ms']}ms")
print(f"Kernel launches: {stats['kernel_launches']}")

# Reset stats for next benchmark
ctx.reset_stats()
```

### Sparse Vector Class

```python
class SparseVector:
    """Sparse vector representation storing only non-zero elements."""
    
    def __init__(self, indices: np.ndarray, values: np.ndarray, dim: int):
        """
        Create sparse vector.
        
        Args:
            indices: 1D int32 array of non-zero indices (must be sorted)
            values: 1D float32 array of non-zero values
            dim: Total dimension of the vector
        
        Raises:
            ValueError: If indices are not sorted or out of bounds
        """
    
    @property
    def indices(self) -> np.ndarray:
        """Get non-zero indices."""
    
    @property
    def values(self) -> np.ndarray:
        """Get non-zero values."""
    
    @property
    def dim(self) -> int:
        """Get total dimension."""
    
    def to_dense(self) -> np.ndarray:
        """Convert to dense NumPy array."""
```

## Performance Optimization

### When to Use GPU Acceleration

GPU acceleration provides significant speedups for:
- **Batch operations** with 10,000+ vectors
- **High-dimensional vectors** (512+ dimensions)
- **Repeated queries** on the same database

For small operations (< 1,000 vectors), CPU may be faster due to GPU transfer overhead.

### Memory Considerations

GPU memory limits:
- **NVIDIA RTX 3090**: 24 GB
- **Apple M1 Max**: 32-64 GB (shared)
- **AMD RX 7900 XTX**: 24 GB

For large databases, consider:
1. **Batch processing**: Process database in chunks
2. **Sparse vectors**: Reduce memory usage for sparse data
3. **Binary vectors**: Use bit-packing for binary features

### Best Practices

```python
# ✅ Good: Reuse GPU context
ctx = bsdb.GPUContext.auto_detect()
for query in queries:
    distances = bsdb.l2_distance_batch(query, database, context=ctx)

# ❌ Bad: Create new context each time
for query in queries:
    ctx = bsdb.GPUContext.auto_detect()  # Overhead!
    distances = bsdb.l2_distance_batch(query, database, context=ctx)

# ✅ Good: Use appropriate data types
query = np.array(data, dtype=np.float32)  # float32 is faster on GPU

# ❌ Bad: Use float64 unnecessarily
query = np.array(data, dtype=np.float64)  # Slower and uses more memory
```

## Integration with SQL

The Python API shares the same GPU context with SQL queries:

```python
import benostreamdb as bsdb

# Set global GPU context
ctx = bsdb.GPUContext.auto_detect()
bsdb.set_thread_gpu_context(ctx)

# SQL queries now use GPU acceleration
session = bsdb.Session()
session.register("documents", table)

results = session.sql("""
    SELECT id, content,
           embedding <-> '[0.1, 0.2, 0.3]'::vector AS distance
    FROM documents
    ORDER BY distance
    LIMIT 10
""")

# Check GPU usage
stats = ctx.get_stats()
print(f"GPU time: {stats['total_gpu_time_ms']}ms")
```

## Troubleshooting

### GPU Not Detected

```python
ctx = bsdb.GPUContext.auto_detect()
print(ctx.backend)  # Shows "cpu" instead of GPU backend
```

**Solutions:**
1. Verify GPU drivers are installed: `nvidia-smi`, `rocm-smi`, or check System Preferences on macOS
2. Check backend availability: `ctx.list_available_backends()`
3. Verify CUDA/ROCm installation: `nvcc --version` or `rocminfo`

### Out of Memory Errors

```python
# Error: RuntimeError: GPU out of memory
distances = bsdb.l2_distance_batch(query, huge_database, context=ctx)
```

**Solutions:**
```python
# Process in chunks
chunk_size = 10000
all_distances = []
for i in range(0, len(database), chunk_size):
    chunk = database[i:i+chunk_size]
    distances = bsdb.l2_distance_batch(query, chunk, context=ctx)
    all_distances.append(distances)
all_distances = np.concatenate(all_distances)
```

### Dimension Mismatch

```python
# Error: ValueError: Vector dimensions must match
distance = bsdb.l2_distance(vec1, vec2)
```

**Solutions:**
```python
# Check dimensions
print(f"vec1 shape: {vec1.shape}, vec2 shape: {vec2.shape}")

# Ensure same dimension
assert vec1.shape == vec2.shape
```

## Examples

See [examples/python_distance_api_examples.py](../examples/python_distance_api_examples.py) for complete working examples.

## See Also

- [pgvector SQL Guide](PGVECTOR_SQL_GUIDE.md) - SQL syntax for vector operations
- [Vector Configuration](CONFIGURATION.md) - Index configuration and tuning
- [Benchmarking Guide](BENCHMARKING.md) - Performance testing and optimization


---

## API reference

_(merged from the former `api_reference.md`)_


This page provides an overview of BenoStreamDB APIs across different languages and interfaces.

## Python API

### Table Operations

```python
import benostreamdb as bsdb

# Create/open table
table = bsdb.Table("s3://bucket/my-table")

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
import benostreamdb as bsdb
import numpy as np

# Single-pair distance
distance = bsdb.l2_distance(vec1, vec2)
distance = bsdb.cosine_distance(vec1, vec2)

# Batch operations with GPU acceleration
ctx = bsdb.GPUContext.auto_detect()
distances = bsdb.l2_distance_batch(query, database, context=ctx)

# Sparse vectors
sparse = bsdb.SparseVector(indices, values, dim)
distance = bsdb.l2_distance_sparse(sparse1, sparse2)

# Binary vectors
distance = bsdb.hamming_distance_packed(binary1, binary2)
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
import benostreamdb as bsdb

# Create session
session = bsdb.Session()
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
ctx = bsdb.GPUContext.auto_detect()
bsdb.set_thread_gpu_context(ctx)
```

See [pgvector SQL Guide](PGVECTOR_SQL_GUIDE.md) for SQL syntax reference.

### Iceberg V2/V3 API

```python
import benostreamdb as bsdb

table = bsdb.Table("s3://bucket/table")

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
  .format("benostream")
  .option("path", "s3://bucket/table")
  .load()

// Write
df.write
  .format("benostream")
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
SELECT * FROM benostream.default.my_table
WHERE id > 100;

-- Vector search with pgvector operators
SELECT id, content,
       embedding <-> ARRAY[0.1, 0.2, 0.3] AS distance
FROM benostream.default.documents
WHERE category = 'science'
ORDER BY distance
LIMIT 10;
```

## Configuration

### GPU Context Configuration

BenoStreamDB supports GPU acceleration across NVIDIA CUDA, AMD ROCm, Apple Metal (MPS), and Intel XPU. You can configure execution devices using PyTorch-style strings (`"cuda:0"`) or explicit parameters (`device_id=0` / `index=0`).

```python
import benostreamdb as bsdb

# 1. Auto-detect best available GPU backend
device = bsdb.Device("auto")
print(f"Auto-selected: {device.backend}, device_id: {device.index}")

# 2. Specify backend and device index explicitly
device = bsdb.Device("cuda:0")              # First NVIDIA GPU
device = bsdb.Device("cuda", index=1)       # Second NVIDIA GPU
device = bsdb.Device("rocm:0")              # AMD ROCm GPU
device = bsdb.Device("mps")                 # Apple Silicon (always index 0)
device = bsdb.Device("xpu:0")               # Intel discrete / integrated GPU
device = bsdb.Device("cpu")                 # Force CPU execution

# Or using GPUContext API
ctx = bsdb.GPUContext("cuda", device_id=0)

# 3. Query system availability & performance stats
print("Available backends:", bsdb.Device.list_available_backends())

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
import benostreamdb as bsdb

table = bsdb.Table("s3://bucket/table")

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

See [Vector Configuration Guide](CONFIGURATION.md) for tuning parameters.

## REST Search APIs (`benostreamdb-search`)

BenoStreamDB provides a dual-protocol HTTP gateway exposing OpenSearch / Elasticsearch 7.10 compatibility alongside the Qdrant Vector API from a single server:

### Starting the Server
```bash
cargo run -p benostreamdb-search --bin bsdb-search
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
| `GET` | `/` | Service/version info. |
| `GET` | `/healthz`, `/livez`, `/readyz` | Liveness/readiness probes. |
| `GET` | `/telemetry` | Basic telemetry JSON. |
| `GET` | `/collections` | List available vector collections. |
| `GET` | `/collections/{name}/exists` | Collection existence. |
| `GET` | `/collections/{name}` | Collection info (real point count, vector size, distance). |
| `PUT` | `/collections/{name}` | Create a vector collection with vector parameters (dimension, distance). |
| `PATCH` | `/collections/{name}` | Update collection parameters. |
| `DELETE` | `/collections/{name}` | Drop the collection. |
| `PUT` | `/collections/{name}/index` | Create a payload index (accepted). |
| `DELETE` | `/collections/{name}/index/{field}` | Delete a payload index (accepted). |
| `PUT` | `/collections/{name}/points` | Upsert vector points with payload metadata. |
| `GET`/`POST` | `/collections/{name}/points` | Retrieve points by id. |
| `GET` | `/collections/{name}/points/{id}` | Retrieve a single point. |
| `POST` | `/collections/{name}/points/search` | Vector similarity search with optional payload filters. |
| `POST` | `/collections/{name}/points/query` | Universal query API (`query: [..]` or `{nearest: [..]}`). |
| `POST` | `/collections/{name}/points/scroll` | Paginated point listing. |
| `POST` | `/collections/{name}/points/count` | Count points (optionally filtered). |
| `POST` | `/collections/{name}/points/recommend` | Recommendation by example vectors. |
| `POST` | `/collections/{name}/points/discover` | Discovery with context pairs. |
| `POST` | `/collections/{name}/points/batch` | Batched write operations. |
| `POST`/`PUT` | `/collections/{name}/points/payload` | Set (merge) / overwrite payload. |
| `POST` | `/collections/{name}/points/payload/delete` | Delete payload keys. |
| `POST` | `/collections/{name}/points/payload/clear` | Clear payload. |
| `PUT` | `/collections/{name}/points/vectors` | Update point vectors. |
| `POST` | `/collections/{name}/points/delete` | Delete by ids and/or filter. |
| `GET`/`POST` | `/collections/aliases` | List / create / delete / rename aliases. |

> [!NOTE]
> See [QDRANT_COMPATIBILITY.md](QDRANT_COMPATIBILITY.md) for the full support
> matrix, score semantics, and known approximations.

## Error Handling

### Python Exceptions

```python
import benostreamdb as bsdb

try:
    distance = bsdb.l2_distance(vec1, vec2)
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
import benostreamdb as bsdb

# GPU operations automatically fall back to CPU on error
ctx = bsdb.GPUContext.auto_detect()
try:
    distances = bsdb.l2_distance_batch(query, database, context=ctx)
except RuntimeError as e:
    print(f"GPU error, falling back to CPU: {e}")
    distances = bsdb.l2_distance_batch(query, database)  # No context = CPU
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
- [Vector Configuration](CONFIGURATION.md) - Index tuning and optimization
- [Benchmarking Guide](BENCHMARKING.md) - Performance testing
