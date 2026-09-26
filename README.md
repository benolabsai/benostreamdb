<p align="center">
  <img src="BenoStreamDB.png" alt="BenoStreamDB Logo" width="300">
</p>

# BenoStreamDB
**Serverless Index-Streaming Database with Overlay Indexing**

An **index-overlay engine for the lakehouse**. BenoStreamDB layers reconstructible, persistent secondary indexes — scalar bitmaps, BM25 Okapi full-text, HNSW/IVF vector search, and CSR graph indexes — onto Parquet data that already lives in object storage, and exposes all of it through standard SQL (DataFusion) with pgvector-compatible syntax.

It is not a storage engine you have to migrate into. You can either write through it, or point it at an **existing Apache Iceberg table you do not own** and build indexes over that data in place. The authoritative Parquet files and the advisory index overlays are stored separately, so indexing never rewrites or duplicates your data.

## 🎯 Architecture: The Indexed Lakehouse

BenoStreamDB implements an indexed, compute-disaggregated lakehouse architecture that pairs authoritative open table storage with advisory, persistent secondary indexes and a unified retrieval layer. The indexes are **overlays**: they attach to data files as sidecar artifacts and can be layered onto tables BenoStreamDB did not create (see [Layered Indexing](#layered-indexing-existing-iceberg-tables) below).

```text
               Iceberg Table
                     │
       ┌─────────────┴──────────────┐
       │                            │
Authoritative Storage        Advisory Index Overlay
       │                            │
  Parquet Files              Scalar Bitmap / BM25 / HNSW / IVF / TQ / CSR Graph
```

> ### Core Architecture Invariants
> 1. **The Overlay Invariant**: Index files are derived, reconstructible state. They may be absent, stale, or deleted without compromising snapshot correctness. Queries may degrade to Parquet scanning or background recovery, but never return incorrect results.
> 2. **Publication Invariant**: A published manifest may reference only immutable artifacts that have already been successfully uploaded and verified to storage.
> 3. **Durability Invariant**: WAL truncation is permitted only after the corresponding data is durably represented by a committed manifest snapshot.
> 4. **Maintenance Invariant**: Maintenance operations may delete an artifact only if it is neither referenced by any active snapshot nor currently in-flight.
> 5. **Resource Invariant**: Index builds are gated to a memory-scaled concurrency budget, and production paths follow a no-panic policy — malformed input degrades to an error, never a crash.

BenoStreamDB sits between two categories of system that do not overlap. Vector/graph databases own your data and cannot do general SQL analytics over a lakehouse; lakehouse query engines do SQL over Iceberg/Parquet but have no native vector, full-text, or graph indexing. BenoStreamDB fills that seam:

| Capability | Vector/Graph DBs (Pinecone, Qdrant, Neo4j) | Lakehouse Engines (Trino, Spark, DuckDB) | BenoStreamDB |
|------------|--------------------------------------------|------------------------------------------|--------------|
| **Data ownership** | Must ingest/duplicate your data | Queries data in place | Queries data in place |
| **Vector search** | ✅ Native | ❌ Bolt-on | ✅ Native overlay index |
| **Full-text (BM25)** | Partial | ❌ Bolt-on | ✅ Native overlay index |
| **Scalar indexes** | Partial | ❌ Bolt-on | ✅ RoaringBitmap |
| **Graph analytics** | ✅ (separate system) | ❌ | ✅ SQL UDFs on edge tables |
| **SQL analytics** | ❌ | ✅ | ✅ DataFusion |
| **pgvector syntax** | ❌ | ❌ | ✅ |
| **Indexes on existing Iceberg tables** | ❌ | ❌ | ✅ Layered Indexing |
| **Deployment** | Managed cluster | Cluster | Embedded library / scale-to-zero |

## ⚡ Iceberg V2/V3 Compatibility

BenoStreamDB implements the core Apache Iceberg table format V2 and V3 features required for its indexed-lakehouse model:

| Feature | V1 | V2 | V3 | BenoStreamDB |
|---------|----|----|----|--------------| 
| **Sort Orders** | ❌ | ✅ | ✅ | ✅ Implemented |
| **Partition Evolution** | ❌ | ✅ | ✅ | ✅ Implemented |
| **Statistics (NDV)** | ❌ | ✅ | ✅ | ✅ Parquet `distinct_count` |
| **Row Lineage** | ❌ | ❌ | ✅ | ✅ `_row_id` (monotonic long), `_last_updated_sequence_number`, `next-row-id`, `first-row-id` |
| **Default Values** | ❌ | ❌ | ✅ | ✅ `initial-default` & `write-default` |
| **Deletion Vectors** | ❌ | ❌ | ✅ | ✅ Puffin Format Integrated |
| **Delete Files** | ❌ | ✅ | ✅ | ✅ Position + Equality Deletes |
| **Nanosecond Timestamps** | ❌ | ❌ | ✅ | ✅ `timestamp(ns)` & `time64(ns)` |

### New APIs

```python
import benostreamdb as bsdb

# Create table with sort order (V2)
table = bsdb.Table("s3://bucket/table")
table.replace_sort_order(["timestamp", "user_id"], ascending=[False, True])

# V3 tables emit row lineage: _row_id (monotonic long) and
# _last_updated_sequence_number, with next-row-id / first-row-id tracked
# in table metadata. Enable with table.format_version = 3.
table.format_version = 3
```

### Migration Guide: V2 → V3

Upgrading to V3 enables row-level operations and enhanced tracking:

1. **Enable**: set `table.format_version = 3` (persisted on the next commit)
2. **No Data Rewrite**: Existing data remains compatible
3. **Row lineage**: new rows get a monotonic `_row_id` (long) and `_last_updated_sequence_number`; each data file records its `first-row-id` and table metadata tracks `next-row-id`


## 🌐 REST APIs (OpenSearch & Qdrant)

BenoStreamDB includes a highly optimized HTTP frontend (`benostreamdb-search`) that exposes the core engine over standard REST protocols. By translating incoming requests into native BenoStreamDB columnar operations, it allows you to use existing tools without running traditional clustered databases.

- **OpenSearch / Elasticsearch 7.10 API (Port 9200)**: Drop-in compatibility for standard text indexing, bulk writes, and keyword search. (e.g., connect Kibana or Grafana directly). See the [OpenSearch compatibility matrix](docs/OPENSEARCH_COMPATIBILITY.md).
- **Qdrant Vector API (Port 6333)**: Qdrant v1.x REST emulation covering collections, points, payloads, vectors, aliases, and the universal query API. Qdrant's unstructured JSON payloads are dynamically inferred and converted into highly compressed Arrow columns on write. See the [Qdrant compatibility matrix](docs/QDRANT_COMPATIBILITY.md) for the exact supported surface and known approximations.

Both APIs are hosted concurrently from a single binary, completely share the exact same underlying `AppState` and data files, and require zero data duplication. You can write a collection of embeddings via the Qdrant API and instantly query it via the OpenSearch API!

To start the dual-API server:
```bash
# Uses BENOSEARCH_PORT=9200 and BENOSEARCH_QDRANT_PORT=6333 by default
cargo run -p benostreamdb-search
```

## 🚀 Quick Start

### 🐳 Docker Quickstart (3 Minutes to First Query)

Run the full BenoStreamDB gateway stack with one command:

```bash
# Standalone All-in-One Container (Local storage)
docker run -d --name benostreamdb \
  -p 9200:9200 \
  -p 6333:6333 \
  -p 50051:50051 \
  benostreamdb/quickstart:latest

# Or Full-Stack Compose (MinIO S3 + Nessie Catalog + BenoStreamDB)
docker compose -f docker/docker-compose.quickstart.yml up -d
```

| Service | Protocol | Port | Description |
| :--- | :--- | :--- | :--- |
| **Elasticsearch 7.10** | REST / JSON | `9200` | Text indexing, BM25, and hybrid search |
| **Qdrant Vector** | REST / JSON | `6333` | Collections, point upsert/retrieve, payload & vector edits, vector search, aliases |
| **Arrow Flight SQL** | gRPC / Flight | `50051` | Zero-copy SQL for DuckDB, Polars, BI tools |

Verify cluster health:
```bash
curl http://localhost:9200/_cluster/health
```

#### GPU-Accelerated Docker (NVIDIA CUDA, AMD ROCm, Intel XPU)
Run with hardware acceleration across NVIDIA, AMD, or Intel GPUs:
```bash
# Launch with GPU override:
docker compose -f docker/docker-compose.quickstart.yml -f docker/docker-compose.gpu.yml up -d

# Verify compute engine reported by the Search API:
curl -s http://localhost:9200/
```
Output:
```json
{
  "name": "bsdb-search-1",
  "cluster_name": "bsdb-search",
  "version": { "number": "7.10.2", ... },
  "compute": {
    "backend": "cuda",
    "device_id": 0,
    "gpu_accelerated": true,
    "available": true
  },
  "tagline": "You know, you search"
}
```


### Python Installation

**Standard Install (CPU + WGPU/Vulkan):**
The default package includes automatic high-performance hardware detection for NVIDIA CUDA, Apple Metal, Intel Graphics/XPU, and AMD ROCm.

```bash
pip install benostreamdb
```

**Windows Users:**
BenoStreamDB is optimized for Linux/POSIX. Windows users should use **WSL2**.


### GPU Acceleration (Optional)

For GPU-accelerated vector operations, install the appropriate backend:

**NVIDIA CUDA:**
```bash
# Ubuntu/Debian
sudo apt-get install cuda-toolkit-12-3
# Verify: nvidia-smi
```

**AMD ROCm:**
ROCm support is now native on Linux via WGPU/Vulkan.
```bash
# Verify Vulkan support (standard in modern ROCm drivers)
vulkaninfo | grep vendor
# Verify: rocm-smi
```

**Apple Metal:**
- Included with macOS 12.3+ on Apple Silicon (M1, M2, M3, M4, M5)
- No additional installation required

**Intel XPU / Graphics:**
Intel Arc and Data Center GPUs are supported natively on Linux.
```bash
# Verify intel-media-va-driver or similar is present
clinfo | grep Intel
```

See [Python Vector API Documentation](docs/PYTHON_VECTOR_API.md) for detailed GPU setup instructions.

### pgvector SQL Compatibility

BenoStreamDB provides full pgvector-compatible SQL syntax for vector operations:

```sql
-- Use familiar pgvector operators
SELECT id, content, 
       embedding <-> '[0.1, 0.2, 0.3]'::vector AS l2_distance,
       embedding <=> '[0.1, 0.2, 0.3]'::vector AS cosine_distance
FROM documents
WHERE category = 'science'
ORDER BY l2_distance
LIMIT 10;

-- All six distance operators supported
-- <->  L2 (Euclidean)
-- <=>  Cosine  
-- <#>  Inner Product
-- <+>  L1 (Manhattan)
-- <~>  Hamming (direct on ::vector)
-- <%>  Jaccard (direct on ::vector)
```

> **💡 pgvector Compatibility Note on `<~>` (Hamming) & `<%>` (Jaccard):**  
> In BenoStreamDB, `<~>` and `<%>` operate directly on standard float `::vector` embeddings (evaluating binary indicator sets and quantized vectors) for developer convenience. In upstream PostgreSQL `pgvector`, these two operators are restricted exclusively to the `bit` data type.  
> 
> **PostgreSQL Conversion Equivalent:**
> ```sql
> -- BenoStreamDB:
> SELECT * FROM documents ORDER BY embedding <~> '[1, 0, 1]'::vector LIMIT 10;
> 
> -- PostgreSQL (pgvector 0.7.0+): requires binary_quantize() to produce bit types
> SELECT * FROM documents ORDER BY binary_quantize(embedding) <~> binary_quantize('[1, 0, 1]'::vector) LIMIT 10;
> ```

See [pgvector SQL Guide](docs/PGVECTOR_SQL_GUIDE.md) for complete documentation and conversion guide.

### Basic Usage

```python
import benostreamdb as bsdb

# Create table
table = bsdb.Table("s3://bucket/my-table")

# Write data (Pandas/PyArrow)
import pandas as pd
df = pd.DataFrame({
    "id": [1, 2, 3],
    "embedding": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
})
table.write_pandas(df)
table.commit()

# Create high-performance vector index (TQ8 - 4x compression)
table.add_index("embedding", "hnsw_tq8")

# Query with filters (uses indexes!)
results = table.to_pandas(filter="id > 1")

# Vector search
query_vec = [0.15, 0.25]
results = table.to_pandas(
    vector_filter={"column": "embedding", "query": query_vec, "k": 10}
)

# Hybrid query (scalar + vector)
results = table.to_pandas(
    filter="category = 'science'",
    vector_filter={"column": "embedding", "query": query_vec, "k": 10}
)
```

## 🔄 Fluent Query API

BenoStreamDB features a fluent query API in Rust with method chaining. Python uses the `to_pandas()` API with filter and vector_filter arguments.

### Rust Fluent API

The same fluent interface is available in native Rust:

```rust
use benostreamdb::{Table, VectorValue};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let table = Table::new("s3://bucket/my-table".to_string())?;
    let query_vec = vec![0.1f32, 0.2, 0.3];

    // Method chaining
    let results = table
        .query()
        .filter("age > 25")
        .vector_search("embedding", VectorValue::Float32(query_vec), 10)
        .select(vec!["name".to_string(), "score".to_string()])
        .to_batches()
        .await?;
    
    println!("Found {} result batches", results.len());
    Ok(())
}
```

### Benefits

- **Method Chaining**: Intuitive, readable query construction  
- **Type Safe**: Compile-time validation in Rust, runtime validation in Python
- **Performance**: Same underlying optimized execution as traditional APIs
- **Interoperable**: Mix with SQL queries and traditional `to_pandas()` calls
- **GPU Acceleration**: Automatic GPU context propagation for vector operations
- **TurboQuant Optimized**: Seamless integration with 8-bit/4-bit quantization

### TurboQuant Quantization (TQ8 / TQ4)

BenoStreamDB features **TurboQuant**, an optimized quantization engine that reduces vector storage costs while maintaining high search accuracy:

- **TQ8 (8-bit)**: 4x compression vs. float32. Near-lossless accuracy (typically >99% recall retention). Ideal for general-purpose RAG.
- **TQ4 (4-bit)**: 8x compression vs. float32. Maximum efficiency for massive datasets where storage cost is the primary bottleneck.

```python
# High-performance community default (HNSW-TQ8)
table.add_index("embedding", "hnsw_tq8")

# High-compression mode
table.add_index("embedding", "hnsw_tq4")

# Custom HNSW-PQ configuration
table.add_index("embedding", {
    "type": "hnsw_pq",
    "complexity": 32,
    "quality": 300,
    "compression": 32 # PQ subspaces
})
```

### Python Vector Distance API with GPU Acceleration

BenoStreamDB provides a comprehensive Python API for vector distance computations with GPU acceleration:

```python
import benostreamdb as bsdb
import numpy as np

# GPU-accelerated batch distance computation
# `Device` is the canonical class; `GPUContext` / `ComputeContext` are aliases.
ctx = bsdb.Device.auto_detect()  # Auto-detect CUDA/ROCm/Metal/XPU
print(f"Using GPU backend: {ctx.backend}")

# Create query and database vectors
query = np.random.randn(768).astype(np.float32)
database = np.random.randn(100000, 768).astype(np.float32)

# Compute distances on GPU (10x+ faster for large databases)
distances = bsdb.l2_batch(query, database, device=ctx)

# Find top-k nearest neighbors
k = 10
top_k_indices = np.argsort(distances)[:k]

# Single-pair distance computation
vec1 = np.array([1.0, 2.0, 3.0])
vec2 = np.array([4.0, 5.0, 6.0])
distance = bsdb.cosine(vec1, vec2)

# Sparse vector support for high-dimensional sparse data
sparse1 = bsdb.SparseVector(
    indices=np.array([0, 5, 100], dtype=np.int32),
    values=np.array([1.0, 2.5, 0.8], dtype=np.float32),
    dim=1000
)
sparse2 = bsdb.SparseVector(
    indices=np.array([5, 50, 100], dtype=np.int32),
    values=np.array([2.0, 1.5, 0.9], dtype=np.float32),
    dim=1000
)
distance = bsdb.l2_sparse(sparse1, sparse2)

# Binary vector operations (bit-packed for efficiency)
binary1 = np.packbits(np.random.randint(0, 2, 128))
binary2 = np.packbits(np.random.randint(0, 2, 128))
distance = bsdb.hamming_packed(binary1, binary2)
```

**Supported GPU Backends:**
- **CUDA** - NVIDIA GPUs (Linux, Windows via WSL2)
- **ROCm** - AMD GPUs (Native Linux via WGPU)
- **Intel XPU** - Intel Graphics (Native Linux via WGPU)
- **Metal (MPS)** - Apple Silicon (macOS)
- **Torch Alignment** - Automatically aliases `cuda` to `rocm` on AMD hardware if `torch.version.hip` is detected.
- **CPU** - Fallback for all platforms

**Supported Distance Metrics:**
- L2 (Euclidean), Cosine, Inner Product, L1 (Manhattan), Hamming, Jaccard

See [Python Vector API Documentation](docs/PYTHON_VECTOR_API.md) for complete API reference and GPU installation instructions

### SQL queries (full DataFusion support with pgvector syntax)

```python
import benostreamdb as bsdb
session = bsdb.Session()
session.register("users", table)

# Optional: Enable GPU acceleration for SQL queries
device = bsdb.Device.auto_detect()
device.activate()

# Simple SQL (via table — registers as table 't')
results = table.execute_sql("SELECT * FROM t WHERE id > 100")

# Vector similarity search with pgvector operators (GPU-accelerated)
results = session.sql("""
    SELECT id, content,
           embedding <-> '[0.1, 0.2, 0.3]'::vector AS distance
    FROM documents
    WHERE category = 'science'
    ORDER BY distance
    LIMIT 10
""")

# Joins (uses Index Nested Loop Join optimization)
results = session.sql("""
    SELECT u.name, o.amount
    FROM users u
    JOIN orders o ON u.id = o.user_id
    WHERE u.category = 'premium'
""")

# Maintenance
table.compact()
```

## 📊 Performance

The only benchmark we stand behind is the **full-site English-Wikipedia Graph-RAG
demo** — end-to-end, reproducible, and measured on documented hardware. The
earlier OpenSearch / Elasticsearch / LanceDB comparisons, the NYC-Taxi micro-runs,
and the Criterion `benches/` targets have been **removed as stale**; if a number is
not below, treat it as unverified.

| Stage (whole enwiki: 51.8M live pages / 383M edges) | Wall time | Peak memory |
| :--- | :--- | :--- |
| embed (all-MiniLM-L6-v2, 384-d, RTX 3090) | 2.1 h | 8.3 GB RSS, 5.9 GB VRAM |
| load — nodes (51.8M + HNSW-TQ8 + BM25) | 41 min | 14.6–18.0 GB per chunk |
| load — edges (383M + CSR) | 271 s | ~6 GB |

Run it with `python scripts/prepare_demo.py`. Full per-stage timings, the
per-chunk node-load table, and reproduction commands live in
[`examples/web_ui/README.md`](examples/web_ui/README.md); methodology is in
[`docs/BENCHMARKING.md`](docs/BENCHMARKING.md).

## 🏗️ Architecture

### Overlay Indexing

BenoStreamDB stores indexes as **sidecar files** alongside Parquet data:

```
s3://bucket/table/
├── data/
│   ├── segment_001.parquet                   # Main Data (Parquet)
│   ├── segment_001.id.inv.parquet           # Scalar index (Inverted Parquet)
│   ├── segment_001.emb.centroids.parquet    # Vector index centroids
│   └── segment_001.emb.cluster_0.hnsw.graph # Vector index graph (HNSW)
├── _manifest/
│   ├── v1.avro                              # Manifest (Iceberg/Avro)
│   └── v2.avro
└── _metadata/
    └── v1.metadata.json
```

### Manifest Format

**Apache Iceberg V2/V3 compliant** (Avro encoding):

```json
{
  "version": 2,
  "timestamp_ms": 1705512000000,
  "entries": [
    {
      "file_path": "segment_001.parquet",
      "file_size_bytes": 104857600,
      "record_count": 1000000,
      "index_files": [
        {
          "file_path": "segment_001.id.inv.parquet",
          "index_type": "scalar",
          "column_name": "id"
        },
        {
          "file_path": "segment_001.embedding.cluster_0.hnsw.graph",
          "index_type": "vector",
          "column_name": "embedding"
        }
      ]
    }
  ],
  "prev_version": 1
}
```

### Layered Indexing (Existing Iceberg Tables)

Because indexes are overlays, BenoStreamDB can index data it does not own. `Table.register_external` points at an existing Iceberg table's metadata, maps its schema, imports the current snapshot, and builds index overlays over the referenced Parquet files — without rewriting, copying, or taking ownership of the data.

```python
import benostreamdb as bsdb

# Attach to an existing Iceberg table and index it in place.
# The first argument is where the overlay indexes are stored; the second is
# the existing table's Iceberg metadata.
table = bsdb.Table.register_external(
    "s3://my-index-bucket/overlays/events",
    "s3://my-lake/warehouse/db/events/metadata/v1.metadata.json",
)

# Build overlays over the existing data
table.add_index("embedding", "hnsw_tq8")
table.add_index("body", "bm25")
```

The authoritative Parquet files stay where they are. The overlays are derived, reconstructible state: if they are lost or stale, queries degrade to Parquet scanning and the indexes can be rebuilt (see the Overlay Invariant above).

## 🔌 Connectors

> [!NOTE]
> **MERGE INTO Support**
> BenoStreamDB implements `MERGE INTO` **natively in its SQL layer**. DataFusion's planner has no logical plan for `MERGE`, so the engine intercepts the parsed statement before planning and translates it into the key-based upsert (`Table::merge`, Merge-on-Read) and delete (`Table::delete_async`) primitives. Supported clauses:
>
> ```sql
> MERGE INTO target t USING source s ON t.id = s.id
> WHEN MATCHED AND t.flag THEN UPDATE SET name = s.name
> WHEN MATCHED THEN DELETE
> WHEN NOT MATCHED THEN INSERT (id, name) VALUES (s.id, s.name)
> WHEN NOT MATCHED THEN INSERT ROW
> WHEN NOT MATCHED BY SOURCE THEN DELETE;
> ```
>
> The `ON` condition must be an equi-join on the target's key column(s). Clauses are evaluated in order with first-match-wins semantics. The **Spark connector** additionally intercepts Spark's row-level `MERGE INTO` / `UPDATE` / `DELETE` operations (`BenoStreamMergeBuilder` → `BenoStreamRowLevelOperation`) and sends Iceberg position deletes plus appends through the JNI bridge. The **Trino connector** implements Trino's MERGE SPI (`beginMerge`/`finishMerge` + a `ConnectorMergeSink`) and a write path (`beginInsert` + `ConnectorPageSink`), so `INSERT` and `MERGE INTO` work through Trino as well.

### Spark
The Spark connector supports **Spark 3.5, 4.0, and 4.1** via a shared JNI FFI bridge. It intercepts row-level operations (like `MERGE INTO`) to take advantage of BenoStreamDB's fast indexing and supports configuring GPU backends.

```scala
// Read
val df = spark.read
  .format("benostream")
  .option("path", "s3://bucket/table")
  // Optionally configure the GPU device (cuda, mps, intel, rocm, auto, or cpu)
  .option("benostream.gpu_device", "cuda")
  .load()

// Write
df.write
  .format("benostream")
  .option("path", "s3://bucket/table")
  .save()
```

You can also globally configure the GPU for Spark stored procedures (e.g. index building):
```scala
spark.conf.set("spark.benostream.gpu.device", "cuda")
```

### Trino
The Trino connector is **read/write**. Reads push scalar and vector filtering down to the BenoStreamDB core via JNI, drastically reducing IO. Writes go through a `ConnectorPageSink` (Trino `Page` → Arrow → native append), `CREATE TABLE AS SELECT` creates the table then writes, and `MERGE INTO` is handled by Trino's MERGE SPI (`beginMerge`/`finishMerge` + a `ConnectorMergeSink`).

Configure the warehouse root and GPU backend in the catalog properties:

```properties
connector.name=benostreamdb
benostream.warehouse=s3://my-bucket/warehouse
benostream.gpu-device=cuda
```

Tables resolve to `{warehouse}/{schema}/{table}` (default warehouse `s3://default`).

```sql
-- Read (scalar/vector filter pushdown via JNI)
SELECT * FROM benostream.default.my_table
WHERE id > 100;

-- Write
INSERT INTO benostream.default.my_table VALUES (1, 'a');

-- CTAS
CREATE TABLE benostream.default.new_table AS
SELECT * FROM benostream.default.my_table;

-- MERGE (row-level, via the connector's merge sink)
MERGE INTO benostream.default.my_table t
USING benostream.default.staging s ON t.id = s.id
WHEN MATCHED THEN UPDATE SET name = s.name
WHEN NOT MATCHED THEN INSERT (id, name) VALUES (s.id, s.name);
```

You can configure the GPU backend for Trino globally or per-catalog using the properties file (e.g. `etc/catalog/benostream.properties`):
```properties
connector.name=benostreamdb
benostream.gpu-device=cuda
```

### Arrow Flight SQL Gateway
BenoStreamDB provides a high-performance Arrow Flight SQL server (`benostreamdb-flight`) running over gRPC (port 50051). This enables any JDBC, ODBC, ADBC, or Arrow-native client (including BI tools and distributed query engines) to query BenoStreamDB with zero-copy Arrow serialization and native index pushdown.

```bash
cargo run -p benostreamdb-flight
```

### dbt (`dbt-benostreamdb`)
Official dbt adapter for BenoStreamDB over Arrow Flight SQL. Provides native vector search macros and custom materializations:

- **Vector Macros**: `vector_distance(...)`, `knn_search(...)`, `vector_avg(...)`, `type_vector(...)`, `type_sparsevec(...)` with pgvector-compatible operators.
- **Custom Materializations**: Table and incremental materialization with support for `append`, `delete+insert`, and partition-looping `insert_overwrite`.
- **DDL Support**: Iceberg-compatible `PARTITIONED BY` syntax.

```bash
cd dbt-benostreamdb
pip install -e .
```

### Python (Direct)
```python
# No Spark needed for local/notebook work
import benostreamdb as bsdb
df = bsdb.Table("s3://bucket/table").query().execute()
# Or using traditional API: df = bsdb.Table("s3://bucket/table").to_pandas()
```

## 🔨 Building Connectors

The Spark and Trino connectors require building shaded "fat" JARs that bundle the native Rust core.

### Matrix Build
We provide a script to build a full matrix of connectors (Java 17/21, Spark 3.5/4.0):
```bash
./build-connectors.sh
```

### Hardware Acceleration
- **Standard**: Build with CPU + Intel Graphics/XPU support (default).
- **CUDA**: Build for NVIDIA GPUs:
  ```bash
  ./build-connectors.sh --cuda
  ```

### Portable Toolchain
The build script automatically downloads a project-local Maven and JDK 21 if they are missing from your system, ensuring a consistent build environment.

### Artifacts
Final JARs and ZIPs are collected in the `connector-artifacts/` directory.

## 🧪 Development

### Build & Test

```bash
# Build Rust library
cargo build --release

# Run tests
cargo test

# Run benchmarks
cargo bench

# Build Python bindings
maturin develop

# Python tests
pytest tests/
```

### Project Structure

```
benostreamdb/
├── src/
│   ├── lib.rs                  # Main library & PyO3 module registration
│   ├── core/
│   │   ├── table/              # Table API (read, write, schema, fluent query)
│   │   ├── reader/             # Index-aware Parquet reader
│   │   ├── manifest/           # Manifest management (Iceberg/Avro)
│   │   ├── index/              # HNSW, inverted, bitmap indexes
│   │   ├── catalog/            # REST, Nessie, Glue, Hive, Unity catalogs
│   │   ├── sql/                # DataFusion integration & pgvector operators
│   │   ├── planner.rs          # Query planner & optimizer
│   │   ├── iceberg/            # Iceberg V2/V3 metadata & schema
│   │   ├── lock.rs             # Vendor-neutral distributed locking (FileBasedLock via object store CAS)
│   │   ├── compaction.rs       # Compaction engine
│   │   ├── maintenance.rs      # Vacuum/GC
│   │   ├── storage.rs          # Multi-cloud storage (S3, GCS, Azure, local)
│   │   ├── wal.rs              # Write-Ahead Log
│   │   ├── ffi.rs              # JNI bindings (Spark/Trino)
│   │   └── error.rs            # Structured error types
│   ├── telemetry/              # Structured tracing (OpenTelemetry) & Prometheus metrics
│   ├── python/                 # PyO3 bindings (table, graph, session, catalogs)
│   ├── python_distance.rs      # Vector distance API
│   └── python_gpu_context.rs   # GPU device management
├── benostreamdb-flight/        # Arrow Flight SQL gRPC server
├── benostreamdb-search/        # OpenSearch 7.10 & Qdrant REST search gateway
├── dbt-benostreamdb/           # Official dbt adapter (Arrow Flight SQL)
├── benostreamdb-enterprise/    # Enterprise extensions (Continuous Indexing, Enterprise Security)
├── spark-benostreamdb/         # Spark connector (Java)
├── trino-benostreamdb/         # Trino connector (Java)
├── tests/
│   ├── integration/            # Infrastructure integration tests
│   ├── benchmarks/             # Performance benchmarks
│   └── python/                 # Python binding tests
└── benchmarks/                 # Benchmarks
```

## 🔎 Search API (OpenSearch / Elasticsearch 7.10-compatible)

BenoStreamDB ships an optional add-on, **`bsdb-search`** (`benostreamdb-search`),
that serves an **OpenSearch 1.x / Elasticsearch 7.10**-compatible REST API on top of
the engine — plus a **Qdrant**-compatible API for vector workloads. It is built for
website search, document catalogs, and knowledge bases where a 50–200 ms query latency
envelope is acceptable and object-storage-native, scale-to-zero hosting is desired.

```bash
cargo build --release -p benostreamdb-search --bin bsdb-search
BENOSEARCH_BIND=127.0.0.1 BENOSEARCH_PORT=9200 ./target/release/bsdb-search

# Index + search (ES 7.10 wire format)
curl -X POST localhost:9200/articles/_doc -H 'content-type: application/json' \
     -d '{"title":"Hello","body":"Welcome to BenoStreamDB"}'
curl -X POST localhost:9200/articles/_refresh
curl -X POST localhost:9200/articles/_search -H 'content-type: application/json' \
     -d '{"query":{"match":{"body":"BenoStreamDB"}}}'
```

**Running as a Background Service**

For production deployments on Linux and macOS, you can easily install `benostream-search` as a native background daemon (`systemd` or `launchd`) so it runs continuously and starts on boot:

```bash
# Ensure the binary is built and available at /usr/local/bin/benostream-search
sudo benostreamdb install-service
```
This will automatically generate the configuration file and start the service. See [scripts/services/README.md](scripts/services/README.md) for full configuration and uninstallation details.

**Supported:** cluster/health/cat/stats, index CRUD, mapping GET/PUT, `_doc`, `_bulk`,
`_refresh`, `_search` (`match` BM25, `knn` HNSW, hybrid RRF, `filter`/`bool` with
`term`/`terms`/`range`/`exists`, `match_all`), `_count`, `_source` filtering,
`from`/`size`, and Prometheus `/metrics`.

**Not supported (v1):** per-document delete (501, append-only), aggregations, aliases,
reindex, ILM, snapshots, auth, multi-node. See
[docs/OPENSEARCH_COMPATIBILITY.md](docs/OPENSEARCH_COMPATIBILITY.md) for the full matrix and
[docs/INSTALLATION.md](docs/INSTALLATION.md) for a complete quickstart.

## 📈 Roadmap

### ✅ Completed (Core Foundation & Scale Testing)
- [x] **Core Storage**: Hybrid segment format (Parquet + indexes) & Iceberg V2/V3 Manifest management.
- [x] **Operations**: Compaction engine, Maintenance operations, Cloud-agnostic distributed locking, & Optimistic Concurrency Control (OCC).
- [x] **Query Engine**: Native SQL support (DataFusion), Index Nested Loop Join, pgvector-compatible operators.
- [x] **Catalog**: Multi-catalog support (Nessie, REST, AWS Glue, Hive Metastore, Unity; Polaris & Lakekeeper via the REST catalog's OAuth2 client-credentials flow).
- [x] **Vector Search**: Multi-backend GPU support (CUDA, ROCm, Metal, XPU), TurboQuant™ (TQ4/TQ8), Hybrid vector + BM25 search (RRF).
- [x] **Advanced Search & Query**: Zero-Copy Arrow IPC Vector Index traversal, LRU index caching, Async Ingest Memory Buffer & WAL.
- [x] **Graph RAG & Analytics**: Native graph analytics on Iceberg edge tables (PageRank, personalized PageRank, connected components, Louvain/Leiden communities, shortest paths, and GraphRAG-style DRIFT search).
- [x] **APIs & Gateways**: OpenSearch 7.10 & Qdrant REST APIs (`benostreamdb-search`), Arrow Flight SQL Gateway (`benostreamdb-flight`).
- [x] **Connectors**: Spark (V2) & Trino (SPI) connectors, Python Vector Distance API, Official dbt adapter.
- [x] **Benchmarking & Validation**: Historical 100k / 1M doc runs vs OpenSearch and the 4 GB RAM matrix — **superseded**; the only benchmark we stand behind is the full-site Wikipedia Graph-RAG demo (see the Performance section above).
- [x] **Lifecycle Verification**: Streaming Commit & Delete Lifecycle Verification (Iceberg V2 position delete masking in vector graph scans).

### 🔄 Active & In Progress
- [ ] **Codebase Intelligence**: MCP Server Implementation, Git-Diff Incremental CI Indexer.

### 📋 Planned
- [ ] **Client Ecosystem & Packaged Distribution**: LangChain & LlamaIndex integrations.
- [ ] **Enterprise Features [Paid]**: Row-Level Security (RLS), Dynamic Column Masking, Customer-Managed Encryption Keys (CMEK), SIEM Export, Fused SIMD Kernels.

*For a detailed breakdown of all phases, see [docs/ROADMAP.md](docs/ROADMAP.md).*

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

The crate is dual-licensed under **MIT AND Apache-2.0** (see [`Cargo.toml`](Cargo.toml)).

This project contains modified source code from various upstream open-source projects (including `hnsw_rs` for pre-filtering support), which were originally licensed under Apache 2.0. BenoStreamDB maintains compliance by retaining all original copyright notices and providing prominent notice of modifications in the relevant source files.

## 🙏 Acknowledgments

- **Apache Iceberg** - Inspiration for manifest design
- **Apache Arrow** - Columnar format
- **hnsw_rs** - Vector indexing
- **RoaringBitmap** - Scalar indexing

---

**Built with ❤️ in Rust**


