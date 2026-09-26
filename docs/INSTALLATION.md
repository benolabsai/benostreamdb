# Installation

```bash
# Install from PyPI
pip install benostreamdb

# Or install from source
git clone https://github.com/benolabsai/benostreamdb
cd benostreamdb

# Build Python bindings (CPU + AMD/Intel/Apple GPU backends)
pip install maturin
maturin develop

# Build with NVIDIA CUDA support (requires NVIDIA driver)
maturin develop --features cuda

# Windows Users
# BenoStreamDB is optimized for Linux/POSIX environments.
# Windows users should use WSL2 (Windows Subsystem for Linux).
```

## GPU Acceleration (Optional)

For GPU-accelerated vector operations, install the appropriate backend:

**GPU Support (NVIDIA/AMD/Intel/Apple):**
Hardware acceleration is now part of the base package.
```bash
pip install benostreamdb
# Verify: nvidia-smi / rocm-smi / clinfo
```

**AMD ROCm:**
```bash
# Ubuntu
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_5.7.50700-1_all.deb
sudo apt-get install ./amdgpu-install_5.7.50700-1_all.deb
sudo amdgpu-install --usecase=rocm
# Verify: rocm-smi
```

**Apple Metal:**
- Included with macOS 12.3+ on Apple Silicon (M1, M2, M3, M4, M5)
- No additional installation required

**Intel XPU / Graphics:**
Intel Arc, Data Center GPUs, and Iris Xe graphics are supported natively on Linux via WGPU.
```bash
# Verify Vulkan/WGPU support
vulkaninfo | grep vendor
```

See [Python Vector API Documentation](docs/PYTHON_VECTOR_API.md) for detailed GPU setup instructions.

## pgvector SQL Compatibility

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
-- <~>  Hamming
-- <%>  Jaccard
```

See [pgvector SQL Guide](docs/PGVECTOR_SQL_GUIDE.md) for complete documentation.

## Basic Usage

```python
import benostreamdb as bsdb

# Create table
table = bsdb.Table("s3://bucket/my-table")

# Write data (Pandas/PyArrow)
import pandas as pd
df = pd.DataFrame({
    "id": [1, 2, 3],
    "text": ["hello", "world", "test"],
    "embedding": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
})
table.write_pandas(df)

# Query with filters (uses indexes!) - Fluent API
results = table.query().filter("id > 1").execute()

# Vector search - Fluent API
query_vec = [0.15, 0.25]
results = table.query().vector_search(query_vec, column="embedding", k=10).execute()

# Hybrid query (scalar + vector) - Fluent API
results = (table.query()
                .filter("category = 'science'")
                .vector_search(query_vec, column="embedding", k=10)
                .execute())

# Alternative: Traditional API still supported
results = table.to_pandas(
    filter="category = 'science'",
    vector_filter={"embedding": query_vec, "k": 10}
)
```

# 🔄 Fluent Query API

BenoStreamDB features a modern fluent query API that supports method chaining for both Python and Rust:

### Python Fluent API

```python
import benostreamdb as bsdb

table = bsdb.Table("s3://bucket/my-table")

# Method chaining with filters
results = (table.query()
                .filter("age > 25")
                .filter("status = 'active'")  # Automatically combines with AND
                .execute())

# Vector search with fluent API
query_embedding = [0.1, 0.2, 0.3, 0.4]
results = (table.query()
                .vector_search(query_embedding, column="embedding", k=10)
                .execute())

# Combine scalar filtering with vector search
results = (table.query()
                .filter("category = 'documents'")
                .vector_search(query_embedding, column="content_vec", k=5)
                .select(['title', 'score'])
                .execute())

# Complex hybrid queries
results = (table.query()
                .filter("published_date > '2024-01-01'")
                .filter("author IN ('smith', 'jones')")
                .vector_search(query_embedding, column="embedding", k=20)
                .select(['title', 'author', 'score'])
                .execute())
```

### Rust Fluent API

The same fluent interface is available in native Rust:

```rust
use benostreamdb::{Table, VectorValue};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let table = Table::new("s3://bucket/my-table")?;
    
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

### Python Vector Distance API with GPU Acceleration

BenoStreamDB provides a comprehensive Python API for vector distance computations with GPU acceleration:

```python
import benostreamdb as bsdb
import numpy as np

# GPU-accelerated batch distance computation
ctx = bsdb.GPUContext.auto_detect()  # Auto-detect CUDA/ROCm/Metal/XPU
print(f"Using GPU backend: {ctx.backend}")

# Create query and database vectors
query = np.random.randn(768).astype(np.float32)
database = np.random.randn(100000, 768).astype(np.float32)

# Compute distances on GPU (10x+ faster for large databases)
distances = bsdb.l2_distance_batch(query, database, context=ctx)

# Find top-k nearest neighbors
k = 10
top_k_indices = np.argsort(distances)[:k]

# Single-pair distance computation
vec1 = np.array([1.0, 2.0, 3.0])
vec2 = np.array([4.0, 5.0, 6.0])
distance = bsdb.cosine_distance(vec1, vec2)

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
distance = bsdb.l2_distance_sparse(sparse1, sparse2)

# Binary vector operations (bit-packed for efficiency)
binary1 = np.packbits(np.random.randint(0, 2, 128))
binary2 = np.packbits(np.random.randint(0, 2, 128))
distance = bsdb.hamming_distance_packed(binary1, binary2)
```

**Supported GPU Backends:**
- **CUDA** - NVIDIA GPUs (Linux, Windows via WSL2)
- **ROCm** - AMD GPUs (Linux)
- **Metal (MPS)** - Apple Silicon (macOS)
- **Intel XPU** - Intel Graphics (Native Linux via WGPU)
- **CPU** - Fallback for all platforms

**Supported Distance Metrics:**
- L2 (Euclidean), Cosine, Inner Product, L1 (Manhattan), Hamming, Jaccard

See [Python Vector API Documentation](docs/PYTHON_VECTOR_API.md) for complete API reference and GPU installation instructions

# SQL queries (full DataFusion support with pgvector syntax)
import benostreamdb as bsdb
session = bsdb.Session()
session.register("users", table)

# Optional: Enable GPU acceleration for SQL queries
ctx = bsdb.GPUContext.auto_detect()
bsdb.set_thread_gpu_context(ctx)

# Simple SQL
results = table.sql("SELECT * FROM t WHERE id > 100")

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
table.expire_snapshots(retain_last=10)
```

## 📊 Real-World Testing Plan

The Phase-1 synthetic datasets, their generators, and the associated
micro-benchmark harnesses have been removed from the repository. The maintained,
measured end-to-end workload is the full-site Wikipedia Graph RAG demo — see
[`examples/web_ui/README.md`](../examples/web_ui/README.md) and
[`docs/BENCHMARKING.md`](BENCHMARKING.md).

## Phase 2: Nessie Integration (Next)

**Catalog Strategy:**
- ✅ Use Nessie REST v2 (don't build custom catalog)
- Implement Rust client for Iceberg REST Catalog API
- Support Git-like branching for tables

**Why Nessie?**
- Iceberg-standard protocol
- Multi-table transactions
- Battle-tested (Netflix, Apple, Dremio)

## Phase 3: Production Hardening ✅ COMPLETE

- [x] Schema evolution support
- [x] Partition evolution
- [x] Cloud-agnostic distributed locking (`FileBasedLock` via object store CAS)
- [x] CLI tools (`benostream compact`, `vacuum`, REPL SQL)
- [x] Prometheus metrics & tracing
- [x] Error handling & retries

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

## 🔌 Connectors

### Spark
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
```

### Trino
```sql
SELECT * FROM benostream.default.my_table
WHERE id > 100;  -- Uses scalar index
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
│   ├── lib.rs              # Main library
│   ├── segment.rs          # Hybrid segment writer
│   ├── reader.rs           # Index-aware reader
│   ├── manifest.rs         # Manifest management
│   ├── compaction.rs       # Compaction engine
│   ├── maintenance.rs      # Vacuum/GC
│   ├── python_binding.rs   # PyO3 bindings
│   └── storage.rs          # Multi-cloud storage
├── spark-benostream/      # Spark connector (Java)
├── trino-benostream/      # Trino connector (Java)
├── tests/
│   ├── data/               # Test datasets
│   ├── integration/        # Integration tests
│   └── benchmarks/         # Performance tests
└── benches/                # Criterion benchmarks
```

## 📈 Roadmap

### ✅ Completed
- [x] Hybrid segment format (Parquet + indexes)
- [x] Manifest management (Iceberg-like)
- [x] Compaction engine
- [x] Maintenance (expire_snapshots, remove_orphan_files)
- [x] Python bindings (Pandas-compatible)
- [x] Native SQL support (DataFusion integration)
- [x] pgvector-compatible SQL operators and syntax
- [x] Index Nested Loop Join optimization
- [x] Boolean column indexing
- [x] Multi-table JOIN support
- [x] Real-world testing (NYC Taxi, Wikipedia, embeddings)
- [x] Nessie catalog integration
- [x] Iceberg V2 compliance (Sort Orders, Partition Evolution, Statistics)
- [x] Iceberg V3 features (Row Lineage, Default Values, HyperLogLog NDV)
- [x] Standard Iceberg API (`update_spec`, `replace_sort_order`, `rewrite_data_files`, `rollback_to_snapshot`)
- [x] Python Vector Distance API with GPU acceleration
- [x] Multi-backend GPU support (CUDA, ROCm, Metal, XPU)
- [x] Sparse and binary vector operations

### 🔄 In Progress
- [ ] 100k / 1M doc competitive benchmarks vs Elasticsearch 7.10
- [ ] Apache Polaris REST catalog integration (OAuth2 client credentials)

### 📋 Planned
- [ ] Trino connector sidecar index predicate pushdown
- [ ] Multi-vector search
- [ ] Universal GPU PyPI wheel and automated CUDA CI

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

The Python wrapper is licensed under the **MIT License**.
The underlying Rust engine and core database logic is licensed under the **Apache License 2.0**.

This project contains modified source code from various upstream open-source projects (including `hnsw_rs` for pre-filtering support), which were originally licensed under Apache 2.0. BenoStreamDB maintains compliance by retaining all original copyright notices and providing prominent notice of modifications in the relevant source files.

## 🙏 Acknowledgments

- **Apache Iceberg** - Inspiration for manifest design
- **Apache Arrow** - Columnar format
- **hnsw_rs** - Vector indexing
- **RoaringBitmap** - Scalar indexing



---

## Quickstart

_(merged from the former `INSTALLATION.md`)_


This guide covers two ways to use BenoStreamDB:

1. **The `bsdb-search` REST server** — an OpenSearch / Elasticsearch 7.10-compatible
   API (plus a Qdrant v1.x-compatible API; see
   [QDRANT_COMPATIBILITY.md](QDRANT_COMPATIBILITY.md)) served on top of the
   BenoStreamDB engine.
2. **The Python client** — direct, in-process access to the engine via `pyo3` bindings.

---

## 1. The `bsdb-search` REST server

`bsdb-search` is an optional add-on crate (`benostreamdb-search`) that exposes an
Elasticsearch/OpenSearch 7.10 wire-compatible REST API. It is ideal for website
search, document catalogs, and knowledge bases where a 50–200 ms query latency
envelope is acceptable and object-storage-native, scale-to-zero hosting is desired.

### Build

```bash
# From the repository root (the workspace builds both the core and the add-on):
cargo build --release -p benostreamdb-search --bin bsdb-search
```

The binary is produced at `target/release/bsdb-search`.

### Run

```bash
# Defaults: bind 127.0.0.1:9200, store indexes under file://~/.benostreamdb/search
./target/release/bsdb-search

# Or with explicit configuration:
BENOSEARCH_BIND=0.0.0.0 \
BENOSEARCH_PORT=9200 \
BENOSEARCH_STORAGE_URI=file:///data/search \
./target/release/bsdb-search
```

### Configuration (environment variables)

| Variable | Default | Purpose |
|----------|---------|---------|
| `BENOSEARCH_STORAGE_URI` | `file://~/.benostreamdb/search` | Index root. Each index `<name>` is a table at `{root}/{name}`. Supports `file://`, `s3://`, `gs://`, `az://`, `http(s)://`. |
| `BENOSEARCH_BIND` | `127.0.0.1` | OpenSearch/ES API bind address. |
| `BENOSEARCH_PORT` | `9200` | OpenSearch/ES API port. |
| `BENOSEARCH_AUTO_REFRESH_SECS` | `0` (off) | Periodically flush every index so new docs become searchable without an explicit `_refresh`. |
| `BENOSEARCH_RRF_K` | `60` | Default RRF fusion constant for hybrid (BM25 + HNSW) search. Overridable per-request with `rrf_k`. |
| `QDRANT_BIND` | `127.0.0.1` | Qdrant-compatible API bind address. |
| `QDRANT_PORT` | `6333` | Qdrant-compatible API port. |
| `BENOSTREAM_CACHE_GB` | — | (inherited) read-cache size in GB. |
| `BENOSTREAM_WAL_SYNC_INTERVAL_MS` | — | (inherited) WAL sync interval. |

> **Security note:** `bsdb-search` v1 has **no authentication** and binds to
> `127.0.0.1` by default. If you expose it beyond localhost, place it behind a
> reverse proxy with authentication (e.g. an auth-enabled gateway) and TLS.

### Smoke test

```bash
# Cluster info (reports ES 7.10.2 wire format)
curl -s localhost:9200/ | jq

# Index a document (auto-creates the index on first write)
curl -s -X POST localhost:9200/articles/_doc -H 'content-type: application/json' \
  -d '{"title":"Hello","body":"Welcome to BenoStreamDB"}' | jq

# Make it searchable
curl -s -X POST localhost:9200/articles/_refresh | jq

# Lexical (BM25) search
curl -s -X POST localhost:9200/articles/_search -H 'content-type: application/json' \
  -d '{"query":{"match":{"body":"BenoStreamDB"}}}' | jq

# Vector (HNSW) search
curl -s -X POST localhost:9200/articles/_search -H 'content-type: application/json' \
  -d '{"knn":{"field":"vec","vector":[0.1,0.2,0.3],"k":5}}' | jq

# Prometheus metrics
curl -s localhost:9200/metrics
```

### Supported endpoints (OpenSearch / ES 7.10)

| Area | Endpoints |
|------|-----------|
| Cluster | `GET /`, `GET /_health`, `GET /_cluster/health`, `GET /_cluster/stats`, `GET /_cat/indices` |
| Index CRUD | `PUT /{index}`, `GET /{index}`, `DELETE /{index}` |
| Mapping | `GET /{index}/_mapping`, `PUT /{index}/_mapping` |
| Documents | `POST /{index}/_doc[/{id}]`, `DELETE /{index}/_doc/{id}` (501 — append-only) |
| Bulk | `POST /_bulk`, `POST /{index}/_bulk` |
| Search | `POST /{index}/_search`, `GET /{index}/_search?q=`, `POST /{index}/_count` |
| Refresh | `POST /{index}/_refresh`, `POST /_refresh` |
| Metrics | `GET /metrics` (Prometheus text format) |

See [OPENSEARCH_COMPATIBILITY.md](OPENSEARCH_COMPATIBILITY.md) for the full
supported / unsupported matrix.

---

## 2. The Python client

The core engine ships `pyo3` bindings so you can use BenoStreamDB in-process
without the REST server.

### Install

```bash
# Build and install the Python bindings (requires a Rust toolchain):
pip install -e .
# or, from the python/ directory:
cd python && pip install -e .
```

### Quickstart

```python
import benostreamdb as bsdb

# Open (or create) a table on local disk or object storage.
table = bsdb.Table("file:///tmp/my_table")

# Write rows (schema-on-write; columns are inferred and evolved).
table.write([
    {"title": "alpha", "body": "quick brown fox", "vec": [0.1, 0.2]},
    {"title": "beta",  "body": "lazy dog sleeps", "vec": [0.9, 0.1]},
])

# Commit so the data is durable and indexed.
table.commit()

# Vector search.
results = table.vector_search("vec", [0.1, 0.2], k=2)

# Scalar / SQL search.
rows = table.read(filter="body LIKE '%fox%'")
```

### Vector Quantization with TurboQuant (TQ8 / TQ4)

BenoStreamDB includes **TurboQuant™** out-of-the-box in the free community core engine. TurboQuant uses Fast Walsh-Hadamard Transform (FWHT) followed by scalar quantization to deliver outlier-robust compression with high recall retention:

- **TQ8 (8-bit)**: 4x RAM and disk compression with >99% recall retention. Ideal default for production RAG and semantic search.
- **TQ4 (4-bit)**: 8x RAM and disk compression for massive datasets.

```python
import benostreamdb as bsdb

table = bsdb.Table("file:///tmp/my_rag_table")

# High-Performance Default: HNSW with TurboQuant 8-bit (4x compression)
table.add_index("embedding", "hnsw_tq8")

# Maximum Compression: HNSW with TurboQuant 4-bit (8x compression)
table.add_index("embedding", "hnsw_tq4")

# Or use the explicit quantize() API with tuning knobs
table.quantize(
    column="embedding",
    type_="TQ8",           # "TQ8", "TQ4", or "PQ"
    metric="l2",           # "l2", "cosine", or "dot"
    complexity=16,         # HNSW M connections
    quality=200            # HNSW ef_construction
)

# Search transparently leverages Asymmetric Distance Calculation (ADC)
results = table.vector_search("embedding", [0.1, 0.2], k=10)
```

### Running the test suites

```bash
# Rust unit + integration tests (workspace):
cargo test --workspace

# Python test suite:
pytest tests/

# bsdb-search REST conformance suite:
pytest benostreamdb-search/tests/test_search_api.py
```

---

## Next steps

- [OPENSEARCH_COMPATIBILITY.md](OPENSEARCH_COMPATIBILITY.md) — full API compatibility matrix.
- [OPENSEARCH_COMPATIBILITY.md](OPENSEARCH_COMPATIBILITY.md) — design and positioning.
- [README.md](README.md) — core engine features, Iceberg compliance, and query engines.
