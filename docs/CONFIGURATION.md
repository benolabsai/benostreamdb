# Configuration Guide

BenoStreamDB is designed to be highly configurable through environment variables, runtime options, and a centralized configuration file.

## Environment Variables

### Core Engine & Memory Management

| Variable | Description | Default |
|:---|:---|:---|
| `BENOSTREAM_CACHE_GB` | Memory limit for vector (HNSW-IVF), inverted, and byte caches in GB. | `1` |
| `BENOSTREAM_BLOCK_CACHE_GB` | Memory limit for decoded RecordBatch Hot Row Cache in GB. | `1` |
| `BENOSTREAM_DISK_CACHE_DIR` | Local disk directory used for caching remote index files. | `/tmp/hdb_cache` |
| `BENOSTREAM_CONFIG` | Path to a centralized `benostream.toml` configuration file. | None |
| `BENOSTREAM_MAX_CONCURRENCY` | Maximum concurrent segment reader threads per query. | Auto (min(cores, 64)) |
| `RAYON_NUM_THREADS` | Global Rayon worker pool thread count for parallel index construction. | Auto (50% of CPU cores) |

#### Concurrency Tuning: `BENOSTREAM_MAX_CONCURRENCY` vs `RAYON_NUM_THREADS`

`BENOSTREAM_MAX_CONCURRENCY` and `RAYON_NUM_THREADS` operate on different execution layers and should generally **not** be set to the same value:

* **`RAYON_NUM_THREADS` (CPU-Bound Data Parallelism)**:
  * Governs Rayon's worker threadpool for compute-intensive tasks: AVX-512/NEON SIMD vector distance calculations, HNSW graph construction and deserialization, and Product Quantization (PQ) training.
  * **Rule**: Keep $\le$ physical CPU cores (typically `cores / 2` or `cores - 1`). Setting it higher causes OS thread context switching and CPU cache thrashing.
* **`BENOSTREAM_MAX_CONCURRENCY` (I/O-Bound Async Stream Concurrency)**:
  * Governs Tokio async task concurrency for opening and streaming Parquet segments from local disk or remote object storage (`s3://`, `gs://`).
  * Non-blocking I/O tasks spend most of their time awaiting network packets or disk blocks without consuming CPU.
  * **Rule**: Can be $2\times$ to $4\times$ CPU cores (e.g. 8–32) for fast parallel segment fetching, bounded only by RAM memory buffer headroom.

##### Sizing & Tuning Matrix

| Deployment Profile | Machine Specs | `RAYON_NUM_THREADS` | `BENOSTREAM_MAX_CONCURRENCY` | Notes |
|:---|:---|:---|:---|:---|
| **Benchmark / Container** | 4 Cores, 4 GB RAM | `2` – `4` | `4` – `8` | Stable profile used in 100k & 1M OpenSearch benchmark. |
| **Production Server** | 16 Cores, 32 GB RAM | `8` – `14` | `16` – `32` | High throughput for mixed scalar + vector queries. |
| **Cloud Object Store (S3/GCS)** | 8 Cores, 32 GB RAM | `6` | `32` – `48` | High I/O concurrency hides cloud object storage latency. |

### Vector Indexing

| Variable | Description | Default |
|:---|:---|:---|
| `BENOSTREAM_HNSW_CHUNK_SIZE` | Chunk size (vector count) for building multi-chunk HNSW graph sidecars. | `100,000` |

### Write-Ahead Log (WAL) & Ingestion

| Variable | Description | Default |
|:---|:---|:---|
| `BENOSTREAM_WAL_DIR` | Directory for the Write-Ahead Log (WAL) used for streaming recovery. | `{table_uri}/_wal` |
| `BENOSTREAM_WAL_DURABILITY` | WAL flush durability policy (`always`, `adaptive`, `periodic`). | `adaptive` |
| `BENOSTREAM_WAL_COMPACT_MB` | File size threshold in MB before triggering WAL log segment compaction. | `1024` (1 GB) |
| `BENOSTREAM_WAL_SYNC_BATCH_SIZE`| Appended operations batch size before triggering a WAL sync flush. | `10` |
| `BENOSTREAM_WAL_SYNC_INTERVAL_MS`| Maximum elapsed milliseconds between background WAL sync flushes. | `100` |
| `BENOSTREAM_STREAMING_FLUSH_INTERVAL_SECS`| Background timer interval in seconds to automatically flush write buffers to Iceberg snapshots. | None |

### Search Gateway (`benostreamdb-search`)

| Variable | Description | Default |
|:---|:---|:---|
| `BENOSEARCH_PORT` | Port for the OpenSearch/Elasticsearch 7.10 compatible HTTP REST API. | `9200` |
| `BENOSEARCH_BIND` | Bind IP address for the OpenSearch REST server. | `0.0.0.0` |
| `BENOSEARCH_DEVICE` | Compute device target (`"auto"`, `"cpu"`, `"cuda"`, `"rocm"`, `"metal"`). | `"auto"` |
| `BENOSEARCH_STORAGE_URI` | Storage URI for search indices (e.g. `s3://bucket/search` or `file:///data`). | `./data` |
| `BENOSEARCH_INDEX_CACHE_GB` | Size cap in GB for on-demand sidecar index file LRU cache. | `1` |
| `BENOSEARCH_AUTO_REFRESH_SECS` | Background timer interval in seconds to flush memtables to Iceberg snapshots. | `1.0` |
| `BENOSEARCH_WAL_DURABILITY` | Gateway ingestion durability (`"async"` for batched WAL, `"sync"` for per-doc fsync). | `"async"` |
| `BENOSEARCH_RRF_K` | Smoothing constant ($k$) for Reciprocal Rank Fusion hybrid search scoring. | `60` |
| `QDRANT_BIND` | Bind IP address for the Qdrant-compatible Vector REST API listener. | Matches `BENOSEARCH_BIND` |
| `QDRANT_PORT` | Port for the Qdrant-compatible Vector REST API listener. | `6333` |

### Search Gateway Iceberg Catalog Integration

`benostreamdb-search` can automatically synchronize search indices with an external Apache Iceberg catalog (e.g. **Snowflake / Apache Polaris**, Project Nessie, AWS Glue, Hive Metastore, or Unity Catalog) so that ingested documents immediately advance the catalog's current snapshot.

| Variable | Description | Default |
|:---|:---|:---|
| `BENOSEARCH_CATALOG_TYPE` | Catalog provider: `"rest"` (Snowflake Polaris / Lakekeeper), `"nessie"`, `"glue"`, `"hive"`, `"unity"`. | None (path-based Iceberg) |
| `BENOSEARCH_CATALOG_URL` | Catalog REST/Service endpoint (e.g. `https://polaris.example.com/api/catalog/v1`). | None |
| `BENOSEARCH_CATALOG_NAMESPACE` | Catalog namespace where search index tables are created and committed. | `default` |
| `BENOSEARCH_CATALOG_CREDENTIAL` | OAuth2 Client Credentials (`<client_id>:<client_secret>`) for Polaris / REST. | None |
| `BENOSEARCH_CATALOG_TOKEN` | Bearer token for Iceberg REST authentication. | None |
| `BENOSEARCH_CATALOG_PREFIX` | Warehouse or catalog prefix (e.g. Polaris warehouse name). | None |
| `BENOSEARCH_CATALOG_ID` | AWS Glue Account / Catalog ID (when using Glue). | None |

> [!TIP]
> **Snowflake Polaris Catalog Setup:**
> Set `BENOSEARCH_CATALOG_TYPE=rest`, `BENOSEARCH_CATALOG_URL=https://<account>.snowflakecomputing.com/polaris/api/catalog/v1`, and `BENOSEARCH_CATALOG_CREDENTIAL=<client_id>:<client_secret>`. Every bulk ingest via `/_bulk` or `_doc` will automatically execute an Iceberg Atomic Swap against Snowflake Polaris. For an end-to-end walkthrough connecting Snowflake to BenoStreamDB, see the [Snowflake + Polaris Integration Guide](catalog_usage.md).


### Cloud Storage & Telemetry

| Variable | Description | Default |
|:---|:---|:---|
| `AWS_ENDPOINT_URL` | Custom S3 endpoint URL (used for MinIO, LocalStack, Ceph). | AWS default |
| `JAEGER_ENABLED` | Enable distributed OpenTelemetry tracing via Jaeger / OTLP. | `false` |

---

## The benostream.toml File

You can use a TOML file to manage complex configurations, especially for catalogs and multi-cloud storage.

BenoStreamDB looks for this file in the following order:
1. Environment variable `BENOSTREAM_CONFIG`
2. `./benostream.toml` (current directory)
3. `~/.benostream/config.toml`

### Example Configuration

```toml
[storage]
type = "s3"
bucket = "my-data-lake"
region = "us-east-1"
endpoint = "http://minio:9000"

[cache]
memory_limit_gb = 8
block_cache_gb = 8
disk_cache_enabled = true
disk_cache_path = "/mnt/fast-ssd/hdb_cache"

[wal]
durability = "adaptive"

[catalog]
type = "rest"
url = "https://polaris.example.com/api/catalog/v1"
warehouse = "main_warehouse"
credential = "CLIENT_ID:CLIENT_SECRET"
scope = "PRINCIPAL_ROLE:ALL"
token_refresh_interval_secs = 300
```

---

## Storage Credentials

BenoStreamDB uses the standard `object-store` crate, which automatically picks up credentials from:
- **AWS**: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, or IAM Roles.
- **GCP**: `GOOGLE_APPLICATION_CREDENTIALS` (JSON key file path).
- **Azure**: `AZURE_STORAGE_ACCOUNT`, `AZURE_STORAGE_KEY`.

---

## Query Configuration (QueryConfig)

Query-level options can be set via the `QueryConfig` struct (Rust) or passed as keyword arguments in Python.

| Option | Description | Default |
|:---|:---|:---|
| `query_timeout_secs` | Maximum duration before a query is cancelled. Set to `0` for no timeout. | `0` (no timeout) |
| `max_result_rows` | Hard cap on rows returned. Prevents unbounded result sets from exhausting memory. | `10,000,000` |
| `max_concurrency` | Maximum parallel segment readers. Capped at **64** to prevent oversubscription. | Auto-detected (`available_parallelism()`, max 64) |

### Python Example

```python
import benostreamdb as bsdb

table = bsdb.Table("s3://bucket/my-table")

# Apply query-level limits
results = table.sql(
    "SELECT * FROM documents WHERE embedding <-> '[0.1, 0.2]'::vector",
    query_timeout_secs=30,
    max_result_rows=100_000,
)
```

### Rust Example

```rust
use benostreamdb::{Table, QueryConfig};

let table = Table::new("s3://bucket/my-table")?;
let config = QueryConfig::default()
    .query_timeout_secs(30)
    .max_result_rows(100_000);

let batches = table.query_with_config("SELECT * FROM documents", config).await?;
```


---

## Vector index configuration

_(merged from the former `CONFIGURATION.md`)_


This guide covers configuration parameters for tuning vector search performance in BenoStreamDB.

## Configuration Parameters

### HNSW Parameters

#### `hnsw.ef_search`

Controls the search beam width for HNSW index queries.

- **Type**: Integer
- **Default**: 64
- **Range**: 1 - 1000
- **Effect**: Higher values increase accuracy but reduce speed

**Usage**:
```sql
-- Set for session
SET hnsw.ef_search = 128;

-- Query with custom ef_search
SELECT id, embedding <-> '[0.1, 0.2, 0.3]'::vector AS distance
FROM documents
ORDER BY distance
LIMIT 10;
```

**Python API**:
```python
import benostreamdb as bsdb

session = bsdb.Session()
session.set_config("hnsw.ef_search", 128)

results = session.sql("""
    SELECT id, embedding <-> '[0.1, 0.2, 0.3]'::vector AS distance
    FROM documents
    ORDER BY distance
    LIMIT 10
""")
```

**Performance Impact**:
- `ef_search = 32`: Fast, ~90% recall
- `ef_search = 64`: Balanced, ~95% recall (default)
- `ef_search = 128`: Accurate, ~98% recall
- `ef_search = 256`: Very accurate, ~99.5% recall

### IVF Parameters

#### `ivf.probes`

Controls the number of IVF clusters to search.

- **Type**: Integer
- **Default**: 10
- **Range**: 1 - 1000
- **Effect**: Higher values increase accuracy but reduce speed

**Usage**:
```sql
-- Set for session
SET ivf.probes = 20;

-- Query with custom probes
SELECT id, embedding <-> '[0.1, 0.2, 0.3]'::vector AS distance
FROM documents
ORDER BY distance
LIMIT 10;
```

**Python API**:
```python
session.set_config("ivf.probes", 20)
```

**Performance Impact**:
- `probes = 5`: Fast, ~85% recall
- `probes = 10`: Balanced, ~92% recall (default)
- `probes = 20`: Accurate, ~97% recall
- `probes = 50`: Very accurate, ~99% recall

### Index Control

#### `vector.use_index`

Controls whether to use vector indexes or force sequential scan.

- **Type**: Boolean
- **Default**: true
- **Effect**: When false, forces sequential scan (useful for testing)

**Usage**:
```sql
-- Force sequential scan (for testing)
SET vector.use_index = false;

-- Re-enable index usage
SET vector.use_index = true;
```

**Python API**:
```python
# Disable index for testing
session.set_config("vector.use_index", False)

# Re-enable
session.set_config("vector.use_index", True)
```

## Tuning Guidelines

### Accuracy vs Speed Tradeoff

| Use Case | ef_search | probes | Expected Recall | Relative Speed |
|----------|-----------|--------|-----------------|----------------|
| Real-time search | 32 | 5 | ~88% | 4x faster |
| Balanced | 64 | 10 | ~94% | 1x (baseline) |
| High accuracy | 128 | 20 | ~98% | 0.5x slower |
| Maximum accuracy | 256 | 50 | ~99.5% | 0.2x slower |

### Dataset Size Recommendations

**Small datasets (< 100K vectors)**:
```sql
SET hnsw.ef_search = 128;
SET ivf.probes = 20;
```

**Medium datasets (100K - 10M vectors)**:
```sql
SET hnsw.ef_search = 64;  -- Default
SET ivf.probes = 10;      -- Default
```

**Large datasets (> 10M vectors)**:
```sql
SET hnsw.ef_search = 32;
SET ivf.probes = 5;
```

### Dimensionality Recommendations

**Low dimensions (< 128)**:
```sql
SET hnsw.ef_search = 64;
SET ivf.probes = 10;
```

**Medium dimensions (128 - 768)**:
```sql
SET hnsw.ef_search = 64;
SET ivf.probes = 10;
```

**High dimensions (> 768)**:
```sql
SET hnsw.ef_search = 128;
SET ivf.probes = 20;
```

## Benchmarking

### Measuring Recall

```python
import benostreamdb as bsdb
import numpy as np

session = bsdb.Session()

# Ground truth (sequential scan)
session.set_config("vector.use_index", False)
ground_truth = session.sql("""
    SELECT id FROM documents
    ORDER BY embedding <-> '[0.1, 0.2, 0.3]'::vector
    LIMIT 100
""")
ground_truth_ids = set(ground_truth['id'])

# Test with index
session.set_config("vector.use_index", True)
session.set_config("hnsw.ef_search", 64)

results = session.sql("""
    SELECT id FROM documents
    ORDER BY embedding <-> '[0.1, 0.2, 0.3]'::vector
    LIMIT 100
""")
result_ids = set(results['id'])

# Calculate recall
recall = len(ground_truth_ids & result_ids) / len(ground_truth_ids)
print(f"Recall@100: {recall:.2%}")
```

### Measuring Latency

```python
import time

session.set_config("hnsw.ef_search", 64)

# Warmup
for _ in range(10):
    session.sql("SELECT id FROM documents ORDER BY embedding <-> '[0.1, 0.2, 0.3]'::vector LIMIT 10")

# Measure
latencies = []
for _ in range(100):
    start = time.time()
    session.sql("SELECT id FROM documents ORDER BY embedding <-> '[0.1, 0.2, 0.3]'::vector LIMIT 10")
    latencies.append(time.time() - start)

print(f"P50: {np.percentile(latencies, 50)*1000:.2f}ms")
print(f"P95: {np.percentile(latencies, 95)*1000:.2f}ms")
print(f"P99: {np.percentile(latencies, 99)*1000:.2f}ms")
```

## Advanced Configuration

### Per-Query Configuration (Future)

Future versions will support per-query hints:

```sql
-- Planned syntax (not yet implemented)
SELECT /*+ INDEX_HINT(ef_search=128, probes=20) */ 
    id, embedding <-> '[0.1, 0.2, 0.3]'::vector AS distance
FROM documents
ORDER BY distance
LIMIT 10;
```

Index build parameters are set during index creation using the fluent `add_index` method:

```python
import benostreamdb as bsdb

table = bsdb.Table("s3://bucket/my-table")

# TurboQuant 8-bit quantization (Recommended Default)
# 4x compression, near-lossless accuracy
table.add_index("embedding", "hnsw_tq8")

# TurboQuant 4-bit quantization
# 8x compression, maximum efficiency
table.add_index("embedding", "hnsw_tq4")

# Custom HNSW parameters
table.add_index(
    column="embedding",
    index_config={
        "type": "hnsw",
        "complexity": 16, # Max connections per node (formerly 'm')
        "quality": 200,    # Construction beam width (formerly 'ef_construction')
    }
)

# Product Quantization (PQ)
table.add_index(
    column="embedding",
    index_config={
        "type": "hnsw_pq",
        "compression": 32 # PQ subspaces (formerly 'subspaces')
    }
)
```

## Troubleshooting

### Slow Queries

**Symptom**: Queries taking longer than expected

**Solutions**:
1. Reduce `ef_search` or `probes` for faster (but less accurate) results
2. Check if index exists: `SHOW INDEXES FROM table_name`
3. Verify index is being used: Check query plan
4. Consider using binary vectors for 32x speedup

### Low Recall

**Symptom**: Missing relevant results

**Solutions**:
1. Increase `ef_search` or `probes`
2. Verify index quality: Rebuild if necessary
3. Check vector normalization (for cosine distance)
4. Ensure query vector has same preprocessing as indexed vectors

### Out of Memory

**Symptom**: OOM errors during queries

**Solutions**:
1. Reduce `ef_search` to lower memory usage
2. Use binary vectors for 32x memory reduction
3. Use sparse vectors for high-dimensional sparse data
4. Increase available memory

## Best Practices

1. **Start with defaults**: Use default parameters (ef_search=64, probes=10) initially
2. **Measure first**: Benchmark recall and latency before tuning
3. **Tune incrementally**: Adjust one parameter at a time
4. **Monitor production**: Track recall and latency metrics in production
5. **Document settings**: Record configuration choices for reproducibility

## Configuration Reference

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `hnsw.ef_search` | Integer | 64 | 1-1000 | HNSW search beam width |
| `ivf.probes` | Integer | 10 | 1-1000 | IVF clusters to search |
| `vector.use_index` | Boolean | true | true/false | Enable/disable index usage |

---

**Last Updated**: 2026-02-08
