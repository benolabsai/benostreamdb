# Configuration Guide

HyperStreamDB is designed to be highly configurable through environment variables, runtime options, and a centralized configuration file.

## Environment Variables

### Core Engine & Memory Management

| Variable | Description | Default |
|:---|:---|:---|
| `HYPERSTREAM_CACHE_GB` | Memory limit for vector (HNSW-IVF), inverted, and byte caches in GB. | `1` |
| `HYPERSTREAM_BLOCK_CACHE_GB` | Memory limit for decoded RecordBatch Hot Row Cache in GB. | `1` |
| `HYPERSTREAM_DISK_CACHE_DIR` | Local disk directory used for caching remote index files. | `/tmp/hdb_cache` |
| `HYPERSTREAM_CONFIG` | Path to a centralized `hyperstream.toml` configuration file. | None |
| `HYPERSTREAM_MAX_CONCURRENCY` | Maximum concurrent segment reader threads per query. | Auto (min(cores, 64)) |
| `RAYON_NUM_THREADS` | Global Rayon worker pool thread count for parallel index construction. | Auto (50% of CPU cores) |

#### Concurrency Tuning: `HYPERSTREAM_MAX_CONCURRENCY` vs `RAYON_NUM_THREADS`

`HYPERSTREAM_MAX_CONCURRENCY` and `RAYON_NUM_THREADS` operate on different execution layers and should generally **not** be set to the same value:

* **`RAYON_NUM_THREADS` (CPU-Bound Data Parallelism)**:
  * Governs Rayon's worker threadpool for compute-intensive tasks: AVX-512/NEON SIMD vector distance calculations, HNSW graph construction and deserialization, and Product Quantization (PQ) training.
  * **Rule**: Keep $\le$ physical CPU cores (typically `cores / 2` or `cores - 1`). Setting it higher causes OS thread context switching and CPU cache thrashing.
* **`HYPERSTREAM_MAX_CONCURRENCY` (I/O-Bound Async Stream Concurrency)**:
  * Governs Tokio async task concurrency for opening and streaming Parquet segments from local disk or remote object storage (`s3://`, `gs://`).
  * Non-blocking I/O tasks spend most of their time awaiting network packets or disk blocks without consuming CPU.
  * **Rule**: Can be $2\times$ to $4\times$ CPU cores (e.g. 8–32) for fast parallel segment fetching, bounded only by RAM memory buffer headroom.

##### Sizing & Tuning Matrix

| Deployment Profile | Machine Specs | `RAYON_NUM_THREADS` | `HYPERSTREAM_MAX_CONCURRENCY` | Notes |
|:---|:---|:---|:---|:---|
| **Benchmark / Container** | 4 Cores, 4 GB RAM | `2` – `4` | `4` – `8` | Stable profile used in 100k & 1M OpenSearch benchmark. |
| **Production Server** | 16 Cores, 32 GB RAM | `8` – `14` | `16` – `32` | High throughput for mixed scalar + vector queries. |
| **Cloud Object Store (S3/GCS)** | 8 Cores, 32 GB RAM | `6` | `32` – `48` | High I/O concurrency hides cloud object storage latency. |

### Vector Indexing

| Variable | Description | Default |
|:---|:---|:---|
| `HYPERSTREAM_HNSW_CHUNK_SIZE` | Chunk size (vector count) for building multi-chunk HNSW graph sidecars. | `100,000` |

### Write-Ahead Log (WAL) & Ingestion

| Variable | Description | Default |
|:---|:---|:---|
| `HYPERSTREAM_WAL_DIR` | Directory for the Write-Ahead Log (WAL) used for streaming recovery. | `{table_uri}/_wal` |
| `HYPERSTREAM_WAL_DURABILITY` | WAL flush durability policy (`always`, `adaptive`, `periodic`). | `adaptive` |
| `HYPERSTREAM_WAL_COMPACT_MB` | File size threshold in MB before triggering WAL log segment compaction. | `1024` (1 GB) |
| `HYPERSTREAM_WAL_SYNC_BATCH_SIZE`| Appended operations batch size before triggering a WAL sync flush. | `10` |
| `HYPERSTREAM_WAL_SYNC_INTERVAL_MS`| Maximum elapsed milliseconds between background WAL sync flushes. | `100` |
| `HYPERSTREAM_STREAMING_FLUSH_INTERVAL_SECS`| Background timer interval in seconds to automatically flush write buffers to Iceberg snapshots. | None |

### Search Gateway (`hyperstreamdb-search`)

| Variable | Description | Default |
|:---|:---|:---|
| `HYPERSEARCH_PORT` | Port for the OpenSearch/Elasticsearch 7.10 compatible HTTP REST API. | `9200` |
| `HYPERSEARCH_BIND` | Bind IP address for the OpenSearch REST server. | `0.0.0.0` |
| `HYPERSEARCH_DEVICE` | Compute device target (`"auto"`, `"cpu"`, `"cuda"`, `"rocm"`, `"metal"`). | `"auto"` |
| `HYPERSEARCH_STORAGE_URI` | Storage URI for search indices (e.g. `s3://bucket/search` or `file:///data`). | `./data` |
| `HYPERSEARCH_INDEX_CACHE_GB` | Size cap in GB for on-demand sidecar index file LRU cache. | `1` |
| `HYPERSEARCH_AUTO_REFRESH_SECS` | Background timer interval in seconds to flush memtables to Iceberg snapshots. | `1.0` |
| `HYPERSEARCH_WAL_DURABILITY` | Gateway ingestion durability (`"async"` for batched WAL, `"sync"` for per-doc fsync). | `"async"` |
| `HYPERSEARCH_RRF_K` | Smoothing constant ($k$) for Reciprocal Rank Fusion hybrid search scoring. | `60` |
| `QDRANT_BIND` | Bind IP address for the Qdrant-compatible Vector REST API listener. | Matches `HYPERSEARCH_BIND` |
| `QDRANT_PORT` | Port for the Qdrant-compatible Vector REST API listener. | `6333` |

### Search Gateway Iceberg Catalog Integration

`hyperstreamdb-search` can automatically synchronize search indices with an external Apache Iceberg catalog (e.g. **Snowflake / Apache Polaris**, Project Nessie, AWS Glue, Hive Metastore, or Unity Catalog) so that ingested documents immediately advance the catalog's current snapshot.

| Variable | Description | Default |
|:---|:---|:---|
| `HYPERSEARCH_CATALOG_TYPE` | Catalog provider: `"rest"` (Snowflake Polaris / Lakekeeper), `"nessie"`, `"glue"`, `"hive"`, `"unity"`. | None (path-based Iceberg) |
| `HYPERSEARCH_CATALOG_URL` | Catalog REST/Service endpoint (e.g. `https://polaris.example.com/api/catalog/v1`). | None |
| `HYPERSEARCH_CATALOG_NAMESPACE` | Catalog namespace where search index tables are created and committed. | `default` |
| `HYPERSEARCH_CATALOG_CREDENTIAL` | OAuth2 Client Credentials (`<client_id>:<client_secret>`) for Polaris / REST. | None |
| `HYPERSEARCH_CATALOG_TOKEN` | Bearer token for Iceberg REST authentication. | None |
| `HYPERSEARCH_CATALOG_PREFIX` | Warehouse or catalog prefix (e.g. Polaris warehouse name). | None |
| `HYPERSEARCH_CATALOG_ID` | AWS Glue Account / Catalog ID (when using Glue). | None |

> [!TIP]
> **Snowflake Polaris Catalog Setup:**
> Set `HYPERSEARCH_CATALOG_TYPE=rest`, `HYPERSEARCH_CATALOG_URL=https://<account>.snowflakecomputing.com/polaris/api/catalog/v1`, and `HYPERSEARCH_CATALOG_CREDENTIAL=<client_id>:<client_secret>`. Every bulk ingest via `/_bulk` or `_doc` will automatically execute an Iceberg Atomic Swap against Snowflake Polaris. For an end-to-end walkthrough connecting Snowflake to HyperStreamDB, see the [Snowflake + Polaris Integration Guide](SNOWFLAKE_POLARIS_GUIDE.md).


### Cloud Storage & Telemetry

| Variable | Description | Default |
|:---|:---|:---|
| `AWS_ENDPOINT_URL` | Custom S3 endpoint URL (used for MinIO, LocalStack, Ceph). | AWS default |
| `JAEGER_ENABLED` | Enable distributed OpenTelemetry tracing via Jaeger / OTLP. | `false` |

---

## The hyperstream.toml File

You can use a TOML file to manage complex configurations, especially for catalogs and multi-cloud storage.

HyperStreamDB looks for this file in the following order:
1. Environment variable `HYPERSTREAM_CONFIG`
2. `./hyperstream.toml` (current directory)
3. `~/.hyperstream/config.toml`

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

HyperStreamDB uses the standard `object-store` crate, which automatically picks up credentials from:
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
import hyperstreamdb as hdb

table = hdb.Table("s3://bucket/my-table")

# Apply query-level limits
results = table.sql(
    "SELECT * FROM documents WHERE embedding <-> '[0.1, 0.2]'::vector",
    query_timeout_secs=30,
    max_result_rows=100_000,
)
```

### Rust Example

```rust
use hyperstreamdb::{Table, QueryConfig};

let table = Table::new("s3://bucket/my-table")?;
let config = QueryConfig::default()
    .query_timeout_secs(30)
    .max_result_rows(100_000);

let batches = table.query_with_config("SELECT * FROM documents", config).await?;
```
