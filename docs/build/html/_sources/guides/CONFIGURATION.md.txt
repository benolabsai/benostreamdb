# Configuration Guide

HyperStreamDB is designed to be highly configurable through environment variables and a centralized configuration file.

## Environment Variables

These variables control the core behavior of the system, including memory management, caching, and storage paths.

| Variable | Description | Default |
|----------|-------------|---------|
| `HYPERSTREAM_CACHE_GB` | Memory limit for the hybrid vector index (HNSW-IVF) in GB. | `2` |
| `HYPERSTREAM_BLOCK_CACHE_GB` | Memory limit for the decoded RecordBatch block cache in GB. | `4` |
| `HYPERSTREAM_DISK_CACHE_DIR` | Directory used for caching segmented index files on local disk. | `/tmp/hdb_cache` |
| `HYPERSTREAM_WAL_DIR` | Directory for the Write-Ahead Log (WAL) used for fault tolerance. | `{table_uri}/_wal` |
| `HYPERSTREAM_CONFIG` | Path to a centralized `hyperstream.toml` configuration file. | None |
| `JAEGER_ENABLED` | Enable distributed tracing via Jaeger (requires `opentelemetry` feature). | `false` |

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

[cache]
memory_limit_gb = 8
disk_cache_enabled = true
disk_cache_path = "/mnt/fast-ssd/hdb_cache"

[catalog]
type = "nessie"
url = "http://nessie:19120/api/v2"
ref = "main"
```

## Storage Credentials

HyperStreamDB uses the standard `object-store` crate, which automatically picks up credentials from:
- **AWS**: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, or IAM Roles.
- **GCP**: `GOOGLE_APPLICATION_CREDENTIALS` (JSON key file path).
- **Azure**: `AZURE_STORAGE_ACCOUNT`, `AZURE_STORAGE_KEY`.

## Query Configuration (QueryConfig)

Query-level options can be set via the `QueryConfig` struct (Rust) or passed as keyword arguments in Python.

| Option | Description | Default |
|--------|-------------|---------|
| `query_timeout_secs` | Maximum duration before a query is cancelled. Set to `0` for no timeout. | `0` (no timeout) |
| `max_result_rows` | Hard cap on rows returned. Prevents unbounded result sets from exhausting memory. | `10,000,000` |
| `max_concurrency` | Maximum parallel segment readers. Capped at **64** to prevent oversubscription on high-core-count machines. | Auto-detected (`available_parallelism()`, max 64) |

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

> **Note:** `max_concurrency` is automatically derived from `std::thread::available_parallelism()` and capped at 64. It is not intended to be manually overridden in most workloads.
