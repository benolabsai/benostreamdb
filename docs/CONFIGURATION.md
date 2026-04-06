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
