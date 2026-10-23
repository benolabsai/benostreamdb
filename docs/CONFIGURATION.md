# Configuration Guide

BenoStreamDB is designed to be highly configurable through environment variables, runtime options, and a centralized configuration file.

## Environment Variables

### Core Engine & Memory Management

| Variable | Description | Default |
|:---|:---|:---|
| `BSDB_CACHE_GB` | Memory limit for vector (HNSW-IVF), inverted, and byte caches in GB. | `1` |
| `BSDB_BLOCK_CACHE_GB` | Memory limit for decoded RecordBatch Hot Row Cache in GB. | `1` |
| `BSDB_DISK_CACHE_DIR` | Local disk directory used for caching remote index files. | `/tmp/hdb_cache` |
| `BSDB_CACHE_DIR` | Local directory for the layered-index cache when reading external (REST-catalog) tables. | `$TMPDIR/benostream_cache` |
| `BSDB_STORAGE_URI` | Default table/storage URI for the gateway and Iceberg REST binaries. | `file:///tmp` |
| `BSDB_CONFIG` | Path to a centralized `benostream.toml` configuration file. | None |
| `BSDB_MAX_CONCURRENCY` | Maximum concurrent segment reader threads per query. | Auto (min(cores, 64)) |
| `RAYON_NUM_THREADS` | Global Rayon worker pool thread count for parallel index construction. | Auto (50% of CPU cores) |
| `BSDB_DATAFUSION_MEMORY_GB` | DataFusion query memory limit in GB; SQL sorts/joins spill to disk past it. | Half the effective memory budget |
| `BSDB_TARGET_PARTITIONS` | DataFusion target partition count for query execution. | Effective CPU budget (respects cgroup limits) |
| `BSDB_INDEX_BUILD_CONCURRENCY` | Maximum concurrent background index builds. Non-positive values are ignored. | One build per ~8 GB RAM, capped at CPU count, min 1 |
| `BSDB_MIN_FREE_DISK_GB` | Refuse a local flush that would drop free disk below this many GB. | `1.0` |

#### Concurrency Tuning: `BSDB_MAX_CONCURRENCY` vs `RAYON_NUM_THREADS`

`BSDB_MAX_CONCURRENCY` and `RAYON_NUM_THREADS` operate on different execution layers and should generally **not** be set to the same value:

* **`RAYON_NUM_THREADS` (CPU-Bound Data Parallelism)**:
  * Governs Rayon's worker threadpool for compute-intensive tasks: AVX-512/NEON SIMD vector distance calculations, HNSW graph construction and deserialization, and Product Quantization (PQ) training.
  * **Rule**: Keep $\le$ physical CPU cores (typically `cores / 2` or `cores - 1`). Setting it higher causes OS thread context switching and CPU cache thrashing.
* **`BSDB_MAX_CONCURRENCY` (I/O-Bound Async Stream Concurrency)**:
  * Governs Tokio async task concurrency for opening and streaming Parquet segments from local disk or remote object storage (`s3://`, `gs://`).
  * Non-blocking I/O tasks spend most of their time awaiting network packets or disk blocks without consuming CPU.
  * **Rule**: Can be $2\times$ to $4\times$ CPU cores (e.g. 8–32) for fast parallel segment fetching, bounded only by RAM memory buffer headroom.

##### Sizing & Tuning Matrix

| Deployment Profile | Machine Specs | `RAYON_NUM_THREADS` | `BSDB_MAX_CONCURRENCY` | Notes |
|:---|:---|:---|:---|:---|
| **Benchmark / Container** | 4 Cores, 4 GB RAM | `2` – `4` | `4` – `8` | Stable profile used in 100k & 1M OpenSearch benchmark. |
| **Production Server** | 16 Cores, 32 GB RAM | `8` – `14` | `16` – `32` | High throughput for mixed scalar + vector queries. |
| **Cloud Object Store (S3/GCS)** | 8 Cores, 32 GB RAM | `6` | `32` – `48` | High I/O concurrency hides cloud object storage latency. |

### Security

#### Authentication (search gateway + Flight SQL)

Stateless auth — no user database. Identity is delegated to a shared secret or
an external IdP's signing key. The same credentials protect the OpenSearch/ES
API, the Qdrant API, and the Flight SQL gRPC service.

| Variable | Description | Default |
|:---|:---|:---|
| `BSDB_API_KEY` | Shared secret. HTTP: `Authorization: Bearer <key>`, `Authorization: ApiKey <key>`, or `api-key: <key>` (Qdrant). Flight: `authorization: Bearer <key>`. | unset |
| `BSDB_JWT_SECRET` | HS256 shared secret for JWT verification. | unset |
| `BSDB_JWT_PUBLIC_KEY` | RS256 public key (PEM) for JWT verification — the "delegate to your IdP" path. | unset |
| `BSDB_JWT_AUDIENCE` | Required `aud` claim (if set). | unset |
| `BSDB_JWT_ISSUER` | Required `iss` claim (if set). | unset |
| `BSDB_AUTH_REQUIRED` | Fail closed (`401`/`UNAUTHENTICATED`) when no credential is configured. | `false` |
| `BSDB_METRICS_PUBLIC` | Exempt `/metrics` from auth (for an internal Prometheus scraper). | `false` |

Keep the secret in a secret manager (Vault, AWS/GCP/Azure Secrets Manager) or a
`0600` file mounted at `/run/secrets/` (Docker/K8s) or
`/etc/benostreamdb/secrets/`, loaded via systemd `LoadCredential=` /
`EnvironmentFile=` (mode `0600`). Never commit it or place it in a
world-readable env file.

#### SSRF guard

| Variable | Description | Default |
|:---|:---|:---|
| `BSDB_SSRF_GUARD` | Enable the SSRF guard. When set to `1`/`true`/`yes`, external URIs and custom object-store endpoints (`AWS_ENDPOINT_URL`, `AZURE_STORAGE_ENDPOINT`, `AZURE_ENDPOINT`, `GOOGLE_STORAGE_ENDPOINT`, `GOOGLE_ENDPOINT`, `GCS_ENDPOINT`) that resolve to loopback, private, or link-local addresses are rejected. Off by default so local catalogs (`http://localhost:8181`) keep working; enable it in production. | off |

### Embeddings

Embedding provenance is a core table property (`benostream.embedding.*`, see the main README's
"Embedding Provenance" section). These variables configure the embedding generator used by the MCP
server and the Python helpers.

| Variable | Description | Default |
|:---|:---|:---|
| `BSDB_VALIDATE_EMBEDDINGS` | `strict` (also `1`/`true`/`on`) scans every `List`/stringified vector row against the table's declared dimension. By default only `FixedSizeList` columns are checked, at schema level; model identity is never inferred from values. | off (schema checks only) |
| `BSDB_EMBEDDING_PROVIDER` | `local` (in-process Candle), `http`, or `none`. | `local` unless `BSDB_EMBEDDING_URL` is set |
| `BSDB_EMBEDDING_MODEL` | Hugging Face BERT-family model id that publishes `model.safetensors`. | `BAAI/bge-small-en-v1.5` |
| `BSDB_EMBEDDING_CACHE_DIR` | Private model cache; reused after the first download. | `<data-dir>/models` |
| `BSDB_EMBEDDING_URL` | Ollama / OpenAI-compatible / TEI embedding endpoint. | None |
| `BSDB_EMBEDDING_API_KEY` | Bearer token for the HTTP endpoint. | None |
| `BSDB_EMBEDDING_BATCH_SIZE` | Candle inference batch size. | `16` |
| `BSDB_EMBEDDING_MAX_LENGTH` | Tokenizer truncation length. | `512` |
| `BSDB_EMBEDDING_DIM` | Optional dimension override for MRL truncation. Known models supply it from the id map, and loaded models supply it from `config.json` (`hidden_size`). | Model default |
| `BSDB_EMBEDDING_QUERY_PREFIX` | Override the query instruction prefix. Empty string disables it. | Model default |
| `BSDB_EMBEDDING_DOCUMENT_PREFIX` | Override the document instruction prefix. Empty string disables it. | Model default |

The Candle loader runs on CPU in the published build. The legacy `model_path`, `device`,
`quantized`, `intra_threads`, `pooling`, and `show_download_progress` JSON keys are accepted for
backward compatibility but ignored.

### Write-Ahead Log (WAL) & Ingestion

| Variable | Description | Default |
|:---|:---|:---|
| `BSDB_WAL_DIR` | Directory for the Write-Ahead Log (WAL) used for streaming recovery. | `{table_uri}/_wal` |
| `BSDB_WAL_DURABILITY` | WAL flush durability policy (`always`, `adaptive`, `periodic`). | `adaptive` |
| `BSDB_WAL_COMPACT_MB` | File size threshold in MB before triggering WAL log segment compaction. | `1024` (1 GB) |
| `BSDB_WAL_SYNC_BATCH_SIZE`| Appended operations batch size before triggering a WAL sync flush. | `10` |
| `BSDB_WAL_SYNC_INTERVAL_MS`| Maximum elapsed milliseconds between background WAL sync flushes. | `100` |
| `BSDB_STREAMING_FLUSH_INTERVAL_SECS`| Background timer interval in seconds to automatically flush write buffers to Iceberg snapshots. | None |
| `BSDB_MAX_INGEST_RAM_GB` | Ingest RAM high-water mark in GB; writers pause above it (with a fail-open grace) until background tasks reclaim memory. | 80% of available memory |
| `BSDB_INGEST_MEMORY_BUDGET_GB` | Heap-trim budget during bulk ingest: RSS above it triggers `malloc_trim` at unit boundaries. | 80% of available memory |
| `BSDB_INGEST_BACKPRESSURE_GRACE_SECS` | Fail-open grace before a stalled writer proceeds anyway. Non-positive or unparseable values fall back to the default. | `60` |
| `BSDB_MANIFEST_MAX_FILES_BEFORE_CONSOLIDATION` | Manifest files accumulated by the append-only fast path before a full consolidation commit. | `32` |
| `BSDB_VACUUM_MIN_FILE_AGE_SECS` | Minimum age before vacuum deletes an unreferenced file. `0` disables the grace period. | `60` |
| `BSDB_SEARCH_WAL_DURABILITY` | Compatibility alias for `BSDB_WAL_DURABILITY`, read by the search gateway's table opener. | `adaptive` |

### Vector Index & GPU

| Variable | Description | Default |
|:---|:---|:---|
| `BSDB_GPU_DEVICE` | Device or comma-separated pool for vector index builds and distance kernels: `auto`, `cpu`, `cuda[:N]`, `mps`, or `cuda:0,cuda:1`. A pool assigns one device per worker thread round-robin; a single HNSW search still runs on one device. `BENOSTREAM_GPU_DEVICE` is accepted as a fallback. | unset (`auto`) |
| `BSDB_HNSW_M` | HNSW per-node neighbour count. | `16` |
| `BSDB_HNSW_EF_CONSTRUCTION` | HNSW build beam width. | `max(2 × M, 40)` |
| `BSDB_HNSW_N_LISTS` | IVF list count. | Heuristic from vector count |
| `BSDB_NVRTC_PATH` | NVRTC library file or directory used to compile GPU kernels. | Auto-discovery (CUDA home, Python site-packages, `LD_LIBRARY_PATH`) |
| `BSDB_DRIFT_MEMORY_MB` | In-memory budget for graph-algorithm UDAF accumulators before they spill to the CSR index. | Engine memory budget |

### Catalog

| Variable | Description | Default |
|:---|:---|:---|
| `BSDB_CATALOG_TYPE` | Catalog implementation: `rest`, `glue`, `hive`, `jdbc`, `nessie`, or `unity`. | Inferred from the URI scheme or `BSDB_CONFIG` |
| `BSDB_CATALOG_URI` / `BSDB_CATALOG_URL` | Catalog endpoint. | None |
| `BSDB_CATALOG_NAME` | Warehouse / catalog name to register. | None |
| `BSDB_CATALOG_NAMESPACE` | Default namespace for unqualified tables. | None |
| `BSDB_CATALOG_CREDENTIAL` | Vendor credential (`client_id:client_secret`, or the provider's equivalent). | None |
| `BSDB_CATALOG_TOKEN` | Bearer token for catalog requests. | None |

See [Catalog Usage](CATALOG_USAGE.md) and `benostream.toml` below for the full
catalog configuration surface.

### MCP Server (`benostreamdb-mcp`)

| Variable | Description | Default |
|:---|:---|:---|
| `BSDB_WAREHOUSE` | Local warehouse path exposed to the agent. | System temp dir |
| `BSDB_READWRITE_SCHEMAS` | Comma-separated or JSON array of schemas that allow writes. | `["scratch"]` |
| `BSDB_READONLY_SCHEMAS` | Comma-separated or JSON array of schemas locked read-only. | `["default", "public"]` |
| `BSDB_DEFAULT_WRITE_SCHEMA` | Schema for unqualified table creation. | `scratch` |
| `BSDB_EPHEMERAL` | Use a temporary warehouse for this session (`1`/`true`). | `false` |
| `BSDB_MCP_CONFIG` | Full JSON configuration string. | None |
| `BSDB_EMBEDDING_*` | Embedding generator settings (see [Embeddings](#embeddings)). | See above |

See [`contrib/benostreamdb-mcp/README.md`](../contrib/benostreamdb-mcp/README.md#configuration-reference)
for the full CLI/JSON mapping.

### Search Gateway (`benostreamdb-search`)

The OpenSearch/Qdrant gateway is a separate component with its own
configuration surface (`BSDB_SEARCH_*`, `BSDB_QDRANT_BIND`, `BSDB_QDRANT_PORT`).
See the
[search gateway README](../contrib/benostreamdb-search/README.md#configuration)
for the full table.

### Server binaries

The Flight SQL server (`benostreamdb-flight`) has its own variables:

| Variable | Description | Default |
|:---|:---|:---|
| `BSDB_FLIGHT_BIND` | gRPC bind address. | `127.0.0.1` |
| `BSDB_FLIGHT_PORT` | gRPC port. | `50051` |
| `BSDB_FLIGHT_MAX_ROWS` | Row cap applied to Arrow Flight results. | `100000` |
| `BSDB_FLIGHT_READ_ONLY` | Refuse DDL, DML, and ingest. Set to `1`/`true`. | off |
| `BSDB_WAREHOUSE` | Warehouse root for the server. | `./data` |
| `BSDB_CATALOG_NAME` | Catalog name to expose. | From `benostream.toml` |

See the [Flight SQL README](../server/flight_sql/README.md#configuration).


### Cloud Storage & Telemetry

| Variable | Description | Default |
|:---|:---|:---|
| `AWS_ENDPOINT_URL` | Custom S3 endpoint URL (used for RustFS, LocalStack, Ceph). | AWS default |
| `JAEGER_ENABLED` | Enable distributed OpenTelemetry tracing via Jaeger / OTLP. | `false` |
| `BSDB_METRICS_BIND` | Bind address for the embedded/Flight metrics HTTP listener. | `127.0.0.1` |
| `BSDB_METRICS_PORT` | Port for the embedded/Flight metrics HTTP listener. | `9090` |
| `BSDB_METRICS_PUBLIC` | Exempt `/metrics` from auth (for an internal Prometheus scraper). | `false` |
| `BSDB_OTLP_ENDPOINT` | OTLP/HTTP metrics endpoint (e.g. `http://localhost:4318`). The pusher appends `/v1/metrics` and is disabled when unset. | None |
| `BSDB_OTLP_HEADERS` | Comma-separated `key=value` headers for the OTLP push (for example an auth token). | None |
| `BSDB_OTLP_INTERVAL_SECS` | OTLP push period in seconds. | `15` |
| `BSDB_SERVICE_NAME` | `service.name` resource attribute on exported metrics. | `benostreamdb` |

### Development, Test & Benchmark Variables

These are not engine runtime settings. They are read by scripts, tests, or
packaging workflows in this repository.

| Variable | Used by | Description |
|:---|:---|:---|
| `BSDB_ALLOW_DEBUG` | `scripts/bench_csr_subgraph.py` | Allow the CSR benchmark to run against a debug build (`1`). |
| `BSDB_MAX_DEGREE` | `scripts/bench_csr_subgraph.py` | Degree cap for the generated subgraph. |
| `BSDB_DATA` | `scripts/prepare_demo.py`, `scripts/build_three_tier.py`, `scripts/build_demo_dataset.py` | Input directory for downloaded Wikipedia/demo dumps. |
| `BSDB_INDEX` | `benchmarks/competitors/run_bench.sh` | Index type selected by the competitor benchmark harness (`hnsw`, `hnsw_tq8`, `hnsw_pq`, ...). |
| `BSDB_TABLE_DIR` | `scripts/build_three_tier.py` | Output table directory. |
| `BSDB_THREE_TIER_OUT` | `scripts/build_three_tier.py` | Root output directory for the three-tier build. |
| `BSDB_EMBED_SHARD_ROWS` | `scripts/prepare_demo.py` | Rows per embedding shard. |
| `BSDB_LOAD_CHUNK_ROWS` | `scripts/prepare_demo.py` | Rows per load chunk. |
| `BSDB_DEMO_DB` | `examples/web_ui/` | Demo database path for the Streamlit/Dash UIs. |
| `BSDB_SKIP_GPU_PROBE` | `python/benostreamdb/__init__.py` | Internal guard that stops the GPU-probe child from recursing. |
| `BSDB_SOAK_RSS_CEILING_GB` | `tests/soak.rs` | RSS ceiling for the soak test (default 8 GiB). |
| `BSDB_TEST_S3_BUCKET` | S3 multi-writer tests | Bucket used by the concurrency tests. |
| `BSDB_WORKLOAD_SEED` | randomized workload tests | LCG seed (default `12345`). |
| `BSDB_WORKLOAD_STEPS` | randomized workload tests | Number of randomized steps (default `400` for Rust, `100` for the Flight Python test). |
| `BSDB_RELEASE_VERSION` | release workflow / dbt adapter | Forced version string for packaging. |
| `BSDB_VERSION` | `docker-compose.production.yml` | Image tag for the search/Flight containers (`latest` by default). |

See [Monitoring & Operations](monitoring.md) for the full metric catalog and the
per-deployment-mode endpoints.

---

## The benostream.toml File

You can use a TOML file to manage complex configurations, especially for catalogs and multi-cloud storage.

BenoStreamDB looks for this file in the following order:
1. Environment variable `BSDB_CONFIG`
2. `./benostream.toml` (current directory)
3. `~/.benostream/config.toml`

### Example Configuration

```toml
[storage]
type = "s3"
bucket = "my-data-lake"
region = "us-east-1"
endpoint = "http://rustfs:9000"

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

Query-level options are set via the `QueryConfig` struct (Rust). The Python binding exposes the same SQL execution surface through `Table.sql()` and `Session.sql()`.

| Option | Description | Default |
|:---|:---|:---|
| `query_timeout_secs` | Maximum duration before a query is cancelled. Set to `0` for no timeout. | `0` (no timeout) |
| `max_result_rows` | Hard cap on rows returned. Prevents unbounded result sets from exhausting memory. | `10,000,000` |
| `max_concurrency` | Maximum parallel segment readers. Capped at **64** to prevent oversubscription. | Auto-detected (`available_parallelism()`, max 64) |

### Python Example

```python
import benostreamdb as bsdb

table = bsdb.Table("s3://bucket/my-table")

# The table is registered as 't' for the duration of the query
results = table.sql(
    "SELECT * FROM t WHERE embedding <-> '[0.1, 0.2]'::vector"
)
```

> Query timeouts, result-row caps, and reader concurrency are `QueryConfig`
> fields on the Rust surface; they are not keyword arguments on the Python
> `Table.sql()` method.

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

#### `ef_search`

Controls the search beam width for HNSW index queries.

- **Type**: Integer
- **Default**: 64
- **Range**: 1 - 1000
- **Effect**: Higher values increase accuracy but reduce speed

**Usage** (optimizer hint):
```sql
SELECT /*+ INDEX_HINT(ef_search=128) */
       id, embedding <-> '[0.1, 0.2, 0.3]'::vector AS distance
FROM documents
ORDER BY distance
LIMIT 10;
```

**Python API** (per-query vector filter):
```python
import benostreamdb as bsdb

table = bsdb.Table("s3://bucket/documents")

results = table.to_pandas(
    vector_filter={
        "column": "embedding",
        "query": [0.1, 0.2, 0.3],
        "k": 10,
        "ef_search": 128,
    }
)
```

**Performance Impact**:
- `ef_search = 32`: Fast, ~90% recall
- `ef_search = 64`: Balanced, ~95% recall (default)
- `ef_search = 128`: Accurate, ~98% recall
- `ef_search = 256`: Very accurate, ~99.5% recall

### IVF Parameters

#### `probes`

Controls the number of IVF clusters to search.

- **Type**: Integer
- **Default**: 10
- **Range**: 1 - 1000
- **Effect**: Higher values increase accuracy but reduce speed

**Usage** (optimizer hint):
```sql
SELECT /*+ INDEX_HINT(probes=20) */
       id, embedding <-> '[0.1, 0.2, 0.3]'::vector AS distance
FROM documents
ORDER BY distance
LIMIT 10;
```

**Python API** (per-query vector filter):
```python
import benostreamdb as bsdb

table = bsdb.Table("s3://bucket/documents")

results = table.to_pandas(
    vector_filter={
        "column": "embedding",
        "query": [0.1, 0.2, 0.3],
        "k": 10,
        "probes": 20,
    }
)
```

**Performance Impact**:
- `probes = 5`: Fast, ~85% recall
- `probes = 10`: Balanced, ~92% recall (default)
- `probes = 20`: Accurate, ~97% recall
- `probes = 50`: Very accurate, ~99% recall

### Index Control

#### `use_index`

Controls whether to use vector indexes or force sequential scan.

- **Type**: Boolean
- **Default**: true
- **Effect**: When false, forces sequential scan (useful for testing)

**Usage** (optimizer hint):
```sql
-- Force a sequential scan (for testing)
SELECT /*+ INDEX_HINT(use_index=false) */
       id, embedding <-> '[0.1, 0.2, 0.3]'::vector AS distance
FROM documents
ORDER BY distance
LIMIT 10;
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
SELECT /*+ INDEX_HINT(ef_search=128, probes=20) */
       id, embedding <-> '[0.1, 0.2, 0.3]'::vector AS distance
FROM documents
ORDER BY distance
LIMIT 10;
```

**Medium datasets (100K - 10M vectors)**:
```sql
SELECT /*+ INDEX_HINT(ef_search=64, probes=10) */ ...   -- defaults
```

**Large datasets (> 10M vectors)**:
```sql
SELECT /*+ INDEX_HINT(ef_search=32, probes=5) */ ...
```

### Dimensionality Recommendations

**Low dimensions (< 128)**:
```sql
SELECT /*+ INDEX_HINT(ef_search=64, probes=10) */ ...
```

**Medium dimensions (128 - 768)**:
```sql
SELECT /*+ INDEX_HINT(ef_search=64, probes=10) */ ...
```

**High dimensions (> 768)**:
```sql
SELECT /*+ INDEX_HINT(ef_search=128, probes=20) */ ...
```

## Benchmarking

### Measuring Recall

```python
import benostreamdb as bsdb
import numpy as np

table = bsdb.Table("s3://bucket/documents")
query_vector = [0.1, 0.2, 0.3]

# High-recall reference run (wide beam)
reference = table.to_pandas(
    vector_filter={
        "column": "embedding",
        "query": query_vector,
        "k": 100,
        "ef_search": 512,
    }
)
reference_ids = set(reference["id"])

# Run under test (default beam)
results = table.to_pandas(
    vector_filter={
        "column": "embedding",
        "query": query_vector,
        "k": 100,
        "ef_search": 64,
    }
)
result_ids = set(results["id"])

# Calculate recall
recall = len(reference_ids & result_ids) / len(reference_ids)
print(f"Recall@100: {recall:.2%}")
```

### Measuring Latency

```python
import time
import numpy as np
import benostreamdb as bsdb

table = bsdb.Table("s3://bucket/documents")
vector_filter = {
    "column": "embedding",
    "query": [0.1, 0.2, 0.3],
    "k": 10,
    "ef_search": 64,
}

# Warmup
for _ in range(10):
    table.to_pandas(vector_filter=vector_filter)

# Measure
latencies = []
for _ in range(100):
    start = time.time()
    table.to_pandas(vector_filter=vector_filter)
    latencies.append(time.time() - start)

print(f"P50: {np.percentile(latencies, 50)*1000:.2f}ms")
print(f"P95: {np.percentile(latencies, 95)*1000:.2f}ms")
print(f"P99: {np.percentile(latencies, 99)*1000:.2f}ms")
```

## Advanced Configuration

### Per-Query Configuration

Per-query vector-search hints are supported via SQL comments:

```sql
-- Supported: INDEX_HINT(ef_search=..., probes=...)
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
    "embedding",
    {
        "type": "hnsw",
        "complexity": 16, # Max connections per node (formerly 'm')
        "quality": 200,    # Construction beam width (formerly 'ef_construction')
    }
)

# Product Quantization (PQ)
table.add_index(
    "embedding",
    {
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

| Hint key | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `ef_search` | Integer | 64 | 1-1000 | HNSW search beam width |
| `probes` | Integer | 10 | 1-1000 | IVF clusters to search |
| `use_index` | Boolean | true | true/false | Enable/disable index usage |
