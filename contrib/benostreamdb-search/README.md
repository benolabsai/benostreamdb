# BenoStreamDB Search (`benostreamdb-search`)

OpenSearch / Elasticsearch 7.10 and Qdrant REST compatibility gateway for BenoStreamDB.

> **⚠️ Security:** This gateway has **no authentication, no authorization, and no TLS**. Run it on a **trusted internal network**, bound to `127.0.0.1` or an internal interface, behind a gateway/reverse proxy that terminates TLS and enforces authentication. Do **not** expose it directly to the public internet. See [SECURITY.md](../../SECURITY.md).

## Overview

`benostreamdb-search` enables drop-in integration with applications, vector libraries, and ingest pipelines that speak:
1. **OpenSearch / Elasticsearch 7.10 REST API** (`/_bulk`, `/{index}/_doc`, `/{index}/_search`, aggregations, index templates).
2. **Qdrant REST API** (`/collections`, `/collections/{name}/points`, search, scroll).

It runs as a standalone daemon (`bsdb-search`) backed directly by BenoStreamDB tables on local or cloud object storage.

## Installation

### Via Cargo
```bash
cargo install benostreamdb-search
```

### Via Python / PyPI
```bash
pip install benostreamdb-search
```

## Running the Server

```bash
# Start server listening on port 9200 (OpenSearch) and 6333 (Qdrant)
bsdb-search --bind 0.0.0.0 --port 9200 --storage-uri file:///data/benostreamdb
```

## Configuration

All configuration is via environment variables (or the equivalent CLI flags).
These variables are specific to this gateway; the core engine's variables are
documented in [`docs/CONFIGURATION.md`](../../docs/CONFIGURATION.md).

### Server & storage

| Variable | Description | Default |
|:---|:---|:---|
| `BSDB_SEARCH_PORT` | Port for the OpenSearch/Elasticsearch 7.10 compatible HTTP REST API. | `9200` |
| `BSDB_SEARCH_BIND` | Bind IP address for the OpenSearch REST server. | `0.0.0.0` |
| `BSDB_SEARCH_DEVICE` | Compute device target (`"auto"`, `"cpu"`, `"cuda"`, `"rocm"`, `"metal"`). | `"auto"` |
| `BSDB_SEARCH_STORAGE_URI` | Storage URI for search indices (e.g. `s3://bucket/search` or `file:///data`). | `./data` |
| `BSDB_SEARCH_INDEX_CACHE_GB` | Size cap in GB for on-demand sidecar index file LRU cache. | `1` |
| `BSDB_SEARCH_PRELOAD` | Preload index sidecars into memory on open (`0`/`false`/`no`/`off` disables). | `true` |
| `BSDB_SEARCH_PRELOAD_GB` | In-memory budget in GB for preloaded index sidecars; overflow spills to the mmap disk cache. | `4` |
| `BSDB_SEARCH_AUTO_REFRESH_SECS` | Background timer interval in seconds to flush memtables to Iceberg snapshots. | `1.0` |
| `BSDB_SEARCH_WAL_DURABILITY` | Gateway ingestion durability (`"async"` for batched WAL, `"sync"` for per-doc fsync). | `"async"` |
| `BSDB_SEARCH_RRF_K` | Smoothing constant ($k$) for Reciprocal Rank Fusion hybrid search scoring. | `60` |
| `BSDB_QDRANT_BIND` | Bind IP address for the Qdrant-compatible Vector REST API listener. | Matches `BSDB_SEARCH_BIND` |
| `BSDB_QDRANT_PORT` | Port for the Qdrant-compatible Vector REST API listener. | `6333` |

### Iceberg catalog integration

`benostreamdb-search` can automatically synchronize search indices with an
external Apache Iceberg catalog (Snowflake / Apache Polaris, Project Nessie, AWS
Glue, Hive Metastore, or Unity Catalog) so ingested documents immediately advance
the catalog's current snapshot.

| Variable | Description | Default |
|:---|:---|:---|
| `BSDB_SEARCH_CATALOG_TYPE` | Catalog provider: `"rest"` (Snowflake Polaris / Lakekeeper), `"nessie"`, `"glue"`, `"hive"`, `"unity"`. | None (path-based Iceberg) |
| `BSDB_SEARCH_CATALOG_URL` | Catalog REST/Service endpoint (e.g. `https://polaris.example.com/api/catalog/v1`). | None |
| `BSDB_SEARCH_CATALOG_URI` | Alias for `BSDB_SEARCH_CATALOG_URL` (also accepts `BSDB_CATALOG_URI`). | None |
| `BSDB_SEARCH_CATALOG_NAMESPACE` | Catalog namespace where search index tables are created and committed. | `default` |
| `BSDB_SEARCH_CATALOG_CREDENTIAL` | OAuth2 Client Credentials (`<client_id>:<client_secret>`) for Polaris / REST. | None |
| `BSDB_SEARCH_CATALOG_TOKEN` | Bearer token for Iceberg REST authentication. | None |
| `BSDB_SEARCH_CATALOG_PREFIX` | Warehouse or catalog prefix (e.g. Polaris warehouse name). | None |
| `BSDB_SEARCH_CATALOG_ID` | AWS Glue Account / Catalog ID (when using Glue). | None |

> **Note:** The gateway also accepts the core-engine aliases
> `BSDB_CATALOG_TYPE`, `BSDB_CATALOG_URL` / `BSDB_CATALOG_URI`,
> `BSDB_CATALOG_TOKEN`, `BSDB_CATALOG_CREDENTIAL`, and
> `BSDB_CATALOG_NAMESPACE` as fallbacks when the corresponding
> `BSDB_SEARCH_CATALOG_*` variable is unset.

> **Snowflake Polaris setup:** Set `BSDB_SEARCH_CATALOG_TYPE=rest`,
> `BSDB_SEARCH_CATALOG_URL=https://<account>.snowflakecomputing.com/polaris/api/catalog/v1`,
> and `BSDB_SEARCH_CATALOG_CREDENTIAL=<client_id>:<client_secret>`. Every bulk
> ingest via `/_bulk` or `_doc` automatically executes an Iceberg atomic swap
> against Snowflake Polaris.
