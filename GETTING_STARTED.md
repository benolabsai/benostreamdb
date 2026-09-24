# Getting Started

This guide covers two ways to use BenoStreamDB:

1. **The `bsdb-search` REST server** — an OpenSearch / Elasticsearch 7.10-compatible
   API (plus a Qdrant-compatible API) served on top of the BenoStreamDB engine.
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
- [ELASTICSEARCH_INTEGRATION_PLAN.md](ELASTICSEARCH_INTEGRATION_PLAN.md) — design and positioning.
- [README.md](README.md) — core engine features, Iceberg compliance, and query engines.
