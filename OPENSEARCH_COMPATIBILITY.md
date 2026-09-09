# OpenSearch / Elasticsearch Compatibility

`hypersearch` (the `hyperstreamdb-search` add-on) speaks the **OpenSearch 1.x /
Elasticsearch 7.10** wire format. OpenSearch 1.x is the 7.10 fork, so the response
shapes, error envelopes, and query DSL target that dialect. `GET /` reports
`version.number = "7.10.2"` and `tagline = "You know, you search"`.

This document lists exactly what is supported, what is not, and how to work around
the gaps.

---

## Supported

### Cluster & metadata
| Endpoint | Notes |
|----------|-------|
| `GET /` | Cluster info; ES 7.10 version block + tagline. |
| `GET /_health`, `GET /_cluster/health` | Single-node health (`status: green`, 1 node, 1 primary shard per index). |
| `GET /_cluster/stats` | Aggregate index/node statistics. |
| `GET /_cat/indices` | Tab-separated index summary (`health status index pri rep docs.count docs.store store.size`). |
| `GET /metrics` | Prometheus text format (0.0.4) operational telemetry. |

### Index management
| Endpoint | Notes |
|----------|-------|
| `PUT /{index}` | Create an index, optionally with `mappings.properties`. Pre-existing index → 400 `resource_already_exists_exception`. |
| `GET /{index}` | Returns `aliases`, `mappings`, and `settings`. Missing index → 404 `index_not_found_exception`. |
| `DELETE /{index}` | **Hard delete** — removes all store objects (manifest, metadata, data, indexes). |
| `GET /{index}/_mapping` | Renders the Arrow schema as ES properties (`text`, `long`, `double`, `boolean`, `date`, `dense_vector{dims}`, nested `object`). |
| `PUT /{index}/_mapping` | Adds new properties via `Table::add_column`; optional `indexes` block registers index algorithms. |

### Documents & ingestion
| Endpoint | Notes |
|----------|-------|
| `POST /{index}/_doc` | Index a doc with a server-generated id. Auto-creates the index on first write (schema-on-write). 201 `created` / 200 `updated`. |
| `POST /{index}/_doc/{id}` | Index a doc with a client-supplied id. Duplicate id → 400 `resource_already_exists_exception`. |
| `POST /_bulk`, `POST /{index}/_bulk` | NDJSON `index` / `create` / `delete` actions. Batched per index; per-item status in the response. |
| `POST /{index}/_refresh`, `POST /_refresh` | Flush the write buffer (memtable + WAL → segments + indexes) so new docs become searchable. |

### Search
| Feature | Notes |
|---------|-------|
| `POST /{index}/_search` | Full query DSL (below). |
| `GET /{index}/_search?q=` | Lucene-style query string mapped to a multi-field `match` over string columns; honours `size` / `from`. |
| `POST /{index}/_count` | Document count, optionally filtered. |
| `match` | BM25 (Okapi) lexical search over inverted indexes. Multi-field `match` OR-merges per-field results. |
| `knn` | HNSW vector search. `k`, `num_candidates` (→ `ef_search`), `filter`. |
| Hybrid (`match` + `knn`) | Fused with Reciprocal Rank Fusion (RRF). `rrf_k` overridable per-request or via `HYPERSEARCH_RRF_K`. |
| `match_all` | Returns all docs (uniform score 1.0). |
| `filter` / `bool` | `term`, `terms`, `range`, `exists`, and `bool { must, filter, must_not }` compiled to SQL `WHERE` and evaluated with DataFusion (with index-based pruning). |
| `_source` filtering | `_source: { includes, excludes }` (dot-prefixed includes keep nested fields). |
| `from` / `size` | Pagination (default `size` 10, `from` 0). |
| `_id` | Explicit document id (reserved `_id` primary-key column). Synthesized `{segment_id}:{row_id}` fallback for pre-existing tables without an id column. |

### Error envelope
Errors use the ES shape:
```json
{ "error": { "type": "<exception>", "reason": "..." }, "status": <http_code> }
```
Mapped types include `index_not_found_exception` (404), `resource_already_exists_exception`
(400), and `illegal_argument_exception` (400).

---

## Not supported (v1)

| Feature | Behavior / workaround |
|---------|----------------------|
| Per-document delete | `DELETE /{index}/_doc/{id}` returns **501**. The store is append-only (Iceberg); soft-delete is deferred. Use `DELETE /{index}` to drop the whole index. Bulk `delete` actions return a per-item 501. |
| `delete_by_query` | Not implemented. |
| Aggregations (`aggs`) | Not implemented. Use the data-lake path (DuckDB / Trino / Spark) for analytical queries over the same Iceberg/Parquet files. |
| Index aliases | Not implemented (`aliases` is always `{}`). |
| Reindex | Not implemented. |
| ILM (index lifecycle) | Not implemented. |
| Snapshots | Not implemented. |
| Authentication / security | Not implemented. Bind to `127.0.0.1` (default) or put a reverse proxy with auth + TLS in front. |
| Multi-node / sharding / replicas | Single-node only; reports 1 primary shard, 0 replicas. |
| Kibana | No proprietary UI. See "Dashboards" below. |
| `match_phrase`, `multi_match`, `query_string`, `fuzzy`, etc. | Not implemented (return 400 `illegal_argument_exception`). |

---

## Dashboards & observability

- **Grafana Elasticsearch datasource** — point it at `http://<host>:9200` to reuse
  existing Elasticsearch dashboards for basic search queries (the wire format is
  compatible for the supported endpoints).
- **Grafana Prometheus datasource** — scrape `GET /metrics` for operational
  dashboards (query p50/p95/p99, error rates, ingestion throughput, cache hit
  rates, in-flight requests).
- **Data dashboards** — because tables are standard Iceberg/Parquet, Grafana /
  Superset / DuckDB / Trino can query the same data directly for full SQL
  analytics (including aggregations that the REST API does not expose).

---

## Positioning

HyperStreamDB-Search is **not** a drop-in replacement for an in-memory, sub-millisecond
Elasticsearch cluster. It is positioned for **website search, document catalogs, and
log archives** where a 50–200 ms query latency envelope is imperceptible to users, and
the object-storage-native, scale-to-zero operational model (on-demand index fetch from
S3/MinIO, Iceberg/Parquet data-lake format, dynamic schema evolution, GPU-accelerated
index builds) delivers a large TCO reduction versus a JVM + hot-SSD ES deployment.
