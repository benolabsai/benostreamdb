# Qdrant Compatibility

`bsdb-search` (the `benostreamdb-search` add-on) speaks the **Qdrant v1.x REST
wire format** on port `6333` (override with `BSDB_QDRANT_BIND` / `BSDB_QDRANT_PORT`).
Collections are stored as BenoStreamDB tables: a reserved `_id` (Utf8) column,
a `vector` (`FixedSizeList<Float32>`) column, and one Arrow column per inferred
payload field. Collection parameters the engine has no native home for (the
distance metric, HNSW config) are persisted in a small sidecar object
(`_qdrant_collection.json`) under the collection root.

This document lists exactly what is supported, what is not, and how the
semantics map onto the engine.

---

## Supported

### Service
| Endpoint | Notes |
|----------|-------|
| `GET /` | `{title, version, commit}` — `title` is `benostreamdb`, `version` is the crate version. |
| `GET /healthz`, `GET /livez`, `GET /readyz` | `200` + `healthz check passed` when the object store is reachable, `503` otherwise. |
| `GET /telemetry` | Basic JSON (`title`, `version`, `commit`, `collections`, `uptime_seconds`). |

### Collections
| Endpoint | Notes |
|----------|-------|
| `GET /collections` | Lists every table under the storage root. |
| `GET /collections/:name/exists` | `{exists: bool}`. |
| `GET /collections/:name` | Real `points_count` (read from the table, including the write buffer), `vectors_count`, `indexed_vectors_count`, configured dense `size` / `distance`, and `sparse_vectors` map. |
| `PUT /collections/:name` | Creates the collection eagerly with dense vectors (`_id` + `vector` columns), sparse vectors (`sparse_vectors: { "<name>": {} }`), or hybrid collections supporting both. Pre-existing collection → `400`. |
| `PATCH /collections/:name` | Persists `hnsw_config`; other params are accepted and ignored (documented below). |
| `DELETE /collections/:name` | **Hard delete** — removes all store objects and any aliases pointing at the collection. |
| `PUT /collections/:name/index` | Payload index creation. Builds physical BM25 or Bitmap index on the payload column and records in `payload_schema`. |
| `DELETE /collections/:name/index/:field_name` | Payload index deletion. Drops physical index from the underlying table and `payload_schema`. |
| `POST /collections/:name/snapshots` | Creates a point-in-time collection snapshot metadata entry. |
| `GET /collections/:name/snapshots` | Lists available snapshots for the collection. |
| `DELETE /collections/:name/snapshots/:snapshot_name` | Deletes a snapshot. |
| `PUT /collections/:name/snapshots/recover` | Restores/recovers collection snapshot metadata. |

### Points
| Endpoint | Notes |
|----------|-------|
| `PUT /collections/:name/points` | Upsert dense vectors, sparse vectors (`{"indices": [...], "values": [...]}`), or named vector maps. Implemented as *flush → delete-by-id → append* (merge-on-read via Iceberg position deletes), so re-upserting an id overwrites it. |
| `GET` / `POST /collections/:name/points` | Retrieve by `ids`. Both verbs are routed; returns dense or sparse vector wire format when `with_vector` is true. |
| `GET /collections/:name/points/:id` | Single point; `404` when missing. |
| `POST /collections/:name/points/search` | Vector search for dense (Euclid, Cosine, Dot, Manhattan) and sparse vectors (dot product scoring). |
| `POST /collections/:name/points/query` | Universal query API. Accepts dense nearest vectors, sparse nearest vectors (`indices` + `values`), or `query: {fusion: "rrf"}` across dense/sparse `prefetch: [...]` legs; omitting `query` behaves like a scroll. |
| `POST /collections/:name/points/scroll` | `{filter?, limit, offset?, with_payload?, with_vector?}` → `{points, next_page_offset}`. |
| `POST /collections/:name/points/count` | `{filter?, exact?}` → `{count}`. |
| `POST /collections/:name/points/recommend` | Supports `best_score` (max positive score minus max negative score) and `average_vector` (`mean(positive) - mean(negative)`). Sparse vectors automatically evaluate with `best_score`. |
| `POST /collections/:name/points/discover` | Metric-aware context discovery: searches with `target`, then keeps points closer to each context `positive` than its `negative` using the collection distance metric. |
| `POST /collections/:name/points/batch` | `upsert` / `delete` / `set_payload` / `overwrite_payload` / `delete_payload` / `clear_payload` / `update_vectors`. |
| `POST /collections/:name/points/payload` | Set (merge) payload. |
| `PUT /collections/:name/points/payload` | Overwrite payload. |
| `POST /collections/:name/points/payload/delete` | Delete payload keys. |
| `POST /collections/:name/points/payload/clear` | Clear payload. |
| `PUT /collections/:name/points/vectors` | Update point vectors (dense or sparse). |
| `POST /collections/:name/points/delete` | Delete by `points` (ids) and/or `filter`. |

### Aliases
| Endpoint | Notes |
|----------|-------|
| `GET /collections/aliases` | Lists aliases. |
| `POST /collections/aliases` | `create_alias` / `delete_alias` / `rename_alias` actions. |

Aliases are **process-local (in-memory)**, not persisted: they are a routing
convenience and the underlying collection data is durable. They are lost on
restart. Every collection-scoped endpoint resolves an alias to its target.

### Filters
`must` / `should` / `must_not` accept a single condition or an array. Supported
conditions:

| Condition | Notes |
|-----------|-------|
| `{key, match: {value}}` | Equality (`=`) on a payload column. |
| `{key, match: {any: [..]}}` | `IN (...)`. |
| `{key, match: {text}}` | Equality on a string column. |
| `{key, range: {gt, gte, lt, lte}}` | Numeric range. |
| `{has_id: [..]}` | `_id IN (...)`. |
| Nested `Filter` | Recursively translated. |

`must` is AND-combined, `should` is OR-combined (AND-ed with `must`), and
`must_not` is negated.

### Score semantics
The engine returns a *distance* (lower is better) for every metric; the API
converts it to the Qdrant score for the collection's configured `distance`:

| `distance` | Engine value | Qdrant score | Ordering |
|------------|--------------|--------------|----------|
| `Euclid` (default) | squared L2 | `sqrt(d)` (Euclidean distance) | ascending (lower is better) |
| `Cosine` | `1 - cos` | `1 - d` (cosine similarity) | descending (higher is better) |
| `Dot` | `-dot` | `-d` (dot product) | descending (higher is better) |
| `Manhattan` | L1 | `d` | ascending |

For sparse vectors, scoring evaluates the dot product $\sum a_i b_i$ across matching indices (higher is better, descending order).

Results are always ordered best-first.

### Error envelope
Errors use the standard Qdrant shape, and any unmapped routes return this envelope rather than crashing clients:
```text
{ "status": { "error": "..." }, "time": <seconds> }
```
Success responses use `{ "result": <T>, "status": "ok", "time": <seconds> }`.
`time` is the real elapsed request time.

---

## Not supported / approximations

| Feature | Behavior / workaround |
|---------|----------------------|
| Named vectors | A point's `vector` may be a list or a named map; the entry keyed `vector` (or the first entry) is stored in the single `vector` column. Multiple named vectors per point are not stored separately. |
| Sparse vectors | Fully supported: native wire format `{ "indices": [...], "values": [...] }`, dot-product scoring, sparse collection creation (`sparse_vectors: { "<name>": {} }`), hybrid prefetch with RRF fusion, and retrieval. |
| Binary / multi-vector (ColBERT) | Not supported yet; dense `Float32` and sparse vectors are supported. |
| `PATCH` optimizers / quantization / on-disk params | Accepted and ignored (the engine manages its own storage layout). |
| Payload index semantics | Fully supported: `PUT .../index` builds physical BM25 (text/keyword) or Bitmap indexes and tracks them in `payload_schema`. |
| `recommend` strategies | Both `best_score` and `average_vector` strategies are supported. |
| `discover` | Metric-aware context discovery evaluated over positive/negative pairs with the collection distance metric. |
| `query` with `prefetch` / fusion | Universal query supports vector prefetch queries and Reciprocal Rank Fusion (`fusion: "rrf"`). |
| `score_threshold` | Applied after scoring (inclusive). |
| `with_payload` include/exclude | Supported for top-level fields and dot-notation paths (`user.name`). |
| `with_vector` | Supported as a bool; named-vector selection returns the single stored vector. |
| `wait` / `ordering` | Accepted and ignored (writes are synchronous). |
| Snapshots | Collection snapshots supported via `POST`, `GET`, `DELETE`, and `PUT .../recover`. |
| Sharding, replication, distributed mode | Single-node only. |
| Authentication / TLS | Not implemented. Bind to `127.0.0.1` (default) or front with a reverse proxy. |
| gRPC API (port 6334) | Not implemented; REST only. |

---

## Performance notes

- **Use `BSDB_SEARCH_DEVICE=cpu` for the search server unless you have a
  validated GPU backend.** With the default `auto` device, index builds on
  every commit can be dramatically slower on an unaccelerated/software GPU
  backend; the CPU path is consistently fast for the small collections typical
  of Qdrant workloads.
- Qdrant collections index **only the `vector` column** (no BM25 inverted index
  over payload columns), which keeps point writes cheap. Payload filtering
  falls back to a scan.
- Upserts flush the write buffer before deleting the previous rows, so each
  upsert is a commit. Batch many points into one `PUT /points` (or one
  `batch` operation) rather than issuing one request per point.

---

## Positioning

BenoStreamDB-Search is **not** a drop-in replacement for an in-memory,
sub-millisecond Qdrant cluster. It is positioned for **website search, document
catalogs, and knowledge bases** where a 50–200 ms query envelope is
imperceptible, and the object-storage-native, scale-to-zero operational model
(Iceberg/Parquet data, on-demand index fetch, dynamic schema evolution) delivers
a large TCO reduction versus an always-on vector cluster. Because collections
are standard Iceberg tables, the same data is queryable via the OpenSearch API,
SQL (DuckDB / Trino / Spark), and the Python client without duplication.
