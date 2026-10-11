# REST Search Gateway (`benostreamdb-search`)

`benostreamdb-search` is a multi-protocol HTTP search gateway that exposes BenoStreamDB tables over standard search engine wire formats.

---

## Overview

The search gateway runs as a standalone daemon (`bsdb-search`) and exposes two distinct REST interfaces simultaneously:
1. **OpenSearch / Elasticsearch 7.10 Wire Protocol (Port 9200)**: Drop-in support for text search, `_bulk` ingestion, `_search` queries, term aggregations, and Kibana/Grafana connectivity.
2. **Qdrant Vector REST Protocol (Port 6333)**: Compatibility for vector collections, point upserts, payload filtering, and nearest-neighbor search.

> **Compatibility Scope:** The gateway implements a targeted **compatibility subset** covering mainstream ingestion, point lookups, and search endpoints. It does not replicate entire distributed cluster management features (e.g. cross-cluster replication, scripting, or Lucene plugin architectures). Review the specific compatibility matrices below for exact supported APIs:
> - [OpenSearch / Elasticsearch 7.10 Compatibility Matrix](../OPENSEARCH_COMPATIBILITY.md)
> - [Qdrant Compatibility Matrix](../QDRANT_COMPATIBILITY.md)

---

## ⚠️ Deployment Security

* **Stateless Authentication**: Supports static API keys (`BSDB_API_KEY`) and JWT verification. Keys are accepted via `Authorization: Bearer <token>`, `Authorization: ApiKey <token>`, or the `api-key` header.
* **No TLS**: The search gateway does not provide native TLS termination. Bind to `127.0.0.1` or an internal VPC subnet behind a reverse proxy that terminates TLS.

---

## Configuration

| Variable | Description | Default |
|:---|:---|:---|
| `BSDB_SEARCH_BIND` | Bind address for OpenSearch REST server | `0.0.0.0` |
| `BSDB_SEARCH_PORT` | Port for OpenSearch / ES 7.10 REST server | `9200` |
| `BSDB_QDRANT_BIND` | Bind address for Qdrant REST server | Matches search bind |
| `BSDB_QDRANT_PORT` | Port for Qdrant REST server | `6333` |
| `BSDB_SEARCH_STORAGE_URI` | Base storage URI for indices and collections | `./data` |
| `BSDB_API_KEY` | Shared secret API key | Unset |
| `BSDB_SEARCH_WAL_DURABILITY` | Ingestion durability (`"async"` for batching, `"sync"` for fsync) | `"async"` |
| `BSDB_SEARCH_AUTO_REFRESH_SECS` | Background flush interval for memory buffers | `1.0` |

---

## Running the Gateway

```bash
# Start gateway listening on ports 9200 and 6333
bsdb-search --bind 127.0.0.1 --port 9200 --storage-uri s3://my-bucket/search
```

---

## Client Usage Examples

### 1. OpenSearch / Elasticsearch REST API (Port 9200)

```bash
# Cluster health check
curl -H "Authorization: Bearer $BSDB_API_KEY" http://localhost:9200/_cluster/health

# Create an index
curl -X PUT http://localhost:9200/articles -H "Content-Type: application/json" -d '{
  "mappings": {
    "properties": {
      "title": { "type": "text" },
      "category": { "type": "keyword" }
    }
  }
}'

# Ingest a document
curl -X POST http://localhost:9200/articles/_doc/1 -H "Content-Type: application/json" -d '{
  "title": "BenoStreamDB Lakehouse Search",
  "category": "database"
}'

# Search documents
curl -X POST http://localhost:9200/articles/_search -H "Content-Type: application/json" -d '{
  "query": {
    "match": { "title": "Lakehouse" }
  }
}'
```

### 2. Qdrant REST API (Port 6333)

```python
from qdrant_client import QdrantClient

# Connect to the BenoStreamDB Qdrant interface
client = QdrantClient(url="http://localhost:6333", api_key="your-api-key")

# List collections (maps to BenoStreamDB tables)
collections = client.get_collections()

# Search points
results = client.search(
    collection_name="embeddings",
    query_vector=[0.12, 0.45, -0.23],
    limit=5
)
for r in results:
    print(r.id, r.score, r.payload)
```
