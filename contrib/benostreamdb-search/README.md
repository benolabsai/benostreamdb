# BenoStreamDB Search (`benostreamdb-search`)

OpenSearch / Elasticsearch 7.10 and Qdrant REST compatibility gateway for BenoStreamDB.

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
