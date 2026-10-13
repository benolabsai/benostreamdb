# BenoStreamDB Model Context Protocol (MCP) Server

[![crates.io](https://img.shields.io/crates/v/benostreamdb-mcp.svg)](https://crates.io/crates/benostreamdb-mcp)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

`benostreamdb-mcp` is a high-performance Rust server implementing the open [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) for **BenoStreamDB**.

It exposes analytical streaming SQL, vector similarity search, reciprocal rank fusion (RRF) hybrid search, GraphRAG topological traversals, multi-modal ingestion (PDF, Excel, images, code, text), long-term agent memory, and live change data capture (CDC) directly to autonomous AI agents and IDEs (including Claude Desktop, Cursor, Antigravity IDE, and custom agentic frameworks) over standard input/output (stdio JSON-RPC).

---

## Key Capabilities

- **Analytical Streaming SQL**: Powered by Apache DataFusion with native SIMD-accelerated execution.
- **pgvector-Compatible Vector UDFs**: `vector_distance`, `l2_distance`, `cosine_distance`, and `inner_product` over `Vec<f32>` and stringified vector columns.
- **Hybrid Search (BM25 + Dense Vectors)**: Reciprocal Rank Fusion (RRF) combining keyword BM25 retrieval with vector similarity.
- **GraphRAG Search**: Simultaneous vector search and multi-hop CSR graph traversals for connected knowledge graphs.
- **Persistent Long-Term Agent Memory**: `store_memory` and `recall_memory` tools with automated embedding generation, metadata tagging, and similarity scoring.
- **Multi-Modal Document & Asset Ingestion**:
  - **PDF Documents** (`.pdf`): In-memory full-text extraction via `pdf-extract` with line-bounded semantic chunking.
  - **Excel & Spreadsheets** (`.xlsx`, `.xls`, `.xlsb`, `.ods`): Pure-Rust workbook parsing via `calamine` with automatic schema inference, sheet & row tracking, and typed columns.
  - **Multimodal Images** (`.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`, `.svg`): Byte conversion to base64 Data URIs (`data:image/...`) ready for vision LLMs, with optional multimodal vision embeddings.
  - **Source Code & Text** (`.rs`, `.py`, `.ts`, `.md`, `.txt`, `.csv`, `.log`): Configurable line chunking with start/end line offsets.
  - **JSON & NDJSON** (`.json`, `.jsonl`, `.ndjson`): Direct batch or streaming record ingestion.
- **Safe Scratchpad Isolation**: Strict security guard ensuring the primary catalog remains read-only, while granting agents complete freedom to create, populate, query, and index ephemeral tables in the `scratch` schema.
- **Live Change Data Capture (CDC)**: Poll real-time table mutations and stream events using `subscribe_events`.
- **Built-in Agent Prompts & Resources**: Pre-packaged prompt engineering templates (`benostreamdb_sql_expert`, `agent_memory_guide`, `code_rag_strategy`) and table schema inspection resources (`benostreamdb://tables/{schema}/{name}`).

---

## Quick Start with `uvx`

Run `benostreamdb-mcp` instantly without local compilation using [`uv`](https://github.com/astral-sh/uv):

```bash
uvx benostreamdb-mcp
```

### Client Configuration

#### 1. Claude Desktop (`claude_desktop_config.json`)

On macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`  
On Linux: `~/.config/Claude/claude_desktop_config.json`  
On Windows: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "benostreamdb": {
      "command": "uvx",
      "args": ["benostreamdb-mcp"],
      "env": {
        "BSDB_WAREHOUSE": "/path/to/warehouse",
        "BSDB_GPU_DEVICE": "auto",
        "BSDB_EMBEDDING_URL": "http://localhost:11434/api/embeddings",
        "BSDB_EMBEDDING_MODEL": "all-minilm"
      }
    }
  }
}
```

#### 2. Antigravity IDE / Cursor (`mcp_config.json` or `settings.json`)

```json
{
  "mcpServers": {
    "benostreamdb": {
      "command": "/path/to/benostreamdb/target/release/benostreamdb-mcp",
      "args": [],
      "env": {
        "BSDB_WAREHOUSE": "/path/to/warehouse",
        "BSDB_EMBEDDING_URL": "http://localhost:11434/api/embeddings",
        "BSDB_EMBEDDING_MODEL": "all-minilm"
      }
    }
  }
}
```

---

## Installation Options

### From Source (Rust Cargo)

```bash
# Clone repository
git clone https://github.com/benolabsai/benostreamdb.git
cd benostreamdb

# Build optimized release binary
cargo build --release -p benostreamdb-mcp

# Binary is available at target/release/benostreamdb-mcp
./target/release/benostreamdb-mcp
```

### Via Python / Pip

```bash
pip install benostreamdb-mcp
benostreamdb-mcp
```

---

## Environment Variables & Configuration

| Variable | Description | Default |
|:---|:---|:---|
| `BSDB_WAREHOUSE` | Root directory path or object store URI for persistent tables. | System temporary directory |
| `BSDB_GPU_DEVICE` | Hardware acceleration target (`auto`, `cpu`, `cuda`, `mps`). | `auto` |
| `BSDB_CONFIG` | Path to YAML/TOML catalog configuration file. | None |
| `BSDB_EMBEDDING_URL` | Endpoint for generating vector embeddings (e.g. Ollama or OpenAI compatible). | None |
| `BSDB_EMBEDDING_MODEL` | Model identifier to pass to the embedding endpoint. | `all-minilm` |
| `BSDB_EMBEDDING_API_KEY`| Bearer authentication token for external embedding APIs. | None |

> [!TIP]
> If `BSDB_EMBEDDING_URL` is omitted, `ingest_file`, `ingest_records`, `store_memory`, and `hybrid_search` continue to function seamlessly using pre-computed vectors or text-only retrieval.

---

## Complete MCP Tool Reference (11 Tools)

### 1. `execute_sql`
Executes an arbitrary SQL query against the database engine.
- **Security Guard**: Queries on persistent schemas (`default`, `public`, etc.) are read-only (`SELECT`, `EXPLAIN`). Writes, mutations, DDL, and index creations are strictly permitted on the isolated `scratch` schema (`scratch.<table_name>`).
- **Arguments**:
  - `query` (*string*, required): The SQL query string.
- **Example**:
  ```sql
  SELECT id, content, vector_distance(embedding, [0.12, 0.45, -0.08]) AS distance
  FROM scratch.doc_chunks
  ORDER BY distance ASC
  LIMIT 5;
  ```

### 2. `create_scratch_table`
Creates an ephemeral table in the isolated `scratch` schema for intermediate reasoning, multi-step agent plans, or temporary caching.
- **Arguments**:
  - `table_name` (*string*, required): Name of the scratch table (without prefix, or prefixed with `scratch.`).
  - `schema_ddl` (*string*, required): Column definitions (e.g., `id INT, key VARCHAR, val DOUBLE`).
  - `table_type` (*string*, optional): Role tag (`"node"`, `"edge"`, or `"table"`).

### 3. `export_table`
Exports any table from the catalog or `scratch` schema to an external file (Parquet, CSV, or JSON).
- **Arguments**:
  - `table_name` (*string*, required): Source table name.
  - `export_path` (*string*, required): Absolute target file path.
  - `format` (*string*, optional): `"parquet"` (default), `"csv"`, or `"json"`.

### 4. `list_graph_tables`
Inspects the catalog and returns all tables, distinguishing standard tables from declared Property Graph Node and Edge tables (with endpoint and label metadata).
- **Arguments**: None.

### 5. `subscribe_events`
Polls change data capture (CDC) events and mutations on any table.
- **Arguments**:
  - `table` (*string*, required): Table name to monitor.
  - `max_events` (*integer*, optional, default `100`): Maximum events to drain.
  - `timeout_ms` (*integer*, optional, default `500`): Poll timeout in milliseconds.

### 6. `hybrid_search`
Executes Reciprocal Rank Fusion (RRF) search across a table containing text and vector embeddings. Combines BM25 lexical rank with dense vector similarity rank.
- **Arguments**:
  - `table` (*string*, required): Target table.
  - `query_text` (*string*, required): Keyword search query.
  - `query_vector` (*array of numbers*, optional): Dense query embedding vector.
  - `top_k` (*integer*, optional, default `10`): Number of ranked records to return.
  - `bm25_weight` (*number*, optional, default `1.0`): Weight multiplier for keyword rank.
  - `vector_weight` (*number*, optional, default `1.0`): Weight multiplier for vector similarity rank.

### 7. `graph_rag_search`
Performs Graph Retrieval-Augmented Generation (GraphRAG). Finds relevant seed nodes via vector similarity and traverses their topological graph neighborhood using CSR graph indexes.
- **Arguments**:
  - `node_table` (*string*, required): Table containing node documents and embeddings.
  - `edge_table` (*string*, required): Table containing graph edges.
  - `query_vector` (*array of numbers*, required): Query embedding vector.
  - `top_k_nodes` (*integer*, optional, default `5`): Number of seed nodes to retrieve.
  - `traversal_depth` (*integer*, optional, default `2`): Graph traversal hops.

### 8. `store_memory`
Stores persistent semantic memory for the AI agent with automated or custom embedding generation, categorization tags, and timestamps.
- **Arguments**:
  - `key` (*string*, required): Unique memory key or identifier.
  - `content` (*string*, required): The text memory or concept to remember.
  - `tags` (*array of strings*, optional): Filtering tags (e.g. `["user_pref", "ui"]`).
  - `embedding` (*array of numbers*, optional): Pre-computed vector. If omitted and `BSDB_EMBEDDING_URL` is configured, an embedding is generated automatically.

### 9. `recall_memory`
Retrieves memories matching a query text or vector, sorted by semantic similarity score, with optional tag filtering.
- **Arguments**:
  - `query` (*string*, optional): Keyword search text.
  - `embedding` (*array of numbers*, optional): Vector search embedding.
  - `tag` (*string*, optional): Filter by tag.
  - `top_k` (*integer*, optional, default `5`): Maximum memories to recall.

### 10. `ingest_records`
Unified JSON record ingestion tool into scratch tables. Automatically creates the scratch table if it does not already exist, and optionally generates dense vector embeddings for a specified text column (`embed_col`).
- **Arguments**:
  - `table_name` (*string*, required): Target scratch table name.
  - `records` (*array of objects*, required): Array of JSON objects to insert.
  - `embed_col` (*string*, optional): Column name whose string values will be vectorized into an `embedding` vector column.

### 11. `ingest_file`
Universal file ingestion tool supporting documents, spreadsheets, multimodal images, source code, and structured datasets.
- **Arguments**:
  - `file_path` (*string*, required): Absolute path to the file.
  - `table_name` (*string*, optional): Destination table in the `scratch` schema. Defaults to the sanitized file stem (e.g. `report_2024.pdf` becomes `scratch.report_2024`).
  - `chunk_size` (*integer*, optional, default `500`): Character chunk size for text/code/PDF line chunking.
  - `embed_chunks` (*boolean*, optional, default `false`): Whether to compute embeddings for extracted chunks or images.

---

## Supported File Ingestion Formats

| Format Category | Supported Extensions | Processing Engine | Ingested Fields |
|:---|:---|:---|:---|
| **PDF Documents** | `.pdf` | `pdf-extract` (pure Rust) | `path`, `chunk_index`, `start_line`, `end_line`, `content`, `format`, `embedding` (optional) |
| **Excel & Spreadsheets** | `.xlsx`, `.xls`, `.xlsb`, `.ods` | `calamine` (pure Rust) | `sheet`, `row_index`, `<col_headers...>`, `embedding` (optional) |
| **Multimodal Images** | `.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`, `.svg` | Base64 Data URI engine | `path`, `format`, `content` (`data:image/...;base64`), `size_bytes`, `embedding` (optional) |
| **Source Code & Text** | `.rs`, `.py`, `.js`, `.ts`, `.md`, `.txt`, `.csv`, `.sql`, `.log` | Built-in line-based chunker | `path`, `chunk_index`, `start_line`, `end_line`, `content`, `embedding` (optional) |
| **JSON & NDJSON** | `.json`, `.jsonl`, `.ndjson` | `serde_json` | Direct structured table rows matching JSON schema |

---

## Built-in Prompts

`benostreamdb-mcp` exposes 3 pre-engineered prompts via `prompts/list` and `prompts/get`:

1. **`benostreamdb_sql_expert`**: Guides the LLM on formulating optimized DataFusion SQL, pgvector distance queries, graph functions (`graph_neighbors`, `graph_shortest_path`), and scratchpad best practices.
2. **`agent_memory_guide`**: Guides autonomous agents on how to partition, tag, store, and recall knowledge over multi-turn conversations.
3. **`code_rag_strategy`**: Instructs the agent on how to chunk, index, and query large source code repositories using hybrid search.

---

## Built-in Resources

Agents can inspect database resources via `resources/list` and `resources/read`:

- `benostreamdb://tables/{schema}/{table}`: Returns the Arrow / Iceberg table schema, column types, and metadata.
- `benostreamdb://docs/sql_manual.md`: Returns the comprehensive built-in SQL reference manual covering all UDFs, indexing strategies, and graph operations.

---

## Testing

Run the full MCP test suite:

```bash
cargo test -p benostreamdb-mcp
```

All 16 unit tests cover:
- Protocol initialization & tools listing
- Scratch schema write isolation guard
- Scratch table creation, record insertion, and querying
- Parquet export workflow
- Graph table discovery and metadata tagging
- CDC event subscriptions
- Memory storage and semantic recall
- Reciprocal rank fusion hybrid search
- Universal file ingestion:
  - PNG and SVG multimodal image ingestion with base64 Data URIs
  - Text and source code chunking with line offsets
  - PDF document extraction & error handling
  - Spreadsheet workbook parsing & corrupted archive rejection
  - JSON and NDJSON structured dataset loading

---

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](../../LICENSE) for details.
