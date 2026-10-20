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
  - **Multimodal Images** (`.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`, `.svg`): Byte conversion to base64 Data URIs (`data:image/...`) ready for vision LLMs. Image vectors are generated only when `embed_chunks: true` **and** an HTTP multimodal endpoint is configured; the local text model does not embed images.
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

`benostreamdb-mcp` supports multi-channel JSON configuration: CLI flags, JSON string/file flags, environment variables, and standard MCP `initializationOptions` during the JSON-RPC `initialize` handshake.

#### 1. Claude Desktop (`claude_desktop_config.json`)

On macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`  
On Linux: `~/.config/Claude/claude_desktop_config.json`  
On Windows: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "benostreamdb": {
      "command": "uvx",
      "args": [
        "benostreamdb-mcp",
        "--warehouse", "/path/to/warehouse",
        "--readwrite-schemas", "scratch,sandbox",
        "--readonly-schemas", "default,finance_raw"
      ],
      "env": {
        "BSDB_GPU_DEVICE": "auto",
        "BSDB_EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5"
      }
    }
  }
}
```

#### 2. Antigravity IDE / Cursor / VS Code (`settings.json` or `mcp_config.json`)

You can configure options directly via CLI flags, environment variables, or inline JSON configuration:

```json
{
  "mcpServers": {
    "benostreamdb": {
      "command": "/path/to/benostreamdb/target/release/benostreamdb-mcp",
      "args": [
        "--config", "{\"warehouse\":\"/path/to/warehouse\",\"readwrite_schemas\":[\"scratch\",\"analytics\"],\"readonly_schemas\":[\"default\",\"raw\"]}"
      ],
      "env": {
        "BSDB_EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5",
        "BSDB_EMBEDDING_CACHE_DIR": "/home/me/.benostreamdb/models"
      }
    }
  }
}
```

#### 3. MCP `initializationOptions` (Dynamic JSON Configuration)

MCP clients that support `initializationOptions` during protocol negotiation can send configuration parameters directly without modifying server launch arguments:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2024-11-05",
    "capabilities": {},
    "clientInfo": { "name": "Antigravity", "version": "1.0.0" },
    "initializationOptions": {
      "warehouse": "/data/local_warehouse",
      "readwrite_schemas": ["scratch", "workspace"],
      "readonly_schemas": ["default", "archive"],
      "default_write_schema": "scratch",
      "ephemeral": false
    }
  }
}
```

---

## Storage Lockdown & The Dual-Lock Security Model

### The "Right Call" on Hosted Iceberg Tables & Object Stores

When agents interact with analytical catalogs (Iceberg, Delta Lake, object stores), **relying solely on cloud IAM roles is necessary but dangerous on its own**. 

In staging, local development, or hybrid setups:
1. Developers often run MinIO, RustFS, or local S3-compatible object stores where IAM roles may not be configured.
2. Even in cloud environments (AWS, GCP, Azure), CI/CD runners or dev machines frequently have shared write credentials or wildcard S3/GCS bucket write permissions.
3. If an autonomous agent or untrusted SQL query could dynamically mount remote catalogs or execute `LOCATION 's3://...'` DDL, it could accidentally mutate or overwrite production data tables.

Therefore, `benostreamdb-mcp` implements a **Dual-Lock Defense-in-Depth Model**:

1. **Engine-Level Storage Lockdown (AST & URI Barrier)**:
   - **Remote URI Guard**: Strictly forbids remote URI schemes (`s3://`, `minio://`, `rustfs://`, `gcs://`, `http://`, `https://`, `iceberg://`, `rest://`, `abfs://`) in warehouse paths, `export_table`, and `ingest_file`. All storage operations must target local filesystem paths.
   - **Catalog Mounting Ban**: Unconditionally blocks SQL statements with `LOCATION ...`, `CREATE EXTERNAL TABLE`, `ATTACH DATABASE`, `CREATE CATALOG`, `MOUNT`, or engine setting overrides (`SET BENOSTREAM.*`).
   - **Schema Isolation Barrier**: Tables in protected catalogs/schemas (`default`, `public`, `finance_raw`, etc.) are immutable. Write operations (`CREATE TABLE`, `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`) are rejected unless explicitly targeted at configured writable schemas (default `scratch`).
2. **Infrastructure-Level IAM Barrier**:
   - Production IAM roles (e.g. AWS IAM, GCP IAM, Azure RBAC) provide the second lock at the cloud bucket level for release pipelines.

---

## Catalog Policy & Trusted vs. Untrusted Connectors

BenoStreamDB establishes a formal security boundary between **trusted core analytics engines** and **untrusted user/agent connectors**:

### 1. Trusted vs. Untrusted Boundary

| Category | Components | Permissions Policy & Failure Mode |
|:---|:---|:---|
| **Trusted Engines** | Core BenoStreamDB kernel, Arrow Flight SQL server, Spark, Trino, dbt | Standard database authority. Operates with full read/write privileges. Can manage catalogs, namespaces, and call policy management APIs (`session.set_catalog_policy(...)`). |
| **Untrusted Connectors** | BenoStreamDB MCP (`benostreamdb-mcp`), BenoStream Search REST API (`benostreamdb-search`), external AI agents | **Zero-Trust / Conservative Lockdown**: If `.benostream-policy.json` is missing from the warehouse, untrusted connectors are **strictly locked to read-only** across all persistent catalog schemas. Only the ephemeral `scratch` schema is writable. |

### 2. Authoritative Catalog Policy (`.benostream-policy.json`)

To eliminate complex client-side configuration in `settings.json`, permissions are stored directly at the root of the warehouse or bucket as an authoritative catalog property file:

```json
{
  "catalog_id": "enterprise-lakehouse",
  "policy_version": "1.0",
  "readwrite_schemas": ["scratch", "sandbox"],
  "readonly_schemas": ["finance_raw", "sec_edgar"],
  "immutable_schemas": ["default", "public", "information_schema"],
  "default_write_schema": "scratch"
}
```

- When an untrusted connector mounts the warehouse, it reads `.benostream-policy.json`. Client attempts to override locked schemas are rejected.
- **Fail-Safe Conservative Rule**: If `.benostream-policy.json` is absent, the untrusted connector defaults to complete read-only lockdown on all persistent schemas (only `scratch` allowed).

### 3. Core Database Policy Management APIs

The core database product and administrators can inspect and update policies programmatically:

```rust
use benostreamdb::core::catalog::{get_catalog_policy, set_catalog_policy, set_catalog_schema_access, lock_catalog_read_only, CatalogPolicy};

// Lock down a schema as read-only
set_catalog_schema_access("/path/to/warehouse", "finance_raw", true)?;

// Grant an untrusted connector write access to a new sandbox schema
set_catalog_schema_access("/path/to/warehouse", "sandbox", false)?;

// Lock down all persistent schemas to read-only
lock_catalog_read_only("/path/to/warehouse")?;
```

Or via `BenoStreamSession`:

```rust
session.set_schema_access("finance_raw", true)?;
```

---

## Configuration Reference

`benostreamdb-mcp` can be configured via CLI flags, environment variables, or JSON:

| CLI Option | Environment Variable | JSON Key | Description | Default |
|:---|:---|:---|:---|:---|
| `--warehouse <PATH>` | `BSDB_WAREHOUSE` | `"warehouse"` | Local directory path for persistent Iceberg table storage. | System temp dir / `resolve_default_warehouse()` |
| `--readwrite-schemas <S>` | `BSDB_READWRITE_SCHEMAS` | `"readwrite_schemas"` | Comma-separated or JSON array of schemas allowing writes. | `["scratch"]` |
| `--readonly-schemas <S>` | `BSDB_READONLY_SCHEMAS` | `"readonly_schemas"` | Comma-separated or JSON array of schemas locked as read-only. | `["default", "public"]` |
| `--default-write-schema <S>`| `BSDB_DEFAULT_WRITE_SCHEMA`| `"default_write_schema"`| Default schema for unqualified table creations. | `"scratch"` |
| `--ephemeral` | `BSDB_EPHEMERAL` | `"ephemeral"` | If true (`1`), uses an ephemeral temporary directory for the session. | `false` |
| `--config <JSON>` | `BSDB_MCP_CONFIG` | N/A | Full JSON configuration string. | None |
| `--config-file <PATH>` | N/A | N/A | Path to a `.json` configuration file. | None |
| N/A | `BSDB_GPU_DEVICE` | N/A | Hardware acceleration target or pool for vector indexes (`auto`, `cpu`, `cuda[:N]`, `mps`, `cuda:0,cuda:1`). | `auto` |
| `--embedding-model <HF_ID>` | `BSDB_EMBEDDING_MODEL` | `"embedding": {"model": "..."}` | Hugging Face BERT-family embedding model. | `BAAI/bge-small-en-v1.5` |
| `--embedding-provider <P>` | `BSDB_EMBEDDING_PROVIDER` | `"embedding": {"provider": "..."}` | `local` (in-process Candle), `http`, or `none`. | `local` unless `url` is set |
| `--embedding-cache-dir <PATH>` | `BSDB_EMBEDDING_CACHE_DIR` | `"embedding": {"cache_dir": "..."}` | Private model cache; reused after the first download. | `<data-dir>/models` |
| `--embedding-url <URL>` | `BSDB_EMBEDDING_URL` | `"embedding": {"url": "..."}` | Ollama / OpenAI-compatible / TEI endpoint. | None |
| `--embedding-api-key <KEY>` | `BSDB_EMBEDDING_API_KEY` | `"embedding": {"api_key": "..."}` | Bearer token for the HTTP endpoint. | None |
| `--embedding-batch-size <N>` | `BSDB_EMBEDDING_BATCH_SIZE` | `"embedding": {"batch_size": N}` | Candle inference batch size. | `16` |
| `--embedding-max-length <N>` | `BSDB_EMBEDDING_MAX_LENGTH` | `"embedding": {"max_length": N}` | Tokenizer truncation length. | `512` |
| `--embedding-dim <N>` | `BSDB_EMBEDDING_DIM` | `"embedding": {"dim": N}` | Optional dimension override (MRL truncation). Otherwise read from the model config. | Model default |
| N/A | `BSDB_EMBEDDING_QUERY_PREFIX` | `"embedding": {"query_prefix": "..."}` | Override the query instruction prefix. Empty string disables it. | Model default |
| N/A | `BSDB_EMBEDDING_DOCUMENT_PREFIX` | `"embedding": {"document_prefix": "..."}` | Override the document instruction prefix. Empty string disables it. | Model default |
| `--no-embeddings` | N/A | `"embedding": {"provider": "none"}` | Disable embedding generation entirely. | Off |

> Legacy JSON keys `model_path`, `device`, `quantized`, `intra_threads`, `pooling`, and
> `show_download_progress` are still accepted for backward compatibility but are ignored by the
> Candle loader. Local inference runs on CPU in the published build.

> [!TIP]
> `local` is the default and needs no API key or daemon. The first embedding call downloads
> `config.json`, `tokenizer.json`, and `model.safetensors` into the BenoStreamDB private cache
> (`~/.benostreamdb/models` by default) and reuses it on every subsequent start.
> `BSDB_EMBEDDING_URL` switches to HTTP automatically when `provider` is not set, preserving the
> previous Ollama/OpenAI-compatible workflow.

#### Embedding configuration

```json
{
  "warehouse": "/path/to/warehouse",
  "embedding": {
    "provider": "local",
    "model": "BAAI/bge-small-en-v1.5",
    "cache_dir": "/home/me/.benostreamdb/models",
    "batch_size": 16,
    "max_length": 512
  }
}
```

**How embeddings are used**

Embedding generation is automatic once a provider is configured (local is the default):

| Tool | Embedding behavior |
|:---|:---|
| `ingest_file` | Only when `embed_chunks: true`. Text/code/PDF/Office chunks are embedded; images require an HTTP multimodal endpoint. |
| `ingest_records` | Only when `embed_col` names a text column. |
| `store_memory` | Auto-embeds `content` unless an `embedding` is supplied. |
| `recall_memory` | Auto-embeds `query` unless an `embedding` is supplied. |
| `hybrid_search` | Auto-embeds `query_text` unless `query_vector` is supplied. |
| `graph_rag_search` | Requires `query_vector`; compute it with the same model as the table. |

**Local Hugging Face model (default)**

- `BAAI/bge-small-en-v1.5` is retrieval-tuned, 384-d, and runs as F32 safetensors through Candle.
- Embeddings use CLS pooling followed by L2 normalization. Documents are embedded plain; queries get
  `Represent this sentence for searching relevant passages: `. The prefix is applied automatically.
  Override it with `query_prefix` / `BSDB_EMBEDDING_QUERY_PREFIX` only if the model expects
  something different.
- The vector dimension is inferred from the model: the curated id map supplies it before load, and
  `config.json` (`hidden_size`) is authoritative once the model is loaded. Set `dim` only to
  truncate a Matryoshka (MRL) model.
- The model is downloaded once into `~/.benostreamdb/models` (or `$XDG_DATA_HOME/benostreamdb/models`)
  and reused. Set `cache_dir` / `BSDB_EMBEDDING_CACHE_DIR` to move it.
- Loading is lazy: the first embedding call pays the model load; `initialize` and SQL-only sessions
  do not.
- Candle runs on CPU in the published build; CUDA/Metal backends are not enabled in the wheel.

**Selecting a different local model**

Any BERT-family Hugging Face model that publishes `model.safetensors`, `config.json`, and
`tokenizer.json` can be selected by id:

```json
{ "embedding": { "model": "BAAI/bge-base-en-v1.5" } }
```

Common choices: `BAAI/bge-base-en-v1.5` (768-d), `BAAI/bge-large-en-v1.5` (1024-d),
`intfloat/multilingual-e5-small`, `nomic-ai/nomic-embed-text-v1.5`,
`mixedbread-ai/mxbai-embed-large-v1`, `jinaai/jina-embeddings-v2-base-code`.

Models that only ship `pytorch_model.bin` must be converted to safetensors, or served through an
HTTP embedding endpoint (next section). Non-BERT architectures and multimodal models also require
an HTTP endpoint.

**Override with an OpenAI-compatible / LLM vendor endpoint**

Embeddings must come from an **embedding** endpoint, not a chat completion. Set
`provider: "http"` and point `url` at the vendor endpoint:

Inside the server JSON/config file:

```json
{
  "embedding": {
    "provider": "http",
    "model": "text-embedding-3-small",
    "url": "https://api.openai.com/v1/embeddings",
    "api_key": "sk-..."
  }
}
```

Complete MCP client settings example (preferred: keep the key in an environment variable rather
than the JSON file):

```json
{
  "mcpServers": {
    "benostreamdb": {
      "command": "benostreamdb-mcp",
      "args": ["--warehouse", "/path/to/warehouse"],
      "env": {
        "BSDB_EMBEDDING_PROVIDER": "http",
        "BSDB_EMBEDDING_MODEL": "text-embedding-3-small",
        "BSDB_EMBEDDING_URL": "https://api.openai.com/v1/embeddings",
        "BSDB_EMBEDDING_API_KEY": "sk-..."
      }
    }
  }
}
```

The local model is not loaded and the model cache is unused when `provider` is `http`. The
`embedding` object and the `BSDB_EMBEDDING_*` variables are equivalent; use whichever your MCP client
supports.

Ollama:

```json
{
  "embedding": {
    "provider": "http",
    "model": "nomic-embed-text",
    "url": "http://localhost:11434/api/embeddings"
  }
}
```

Hugging Face Text Embeddings Inference (TEI) or any OpenAI-compatible server:

```json
{
  "embedding": {
    "provider": "http",
    "model": "BAAI/bge-small-en-v1.5",
    "url": "http://localhost:8080/v1/embeddings"
  }
}
```

When `provider` is `http`, the `model` string is sent verbatim. Instruction prefixes are applied
only for known BGE/E5/Nomic ids; OpenAI-style ids such as `text-embedding-3-small` are sent as-is.

**Changing models forces a re-embed.** This server stamps `benostream.embedding.model`,
`benostream.embedding.dim`, and the provider/prefix onto the table as Iceberg table properties. Those
properties are mirrored into `metadata/vN.metadata.json`, so Spark, Trino, and PyIceberg can read them
as well; `ALTER TABLE ... SET TBLPROPERTIES` from any SQL client is visible here too.

The engine enforces the same contract on every connector that writes through it: Python and SQL writes
reject vectors whose length does not match the declared dimension, and changing or removing the declared
model/dim on a non-empty table is rejected at the commit point (including Iceberg REST `set-properties`
from Spark/Trino). The engine derives `dim` from a known model id (BGE, E5, Nomic, OpenAI, ...) or from
a `FixedSizeList` column when you do not set it, so only custom models and MRL truncation need an
explicit `dim`. Model identity itself is not recoverable from a vector, so a client that rewrites vectors
outside the table is out of scope. Fixed-size vector columns are checked from the schema on every
write; set `BSDB_VALIDATE_EMBEDDINGS=strict` to also scan list/stringified vectors row by row. Recreate
or re-ingest the table when the model changes so query and document vectors never mix spaces.
`provider: "none"` disables embedding generation entirely and avoids all model work.

**Images:** `ingest_file` accepts PNG/JPEG/WebP/GIF/SVG and stores a base64 data URI for vision
LLMs. Local text models do not embed images; use `provider: "http"` with a multimodal endpoint if
you want image vectors. Do not mix CLIP/image vectors with `bge-small` text vectors in the same
column.

---

## Complete MCP Tool Reference (12 Tools)

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
  - `embedding` (*array of numbers*, optional): Pre-computed vector. If omitted, the configured provider embeds `content` automatically (local BGE-small by default).

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
  - `embed_chunks` (*boolean*, optional, default `false`): Compute embeddings for extracted text/code/PDF/Office chunks. Images are only embedded when an HTTP multimodal endpoint is configured; the local text model skips them.

### 12. `list_embedding_models`
Reports the active embedding provider, resolved model, dimension, cache directory, device, load state, query/document prefixes, and HTTP endpoint (if any), then enumerates a curated shortlist of Hugging Face models for local, no-LLM inference. Use the returned `huggingface_id` as `embedding.model`.
- **Arguments**: None.
- **Example response**:
  ```json
  {
    "provider": "local",
    "model": "BAAI/bge-small-en-v1.5",
    "resolved_model": "BAAI/bge-small-en-v1.5",
    "dim": 384,
    "cache_dir": "/home/me/.benostreamdb/models",
    "device": "cpu",
    "loaded": false,
    "providers": ["local", "http", "none"],
    "local_models": [
      { "huggingface_id": "BAAI/bge-small-en-v1.5", "dim": 384 },
      { "huggingface_id": "BAAI/bge-base-en-v1.5", "dim": 768 }
    ]
  }
  ```

---

## Supported File Ingestion Formats

| Format Category | Supported Extensions | Processing Engine | Ingested Fields |
|:---|:---|:---|:---|
| **PDF Documents** | `.pdf` | `pdf-extract` (pure Rust) | `path`, `chunk_index`, `start_line`, `end_line`, `content`, `format`, `embedding` (optional) |
| **Excel & Spreadsheets** | `.xlsx`, `.xls`, `.xlsb`, `.ods` | `calamine` (pure Rust) | `sheet`, `row_index`, `<col_headers...>`, `embedding` (optional) |
| **Multimodal Images** | `.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`, `.svg` | Base64 Data URI engine | `path`, `format`, `content` (`data:image/...;base64`), `size_bytes`, `embedding` (optional; HTTP multimodal only) |
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

All 45 unit tests cover:
- Protocol initialization & tools listing
- JSON configuration & dynamic MCP `initializationOptions` updates
- Storage lockdown: rejection of remote URIs (`s3://`, `minio://`, `rustfs://`, `gcs://`) and location clauses
- Configurable schema read/write permissions & isolation guards
- Scratch table creation, record insertion, and querying
- Parquet, CSV, and JSON export workflows
- Graph table discovery and metadata tagging
- CDC event subscriptions
- Memory storage and semantic recall
- Reciprocal rank fusion hybrid search
- Universal file ingestion:
  - PNG, JPG, and SVG multimodal image ingestion with base64 Data URIs
  - Text and source code chunking with line offsets
  - PDF document extraction & error handling
  - Spreadsheet workbook parsing & corrupted archive rejection
  - Word (.docx) and PowerPoint (.pptx) extraction
  - XML and ZIP archive ingestion
  - CSV, TSV, and PSV delimiter-separated parsing
  - JSON and NDJSON structured dataset loading
- Local Hugging Face embedding configuration, model/dimension resolution, and private cache
- OpenAI-compatible and Ollama embedding endpoints, query/document prefixes, and bearer auth
- Embedding model/dimension provenance, mismatch refusal, and legacy-table behavior

---

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](../../LICENSE) for details.
