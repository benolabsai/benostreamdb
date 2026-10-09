use anyhow::Result;
use serde_json::{json, Value};
use std::io::Write;
use tokio::io::{AsyncBufReadExt, BufReader};
use tracing::{debug, error, info};

use crate::protocol::{JsonRpcRequest, JsonRpcResponse};
use benostreamdb::core::sql::session::BenoStreamSession;

/// The only schema the MCP server may create objects in.
const SCRATCH_SCHEMA: &str = "scratch";

/// Enforce the MCP write policy: **read any schema, write only to `scratch`**.
///
/// The MCP server is driven by an untrusted agent, so it must not be able to
/// mutate the main catalog. Every write statement's target table must be in the
/// `scratch` schema; reads (SELECT, COPY ... TO, DESCRIBE, …) are unrestricted.
fn guard_scratch_only(query: &str) -> anyhow::Result<()> {
    let collapsed = query
        .to_ascii_uppercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");

    if collapsed.contains("CREATE SCHEMA") || collapsed.contains("CREATE DATABASE") {
        anyhow::bail!("MCP may not create schemas or databases");
    }

    // (keyword, the table name follows it after any IF [NOT] EXISTS).
    let write_stmts = [
        "CREATE EXTERNAL TABLE",
        "CREATE TABLE",
        "INSERT INTO",
        "UPDATE",
        "DELETE FROM",
        "DROP TABLE",
        "ALTER TABLE",
        "TRUNCATE TABLE",
        "OPTIMIZE TABLE",
        "VACUUM",
    ];
    for kw in write_stmts {
        if let Some(pos) = collapsed.find(kw) {
            let rest = collapsed[pos + kw.len()..].trim_start();
            let rest = rest.strip_prefix("IF NOT EXISTS ").unwrap_or(rest);
            let rest = rest.strip_prefix("IF EXISTS ").unwrap_or(rest);
            let name = rest
                .split_whitespace()
                .next()
                .unwrap_or("")
                .trim_matches(|c| c == '"' || c == '`' || c == '(');
            let schema = name
                .split('.')
                .rev()
                .nth(1)
                .unwrap_or("")
                .to_ascii_lowercase();
            if schema != SCRATCH_SCHEMA {
                anyhow::bail!(
                    "MCP may only write to the '{}' scratch schema (got '{}')",
                    SCRATCH_SCHEMA,
                    name
                );
            }
        }
    }

    // `CREATE INDEX <name> ON <table>` — the target follows ` ON `. Indexes may
    // only be created on scratch tables (the main catalog is read-only).
    if let Some(pos) = collapsed.find("CREATE INDEX") {
        if let Some(on) = collapsed[pos..].find(" ON ") {
            let rest = collapsed[pos + on + 4..].trim_start();
            let name = rest
                .split_whitespace()
                .next()
                .unwrap_or("")
                .trim_matches(|c| c == '"' || c == '`' || c == '(');
            let schema = name
                .split('.')
                .rev()
                .nth(1)
                .unwrap_or("")
                .to_ascii_lowercase();
            if schema != SCRATCH_SCHEMA {
                anyhow::bail!(
                    "MCP may only create indexes on the '{}' scratch schema (got '{}')",
                    SCRATCH_SCHEMA,
                    name
                );
            }
        }
    }
    // `DROP INDEX <name>` does not name a table, so it cannot be verified as
    // scratch — reject it (use `ALTER TABLE scratch.t DROP INDEX` instead).
    if collapsed.contains("DROP INDEX") {
        anyhow::bail!("MCP may not drop indexes directly; use ALTER TABLE scratch.t DROP INDEX");
    }
    Ok(())
}

pub struct McpServer {
    session: BenoStreamSession,
}

impl McpServer {
    pub fn new() -> Self {
        let mut session = BenoStreamSession::default();
        // The engine session does not read `BSDB_WAREHOUSE` itself; wire it in so
        // DDL/CTAS can derive table URIs (the Flight server does the same). Fall
        // back to a per-process temp warehouse so scratch tables always work.
        let warehouse = std::env::var("BSDB_WAREHOUSE")
            .ok()
            .filter(|w| !w.is_empty())
            .unwrap_or_else(|| {
                let dir = std::env::temp_dir().join(format!(
                    "benostreamdb_mcp_{}",
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_nanos())
                        .unwrap_or(0)
                ));
                let _ = std::fs::create_dir_all(&dir);
                dir.to_string_lossy().to_string()
            });
        session.set_warehouse(Some(warehouse));
        // Pin the GPU device for this process from BSDB_GPU_DEVICE /
        // BENOSTREAM_GPU_DEVICE (e.g. "cuda:1").
        let _ = benostreamdb::core::index::gpu::apply_gpu_context_from_env();
        Self { session }
    }

    pub async fn run(&self) -> Result<()> {
        let stdin = tokio::io::stdin();
        let mut reader = BufReader::new(stdin).lines();

        loop {
            match reader.next_line().await {
                Ok(Some(line)) => {
                    debug!("Received line: {}", line);
                    if let Ok(req) = serde_json::from_str::<JsonRpcRequest>(&line) {
                        if let Some(id) = req.id {
                            let response = self.handle_request(&req.method, req.params.unwrap_or(json!({}))).await;
                            let send_result = match response {
                                Ok(res) => self.send_response(JsonRpcResponse::success(id.clone(), res)),
                                Err(err) => self.send_response(JsonRpcResponse::error(id.clone(), -32603, err.to_string())),
                            };
                            if let Err(e) = send_result {
                                error!("Stdout closed or write failed, exiting loop: {}", e);
                                break;
                            }
                        } else {
                            // It's a notification, handle without response
                            let _ = self.handle_notification(&req.method, req.params.unwrap_or(json!({}))).await;
                        }
                    } else {
                        error!("Failed to parse JSON-RPC request");
                        if let Err(e) = self.send_response(JsonRpcResponse::error(Value::Null, -32700, "Parse error".to_string())) {
                            error!("Stdout closed or write failed, exiting loop: {}", e);
                            break;
                        }
                    }
                }
                Ok(None) => {
                    info!("Stdin closed by client, exiting event loop.");
                    break;
                }
                Err(e) => {
                    error!("Error reading from stdin, exiting: {}", e);
                    break;
                }
            }
        }
        Ok(())
    }

    async fn query_to_json(&self, query: &str) -> Result<Vec<Value>> {
        let (batches, _) = self.session.sql(query).await?;
        if batches.is_empty() {
            return Ok(Vec::new());
        }
        let mut buf = Vec::new();
        {
            let mut writer = arrow_json::LineDelimitedWriter::new(&mut buf);
            for batch in &batches {
                writer.write(batch)?;
            }
            writer.finish()?;
        }
        let text = String::from_utf8(buf)?;
        let mut rows = Vec::new();
        for line in text.lines() {
            if !line.trim().is_empty() {
                if let Ok(val) = serde_json::from_str::<Value>(line) {
                    rows.push(val);
                }
            }
        }
        Ok(rows)
    }

    async fn handle_request(&self, method: &str, params: Value) -> Result<Value> {
        info!("Handling request method: {}", method);
        match method {
            "initialize" => {
                Ok(json!({
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {
                        "name": "benostreamdb-mcp",
                        "version": "0.1.0"
                    },
                    "capabilities": {
                        "tools": {},
                        "resources": {}
                    }
                }))
            },
            "resources/list" => {
                let mut resources = Vec::new();
                if let Ok(rows) = self.query_to_json("SELECT table_schema, table_name FROM information_schema.tables WHERE table_schema != 'information_schema'").await {
                    for row in rows {
                        let schema = row.get("table_schema").and_then(|v| v.as_str()).unwrap_or("datafusion");
                        let name = row.get("table_name").and_then(|v| v.as_str()).unwrap_or("");
                        if !name.is_empty() {
                            resources.push(json!({
                                "uri": format!("benostreamdb://tables/{}/{}", schema, name),
                                "name": format!("{}.{}", schema, name),
                                "description": format!("Table {}.{}", schema, name)
                            }));
                        }
                    }
                }
                resources.push(json!({
                    "uri": "benostreamdb://docs/sql_manual.md",
                    "name": "BenoStreamDB SQL Manual",
                    "description": "Comprehensive reference manual for BenoStreamDB's SQL extensions, including Vector Search, Graph Traversal, and JSON processing."
                }));
                Ok(json!({
                    "resources": resources
                }))
            },
            "resources/read" => {
                let uri = params.get("uri").and_then(|u| u.as_str()).unwrap_or("");
                
                if uri == "benostreamdb://docs/sql_manual.md" {
                    let md_content = "# BenoStreamDB SQL Reference Manual\n\n\
                    BenoStreamDB extends standard Apache DataFusion SQL with advanced analytical capabilities designed for AI, graph, and document workloads.\n\n\
                    ## 1. Vector Search & Embeddings\n\
                    BenoStreamDB provides native first-class support for high-dimensional vectors and similarity search.\n\
                    - **Usage Example**: \n\
                      ```sql\n\
                      SELECT id, text_chunk \n\
                      FROM embeddings_table \n\
                      ORDER BY vector_distance(embedding, [0.1, 0.2, 0.3]) ASC \n\
                      LIMIT 10;\n\
                      ```\n\n\
                    ## 2. Advanced Indexing\n\
                    Indexes in BenoStreamDB act as overlay accelerators. If an index exists, the engine automatically uses it for point lookups and nearest-neighbor scans.\n\
                    - **Create an index**: `CREATE INDEX idx_name ON scratch.my_table (column) USING algorithm;`\n\
                      or `ALTER TABLE scratch.my_table ADD INDEX (column);`\n\
                    - **Supported algorithms**: `hnsw`, `hnsw_pq`, `hnsw_tq4`, `hnsw_tq8` (vector), `bm25` (lexical), `bloom`, `bitmap`, `composite_bitmap` (scalar), `csr_graph` (graph), `json_path` (JSON).\n\
                    - **Primary key** (enables merge-on-read upserts): `ALTER TABLE scratch.my_table ADD PRIMARY KEY (id);`\n\
                    - **Note**: indexes and primary keys may only be added to `scratch` tables; the main catalog is read-only.\n\n\
                    ## 3. Ephemeral (Scratch) Tables\n\
                    When prototyping or executing multi-step LLM operations, you should use ephemeral tables.\n\
                    - Scratch tables are dynamically placed in the isolated `scratch` schema.\n\
                    - Example: `SELECT * FROM scratch.my_scratch_table`\n\
                    - You can seamlessly `JOIN` an ephemeral table in the `scratch` schema with persistent tables in the `default` schema.\n\n\
                    ## 4. Graph Traversals & Topology\n\
                    BenoStreamDB supports hybrid property graph queries using standard SQL.\n\
                    - **Schema Definition**: Declare a table's role and endpoints with table metadata:\n\
                      ```sql\n\
                      ALTER TABLE edges SET TBLPROPERTIES ('table_type' = 'edge', 'src_col' = 'src', 'dst_col' = 'dst');\n\
                      ALTER TABLE nodes SET TBLPROPERTIES ('table_type' = 'node', 'id_col' = 'id', 'label_col' = 'name');\n\
                      ```\n\
                      The graph functions then resolve the endpoints from the metadata, so you never name the columns.\n\
                      Declaring an edge table also auto-configures the forward (source) and reverse (target) CSR graph indexes.\n\
                    - **Traversal**: Use the graph table functions directly in `FROM`:\n\
                      ```sql\n\
                      SELECT * FROM graph_neighbors('edges', '101', 2, 'auto');\n\
                      SELECT * FROM graph_shortest_path('edges', 101, 205, 'auto');\n\
                      ```\n\
                      or the graph UDAFs (`graph_pagerank(src, dst)`, `graph_louvain_communities(src, dst)`, …) over a `JOIN`.\n\
                    - **Discovery**: Call the `list_graph_tables` tool to enumerate node/edge tables and their endpoints.\n\n\
                    ## 5. JSON & Document Processing\n\
                    For schemaless data, JSON paths can be traversed natively.\n\
                    - **Function**: `json_extract_path(json_column, '$.path.to.field')`\n\
                    - **Example**: `SELECT json_extract_path(metadata, '$.author.name') FROM documents;`\n\n\
                    ## 6. Complete Custom Function Signatures\n\
                    \n\
                    ### Vector Distance Functions (Dense & Sparse)\n\
                    - `vector_distance(vec1, vec2)` / `sparse_l2_distance`: Computes L2 (Euclidean) distance.\n\
                    - `vector_cosine_distance(vec1, vec2)` / `sparse_cosine_distance`: Computes Cosine distance.\n\
                    - `vector_inner_product(vec1, vec2)` / `sparse_inner_product_distance`: Computes Inner Product.\n\
                    - `vector_l1_distance(vec1, vec2)`: Computes L1 (Manhattan) distance.\n\
                    - `vector_hamming_distance(vec1, vec2)`: Computes Hamming distance.\n\
                    - `vector_jaccard_distance(vec1, vec2)`: Computes Jaccard distance.\n\
                    \n\
                    ### Vector Math & Quantization\n\
                    - `vector_add`, `vector_sub`, `vector_mul`: Element-wise arithmetic.\n\
                    - `vector_norm(vec)`, `vector_normalize(vec)`, `vector_dims(vec)`: Vector topology.\n\
                    - `binary_quantize(vec)`, `vector_to_binary(vec)`: Quantization to bit-vectors.\n\
                    - `subvector(vec, offset, len)`: Slice a vector.\n\
                    - `dense_to_sparse(vec)`, `sparse_to_dense(vec)`, `sparsevec_nnz(vec)`: Sparse operations.\n\
                    \n\
                    ### Vector Aggregations & Stats (UDAFs)\n\
                    - `vector_avg(vec)`, `vector_sum(vec)`: Aggregate math.\n\
                    - `centroid(vec)`: Computes the geometric center of vectors.\n\
                    - `vector_max(vec)`, `vector_min(vec)`, `vector_median(vec)`, `vector_stddev(vec)`: Aggregate stats.\n\
                    \n\
                    ### JSON Functions\n\
                    - `json_contains(json, target)`: Checks if the JSON document contains the target.\n\
                    - `json_exists(json, path)`: Checks if a top-level key/element exists.\n\
                    - `json_extract_path(json, path)`: Extracts JSON sub-element at the specified path.\n\
                    - `json_path_exists(json, jsonpath)`: Evaluates if a subset matches the jsonpath.\n\
                    - `json_path_query(json, jsonpath)`: Extracts all elements matching the jsonpath.\n\
                    - `json_typeof(json)`: Returns the type of the JSON value as a string.\n\
                    \n\
                    ### Text & Lexical Functions\n\
                    - `bm25_score(text, query)`: Computes BM25 relevance score.\n\
                    - `tf_idf(text, query)`: Computes TF-IDF relevance score.\n\
                    \n\
                    ### Graph Analysis & Traversals (UDAFs)\n\
                    - **Centrality**:\n\
                      - `graph_pagerank(src, dst)`: Computes PageRank.\n\
                      - `graph_personalized_pagerank(src, dst, node)`: Computes Personalized PageRank.\n\
                      - `graph_degree_centrality(src, dst)`: Computes degree centrality.\n\
                      - `graph_closeness_centrality(src, dst)`: Computes closeness centrality.\n\
                      - `graph_betweenness_centrality(src, dst)`: Computes betweenness centrality.\n\
                    - **Communities**:\n\
                      - `graph_connected_components(src, dst)`: Weakly connected components.\n\
                      - `graph_strongly_connected_components(src, dst)`: Strongly connected components.\n\
                      - `graph_louvain_communities(src, dst)`: Louvain community detection.\n\
                      - `graph_leiden_communities(src, dst)`: Leiden community detection.\n\
                      - `graph_label_propagation(src, dst)`: Label propagation.\n\
                      - `graph_modularity(src, dst, community_id)`: Graph modularity.\n\
                    - **Paths & Topology**:\n\
                      - `graph_shortest_path(src, dst, start, end)`: Shortest path between two nodes.\n\
                      - `graph_all_shortest_paths(src, dst, start, end)`: All shortest paths.\n\
                      - `graph_connecting_paths(src, dst, start, end)`: Connecting paths.\n\
                      - `graph_triangle_count(src, dst)`: Triangle counting.\n\
                      - `graph_preferential_attachment(src, dst)`: Preferential attachment.\n\
                      - `graph_jaccard_coefficient(src, dst)`: Jaccard coefficient.\n\
                    - **Extraction**:\n\
                      - `graph_neighbors(src, dst, node)`: Find all neighbors of a node.\n\
                      - `graph_subgraph(src, dst, node)`: Extract subgraph around a node.\n\
                    \n\
                    ## 7. Live Subscriptions (Change Feed)\n\
                    Subscribe to a table's committed changes (in-process).\n\
                    - **SQL**: `SELECT * FROM subscribe_events('scratch.my_table', 'id > 10', 100, 1000);` returns one row per event (`event_type` = 'batch' | 'commit', `rows`).\n\
                    - **Tool**: call the `subscribe_events` tool with `{ table, filter?, max_events?, timeout_ms? }`.\n\
                    - **Note**: the change feed is in-process; it only observes commits made by writers in the same process.";
                    
                    return Ok(json!({
                        "contents": [{
                            "uri": uri,
                            "mimeType": "text/markdown",
                            "text": md_content
                        }]
                    }));
                }
                
                if uri.starts_with("benostreamdb://tables/") {
                    let parts: Vec<&str> = uri.trim_start_matches("benostreamdb://tables/").split('/').collect();
                    if parts.len() == 2 {
                        let schema = parts[0];
                        let table = parts[1];
                        let query = format!("DESCRIBE {}.{}", schema, table);
                        match self.session.sql(&query).await {
                            Ok((batches, _)) => {
                                let mut buf = Vec::new();
                                {
                                    let mut writer = arrow_json::LineDelimitedWriter::new(&mut buf);
                                    for batch in &batches {
                                        let _ = writer.write(batch);
                                    }
                                    let _ = writer.finish();
                                }
                                let text_output = String::from_utf8_lossy(&buf).to_string();
                                return Ok(json!({
                                    "contents": [
                                        {
                                            "uri": uri,
                                            "mimeType": "application/json",
                                            "text": text_output
                                        }
                                    ]
                                }));
                            }
                            Err(e) => {
                                return Err(anyhow::anyhow!("Failed to read resource: {}", e));
                            }
                        }
                    }
                }
                Err(anyhow::anyhow!("Resource not found: {}", uri))
            },
            "tools/list" => {
                let execute_sql_desc = "Execute a SQL query against the BenoStreamDB engine. Uses Apache DataFusion SQL dialect.\n\n\
                Available custom functions and syntax:\n\
                - Vector Search: Use `l2_distance(embedding, [1.0, 2.0, 3.0])` / `cosine_distance(...)` in SELECT or ORDER BY clauses.\n\
                - Indexes: Create indexes using `CREATE INDEX idx_name ON table_name (col) USING hnsw` (Supports: hnsw, hnsw_tq8, hnsw_tq4, bm25, bitmap).\n\
                - Table Functions: Query graph walks with `SELECT * FROM graph_neighbors('edges', '101', 2, 'auto')`.\n\
                - JSON Path: Extract JSON fields natively using `json_extract_path(metadata, '$.path')`.\n\
                - Graph Queries: Declare node/edge tables with `ALTER TABLE t SET TBLPROPERTIES ('table_type'='edge', 'src_col'='src', 'dst_col'='dst')`, then use the graph functions.\n\
                - Ephemeral Tables: Scratch tables are placed in the `scratch` schema (e.g. `scratch.my_table`).\n\
                - Discovery: Query `information_schema.tables` and `information_schema.columns`, or call the `list_graph_tables` tool.\n\n\
                NOTE: For full syntax examples, read the `benostreamdb://docs/sql_manual.md` resource!";

                Ok(json!({
                    "tools": [
                        {
                            "name": "execute_sql",
                            "description": execute_sql_desc,
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "query": { "type": "string" }
                                },
                                "required": ["query"]
                            }
                        },
                        {
                            "name": "create_scratch_table",
                            "description": "Creates a temporary, ephemeral table in the local scratch catalog.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "table_name": { "type": "string" },
                                    "schema_ddl": { "type": "string", "description": "The column definitions, e.g., 'id INT, name VARCHAR'" },
                                    "table_type": { "type": "string", "description": "Optional type, e.g. 'node' or 'edge'" }
                                },
                                "required": ["table_name", "schema_ddl"]
                            }
                        },
                        {
                            "name": "insert_scratch_records",
                            "description": "Appends JSON rows to an ephemeral scratch table.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "table_name": { "type": "string" },
                                    "records": { "type": "array", "items": { "type": "object" } }
                                },
                                "required": ["table_name", "records"]
                            }
                        },
                        {
                            "name": "export_table",
                            "description": "Exports any table (scratch or main catalog) to local disk as Parquet.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "table_name": { "type": "string" },
                                    "export_path": { "type": "string" }
                                },
                                "required": ["table_name", "export_path"]
                            }
                        },
                        {
                            "name": "list_graph_tables",
                            "description": "List the graph tables (node/edge) and their endpoint columns. Use this to discover the graph before traversing it.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {}
                            }
                        },
                        {
                            "name": "subscribe_events",
                            "description": "Drain the next committed-change events for a table (a bounded live tail). Returns one row per event: event_type ('batch'|'commit') and rows. Only observes commits made in this process.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "table": { "type": "string" },
                                    "filter": { "type": "string", "description": "Optional SQL predicate; only matching rows are counted." },
                                    "max_events": { "type": "integer", "description": "Max events to drain (default 100)." },
                                    "timeout_ms": { "type": "integer", "description": "Wait budget in ms (default 1000)." }
                                },
                                "required": ["table"]
                            }
                        }
                    ]
                }))
            },
            "tools/call" => {
                let name = params.get("name").and_then(|n| n.as_str()).unwrap_or("");
                
                match name {
                    "execute_sql" => {
                        let query = params.get("arguments").and_then(|a| a.get("query")).and_then(|q| q.as_str()).unwrap_or("");
                        if let Err(e) = guard_scratch_only(query) {
                            return Ok(json!({
                                "content": [ { "type": "text", "text": e.to_string() } ],
                                "isError": true
                            }));
                        }
                        match self.session.sql(query).await {
                            Ok((batches, _schema)) => {
                                let mut text_output = String::new();
                                if !batches.is_empty() {
                                    let mut buf = Vec::new();
                                    {
                                        let mut writer = arrow_json::LineDelimitedWriter::new(&mut buf);
                                        for batch in &batches {
                                            writer.write(batch).unwrap();
                                        }
                                        writer.finish().unwrap();
                                    }
                                    text_output = String::from_utf8(buf).unwrap();
                                }
                                if text_output.is_empty() {
                                    text_output = "Success (no rows returned).".to_string();
                                }
                                Ok(json!({
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": text_output
                                        }
                                    ]
                                }))
                            }
                            Err(e) => {
                                Ok(json!({
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": format!("Error executing SQL: {}", e)
                                        }
                                    ],
                                    "isError": true
                                }))
                            }
                        }
                    }
                    "create_scratch_table" => {
                        let table_name = params.get("arguments").and_then(|a| a.get("table_name")).and_then(|v| v.as_str()).unwrap_or("");
                        let schema_ddl = params.get("arguments").and_then(|a| a.get("schema_ddl")).and_then(|v| v.as_str()).unwrap_or("");
                        let table_type = params.get("arguments").and_then(|a| a.get("table_type")).and_then(|v| v.as_str()).unwrap_or("");

                        let _ = self
                            .session
                            .sql(&format!("CREATE SCHEMA IF NOT EXISTS {}", SCRATCH_SCHEMA))
                            .await;

                        let full_table_name = if table_name.contains('.') {
                            let parts: Vec<&str> = table_name.split('.').collect();
                            let schema = parts
                                .get(parts.len().saturating_sub(2))
                                .copied()
                                .unwrap_or("");
                            if schema != SCRATCH_SCHEMA {
                                return Ok(json!({
                                    "content": [ { "type": "text", "text": format!(
                                        "Scratch tables must be created in the '{}' schema (got '{}').",
                                        SCRATCH_SCHEMA, table_name
                                    ) } ],
                                    "isError": true
                                }));
                            }
                            table_name.to_string()
                        } else {
                            format!("{}.{}", SCRATCH_SCHEMA, table_name)
                        };

                        // A real BenoStream table (not a DataFusion listing table),
                        // so it carries table metadata and supports INSERT. It lives
                        // in a local scratch dir, not the warehouse.
                        let scratch_dir = std::env::temp_dir().join(format!(
                            "benostreamdb_scratch_{}_{}",
                            table_name,
                            std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .map(|d| d.as_nanos())
                                .unwrap_or(0)
                        ));
                        std::fs::create_dir_all(&scratch_dir).unwrap();
                        let scratch_path = scratch_dir.to_str().unwrap().replace("\\", "/");

                        let query = format!(
                            "CREATE TABLE {} ({}) LOCATION '{}'",
                            full_table_name, schema_ddl, scratch_path
                        );
                        match self.session.sql(&query).await {
                            Ok(_) => {
                                // Declare the graph role via table metadata (the
                                // engine ignores `WITH (type = ...)`).
                                if !table_type.is_empty() {
                                    let alter = format!(
                                        "ALTER TABLE {} SET TBLPROPERTIES ('table_type' = '{}')",
                                        full_table_name, table_type
                                    );
                                    if let Err(e) = self.session.sql(&alter).await {
                                        error!("Failed to set table_type on {}: {}", full_table_name, e);
                                    }
                                }
                                Ok(json!({
                                    "content": [
                                        { "type": "text", "text": format!("Table {} created successfully.", full_table_name) }
                                    ]
                                }))
                            }
                            Err(e) => {
                                Ok(json!({
                                    "content": [
                                        { "type": "text", "text": format!("Error creating table: {}", e) }
                                    ],
                                    "isError": true
                                }))
                            }
                        }
                    }
                    "insert_scratch_records" => {
                        let table_name = params.get("arguments").and_then(|a| a.get("table_name")).and_then(|v| v.as_str()).unwrap_or("");
                        let records = params.get("arguments").and_then(|a| a.get("records")).and_then(|v| v.as_array());
                        
                        if let Some(records) = records {
                            if records.is_empty() {
                                return Ok(json!({"content": [{"type": "text", "text": "No records to insert."}]}));
                            }
                            // Write records to a temporary NDJSON file for DataFusion insertion
                            use std::io::Write;
                            match tempfile::Builder::new().suffix(".json").tempfile() {
                                Ok(temp_file) => {
                                    let mut writer = std::io::BufWriter::new(temp_file.as_file());
                                    for rec in records {
                                        let _ = serde_json::to_writer(&mut writer, rec);
                                        let _ = writer.write_all(b"\n");
                                    }
                                    let _ = writer.flush();
                                    
                                    let path = temp_file.path().to_str().unwrap().replace("\\", "/");
                                    let temp_import_name = format!("{}.temp_import_{}", SCRATCH_SCHEMA, std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos());

                                    let full_table_name = if table_name.contains('.') {
                                        table_name.to_string()
                                    } else {
                                        format!("{}.{}", SCRATCH_SCHEMA, table_name)
                                    };
                                    
                                    let query1 = format!("CREATE EXTERNAL TABLE {} STORED AS JSON LOCATION '{}'", temp_import_name, path);
                                    let query2 = format!("INSERT INTO {} SELECT * FROM {}", full_table_name, temp_import_name);
                                    let query3 = format!("DROP TABLE {}", temp_import_name);
                                    
                                    let mut success = false;
                                    let mut error_msg = String::new();
                                    match self.session.sql(&query1).await {
                                        Ok(_) => {
                                            match self.session.sql(&query2).await {
                                                Ok(_) => success = true,
                                                Err(e) => error_msg = format!("INSERT error: {}", e),
                                            }
                                            let _ = self.session.sql(&query3).await;
                                        }
                                        Err(e) => error_msg = format!("CREATE EXTERNAL TABLE error: {}", e),
                                    }
                                    
                                    if success {
                                        Ok(json!({
                                            "content": [
                                                { "type": "text", "text": format!("Inserted {} records into {}.", records.len(), table_name) }
                                            ]
                                        }))
                                    } else {
                                        Ok(json!({
                                            "content": [
                                                { "type": "text", "text": format!("Error inserting records into {}: {}", table_name, error_msg) }
                                            ],
                                            "isError": true
                                        }))
                                    }
                                }
                                Err(e) => {
                                    Ok(json!({"content": [{"type": "text", "text": format!("Failed to create temp file: {}", e)}], "isError": true}))
                                }
                            }
                        } else {
                            Ok(json!({"content": [{"type": "text", "text": "Missing 'records' array."}], "isError": true}))
                        }
                    }
                    "export_table" => {
                        let table_name = params.get("arguments").and_then(|a| a.get("table_name")).and_then(|v| v.as_str()).unwrap_or("");
                        let export_path = params.get("arguments").and_then(|a| a.get("export_path")).and_then(|v| v.as_str()).unwrap_or("");
                        
                        let full_table_name = if table_name.contains('.') {
                            table_name.to_string()
                        } else {
                            format!("{}.{}", SCRATCH_SCHEMA, table_name)
                        };

                        let query = format!("COPY {} TO '{}' STORED AS PARQUET", full_table_name, export_path.replace("\\", "/"));
                        match self.session.sql(&query).await {
                            Ok(_) => {
                                Ok(json!({
                                    "content": [
                                        { "type": "text", "text": format!("Successfully exported {} to {}", table_name, export_path) }
                                    ]
                                }))
                            }
                            Err(e) => {
                                Ok(json!({
                                    "content": [
                                        { "type": "text", "text": format!("Error exporting table: {}", e) }
                                    ],
                                    "isError": true
                                }))
                            }
                        }
                    }
                    "list_graph_tables" => {
                        match self.session.list_graph_tables().await {
                            Ok(infos) => {
                                let text = if infos.is_empty() {
                                    "No graph tables found. Declare one with \
                                     ALTER TABLE t SET TBLPROPERTIES ('table_type'='edge', 'src_col'='src', 'dst_col'='dst')."
                                        .to_string()
                                } else {
                                    serde_json::to_string_pretty(&infos).unwrap_or_default()
                                };
                                Ok(json!({
                                    "content": [ { "type": "text", "text": text } ]
                                }))
                            }
                            Err(e) => Ok(json!({
                                "content": [ { "type": "text", "text": format!("Error listing graph tables: {}", e) } ],
                                "isError": true
                            })),
                        }
                    }
                    "subscribe_events" => {
                        let table = params.get("arguments").and_then(|a| a.get("table")).and_then(|v| v.as_str()).unwrap_or("");
                        let filter = params.get("arguments").and_then(|a| a.get("filter")).and_then(|v| v.as_str()).unwrap_or("");
                        let max_events = params.get("arguments").and_then(|a| a.get("max_events")).and_then(|v| v.as_u64()).unwrap_or(100);
                        let timeout_ms = params.get("arguments").and_then(|a| a.get("timeout_ms")).and_then(|v| v.as_u64()).unwrap_or(1000);
                        if table.is_empty() {
                            return Ok(json!({
                                "content": [ { "type": "text", "text": "subscribe_events requires a 'table' argument." } ],
                                "isError": true
                            }));
                        }
                        let esc = |s: &str| s.replace('\'', "''");
                        let query = format!(
                            "SELECT * FROM subscribe_events('{}', '{}', {}, {})",
                            esc(table), esc(filter), max_events, timeout_ms
                        );
                        match self.session.sql(&query).await {
                            Ok((batches, _schema)) => {
                                let mut text_output = String::new();
                                if !batches.is_empty() {
                                    let mut buf = Vec::new();
                                    {
                                        let mut writer = arrow_json::LineDelimitedWriter::new(&mut buf);
                                        for batch in &batches {
                                            writer.write(batch).unwrap();
                                        }
                                        writer.finish().unwrap();
                                    }
                                    text_output = String::from_utf8(buf).unwrap();
                                }
                                if text_output.is_empty() {
                                    text_output = "No events (nothing committed in this process during the wait window).".to_string();
                                }
                                Ok(json!({
                                    "content": [ { "type": "text", "text": text_output } ]
                                }))
                            }
                            Err(e) => Ok(json!({
                                "content": [ { "type": "text", "text": format!("Error draining events: {}", e) } ],
                                "isError": true
                            })),
                        }
                    }
                    _ => Err(anyhow::anyhow!("Unknown tool: {}", name)),
                }
            },
            _ => {
                Err(anyhow::anyhow!("Method not found: {}", method))
            }
        }
    }

    async fn handle_notification(&self, method: &str, _params: Value) -> Result<()> {
        info!("Handling notification method: {}", method);
        Ok(())
    }

    fn send_response(&self, response: JsonRpcResponse) -> Result<()> {
        if let Ok(json) = serde_json::to_string(&response) {
            let mut stdout = std::io::stdout();
            writeln!(stdout, "{}", json)?;
            stdout.flush()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_mcp_initialize() {
        let server = McpServer::new();
        let res = server.handle_request("initialize", json!({})).await.unwrap();
        assert_eq!(res["protocolVersion"], "2024-11-05");
        assert_eq!(res["serverInfo"]["name"], "benostreamdb-mcp");
    }

    #[tokio::test]
    async fn test_mcp_tools_list() {
        let server = McpServer::new();
        let res = server.handle_request("tools/list", json!({})).await.unwrap();
        let tools = res["tools"].as_array().unwrap();
        assert!(tools.iter().any(|t| t["name"] == "execute_sql"));
        assert!(tools.iter().any(|t| t["name"] == "create_scratch_table"));
        assert!(tools.iter().any(|t| t["name"] == "insert_scratch_records"));
        assert!(tools.iter().any(|t| t["name"] == "export_table"));
    }

    #[tokio::test]
    async fn test_mcp_execute_sql() {
        let server = McpServer::new();
        let res = server.handle_request("tools/call", json!({
            "name": "execute_sql",
            "arguments": {
                "query": "SELECT 1 as x"
            }
        })).await.unwrap();
        
        let content = res["content"][0]["text"].as_str().unwrap();
        assert!(content.contains("\"x\":1") || content.contains("\"x\": 1"));
    }

    #[tokio::test]
    async fn test_mcp_scratch_table_workflow() {
        let server = McpServer::new();
        
        // 1. Create table
        let res = server.handle_request("tools/call", json!({
            "name": "create_scratch_table",
            "arguments": {
                "table_name": "test_table",
                "schema_ddl": "id INT, val VARCHAR"
            }
        })).await.unwrap();
        assert!(res.get("isError").is_none());
        
        // 2. Insert records
        let res = server.handle_request("tools/call", json!({
            "name": "insert_scratch_records",
            "arguments": {
                "table_name": "test_table",
                "records": [
                    {"id": 1, "val": "hello"},
                    {"id": 2, "val": "world"}
                ]
            }
        })).await.unwrap();
        assert!(res.get("isError").is_none());
        
        // 3. Select back
        let res = server.handle_request("tools/call", json!({
            "name": "execute_sql",
            "arguments": {
                "query": "SELECT * FROM scratch.test_table ORDER BY id"
            }
        })).await.unwrap();
        let content = res["content"][0]["text"].as_str().unwrap();
        assert!(content.contains("hello"));
        assert!(content.contains("world"));
    }
    
    #[tokio::test]
    async fn test_mcp_export_workflow() {
        let server = McpServer::new();
        
        server.handle_request("tools/call", json!({
            "name": "create_scratch_table",
            "arguments": { "table_name": "export_test", "schema_ddl": "id INT" }
        })).await.unwrap();
        
        server.handle_request("tools/call", json!({
            "name": "insert_scratch_records",
            "arguments": { "table_name": "export_test", "records": [{"id": 42}] }
        })).await.unwrap();
        
        let temp_dir = std::env::temp_dir();
        let export_path = temp_dir.join(format!("export_test_{}.parquet", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        let _ = std::fs::remove_dir_all(&export_path);
        let _ = std::fs::remove_file(&export_path);
        
        let res = server.handle_request("tools/call", json!({
            "name": "export_table",
            "arguments": {
                "table_name": "export_test",
                "export_path": export_path.to_str().unwrap()
            }
        })).await.unwrap();
        assert!(res.get("isError").is_none());
        assert!(export_path.exists());
    }
    
    #[tokio::test]
    async fn test_mcp_scratch_only_guard() {
        let server = McpServer::new();
        // Writes to the main catalog are rejected.
        for q in [
            "CREATE TABLE default.evil (id INT)",
            "INSERT INTO default.evil VALUES (1)",
            "UPDATE default.evil SET id = 2",
            "DELETE FROM default.evil",
            "DROP TABLE default.evil",
            "CREATE INDEX idx ON default.evil (id)",
            "ALTER TABLE default.evil ADD INDEX (id)",
        ] {
            let res = server
                .handle_request(
                    "tools/call",
                    json!({ "name": "execute_sql", "arguments": { "query": q } }),
                )
                .await
                .unwrap();
            assert_eq!(res["isError"], true, "expected rejection for: {q}");
        }
        // Scratch writes (including index creation) are allowed.
        for q in [
            "CREATE TABLE scratch.ok (id INT)",
            "CREATE INDEX idx ON scratch.ok (id)",
        ] {
            let res = server
                .handle_request(
                    "tools/call",
                    json!({ "name": "execute_sql", "arguments": { "query": q } }),
                )
                .await
                .unwrap();
            assert!(res.get("isError").is_none(), "expected success for {q}: {res}");
        }
    }

    #[tokio::test]
    async fn test_mcp_list_graph_tables() {
        let server = McpServer::new();
        // Declare an edge table via the scratch tool (table_type metadata).
        server
            .handle_request(
                "tools/call",
                json!({
                    "name": "create_scratch_table",
                    "arguments": { "table_name": "g_edges", "schema_ddl": "src INT, dst INT", "table_type": "edge" }
                }),
            )
            .await
            .unwrap();
        let res = server
            .handle_request("tools/call", json!({ "name": "list_graph_tables", "arguments": {} }))
            .await
            .unwrap();
        let text = res["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("g_edges"), "expected g_edges in: {text}");
        assert!(text.contains("edge"), "expected table_type edge in: {text}");
    }

    #[tokio::test]
    async fn test_mcp_subscribe_events_tool() {
        let server = McpServer::new();
        // The tool must be advertised.
        let tools = server
            .handle_request("tools/list", json!({}))
            .await
            .unwrap();
        let names: Vec<&str> = tools["tools"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|t| t["name"].as_str())
            .collect();
        assert!(names.contains(&"subscribe_events"), "tools: {names:?}");

        // Create a scratch table, then drain its (empty) change feed.
        server
            .handle_request(
                "tools/call",
                json!({
                    "name": "create_scratch_table",
                    "arguments": { "table_name": "sub_t", "schema_ddl": "id INT" }
                }),
            )
            .await
            .unwrap();
        let res = server
            .handle_request(
                "tools/call",
                json!({
                    "name": "subscribe_events",
                    "arguments": { "table": "scratch.sub_t", "max_events": 5, "timeout_ms": 50 }
                }),
            )
            .await
            .unwrap();
        assert!(
            res.get("isError").is_none(),
            "subscribe_events errored: {res}"
        );
    }

    #[tokio::test]
    async fn test_mcp_resources() {
        let server = McpServer::new();
        
        server.handle_request("tools/call", json!({
            "name": "create_scratch_table",
            "arguments": { "table_name": "resource_test", "schema_ddl": "id INT" }
        })).await.unwrap();
        
        let res = server.handle_request("resources/list", json!({})).await.unwrap();
        let resources = res["resources"].as_array().unwrap();
        println!("Resources: {:?}", resources);
        assert!(resources.iter().any(|r| r["name"].as_str().unwrap().contains("resource_test")));
        
        let res = server.handle_request("resources/read", json!({
            "uri": "benostreamdb://tables/scratch/resource_test"
        })).await.unwrap();
        
        let content = res["contents"][0]["text"].as_str().unwrap();
        assert!(content.contains("id"));
    }
}
