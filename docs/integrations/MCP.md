# Model Context Protocol (MCP) Server (`benostreamdb-mcp`)

`benostreamdb-mcp` is an AI agent tool server implementing the open Model Context Protocol (MCP) over JSON-RPC stdio.

---

## Overview

The MCP server allows LLMs and autonomous AI agents (such as Claude Desktop, Cursor, and Antigravity) to interact directly with BenoStreamDB lakehouse tables without bespoke API integrations:
* **Discover Tables**: Introspect available datasets, node tables, and edge tables.
* **Inspect Schemas**: Read Arrow and Iceberg table schemas as MCP resources.
* **Execute SQL**: Run analytical queries, pgvector distance operations, and graph traversals.
* **Stream Events**: Subscribe to live table change feeds.

---

---

## Installation & Running

### Recommended: Run via `uvx` (No Installation Required)

You can launch `benostreamdb-mcp` instantly with [`uvx`](https://github.com/astral-sh/uv), which automatically downloads the prebuilt native binary from PyPI:

```bash
uvx benostreamdb-mcp
```

### Alternative: Install via Pip

```bash
pip install benostreamdb-mcp
benostreamdb-mcp
```

### From Source via Cargo

```bash
cargo build --release -p benostreamdb-mcp
```
The compiled binary will be placed at `target/release/benostreamdb-mcp`.

---

## Configuration with AI Clients

Add the server to your MCP client configuration (e.g., `claude_desktop_config.json`, Cursor, or Antigravity):

### With `uvx` (Recommended)

```json
{
  "mcpServers": {
    "benostreamdb": {
      "command": "uvx",
      "args": ["benostreamdb-mcp"],
      "env": {
        "BSDB_WAREHOUSE": "/path/to/warehouse",
        "BSDB_GPU_DEVICE": "auto"
      }
    }
  }
}
```

### With Prebuilt Binary / Source Build

```json
{
  "mcpServers": {
    "benostreamdb": {
      "command": "/path/to/target/release/benostreamdb-mcp",
      "env": {
        "BSDB_WAREHOUSE": "/path/to/warehouse",
        "BSDB_GPU_DEVICE": "auto"
      }
    }
  }
}
```

### Remote transport: HTTP + SSE

`stdio` is the default and is what a local IDE/agent spawn uses. For a
cloud-hosted agent that cannot spawn a subprocess, run the HTTP + SSE transport
(MCP spec 2024-11-05):

```bash
benostreamdb-mcp --transport sse --bind 0.0.0.0:8080
# or: BSDB_MCP_TRANSPORT=sse BSDB_MCP_BIND=0.0.0.0:8080 benostreamdb-mcp
```

| Endpoint | Method | Purpose |
|:---|:---|:---|
| `/sse` | `GET` | Opens the event stream. The first event is `endpoint` and carries `/message?sessionId=...`. |
| `/message?sessionId=...` | `POST` | Accepts one JSON-RPC message. Replies `202 Accepted`; the JSON-RPC response is delivered on that session's `/sse` stream. Unknown sessions return `404`. |

Both transports share the same request handler, so tools, resources, and
notifications behave identically. Sessions are dropped when their stream closes
(the session map is pruned on each new connection).

> The SSE transport has no authentication. Bind it to loopback, or put it behind
> an authenticating proxy, before exposing it beyond the host.

### Environment Variables

| Variable | Description | Default |
|:---|:---|:---|
| `BSDB_WAREHOUSE` | Base directory or URI for tables | `./data` |
| `BSDB_GPU_DEVICE` | Compute target or pool (`auto`, `cpu`, `cuda[:N]`, `mps`, `cuda:0,cuda:1`) | `auto` |
| `BSDB_CONFIG` | Catalog configuration file path | Unset |
| `BSDB_MCP_TRANSPORT` | `stdio` or `sse` (flag `--transport` wins) | `stdio` |
| `BSDB_MCP_BIND` | Listen address for `sse` (flag `--bind` wins) | `127.0.0.1:8080` |

---

## Exposed MCP Tools

The agent can invoke the following tools:

| Tool | Purpose | Parameters |
|:---|:---|:---|
| `list_graph_tables` | List all discovered tables, edge tables, and node tables | `{}` |
| `execute_sql` | Execute a SQL query (SELECT, vector distance, graph traversal) | `{"query": "SELECT ..."}` |
| `subscribe_events` | Drain change-feed events for a table | `{"table": "edges", "filter": "...", "max_events": 100}` |

### Resources

Tables are exposed as read-only resources:
* URI format: `benostreamdb://tables/{schema}/{table}`
* Content: JSON schema representation including field types, partitions, and index coverage.

---

## Example Agent Workflow

An AI agent uses the MCP server to solve retrieval tasks in a structured sequence:
1. **Discover**: Calls `list_graph_tables` to identify available tables.
2. **Schema**: Reads `benostreamdb://tables/default/documents` to understand available columns and vector dimensions.
3. **Retrieve**: Calls `execute_sql` with pgvector similarity operators (`<->`, `<=>`) or graph table functions (`graph_neighbors`).
4. **Synthesize**: Answers the user request using the grounded result set.
