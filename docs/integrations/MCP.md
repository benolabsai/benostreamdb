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

### Environment Variables

| Variable | Description | Default |
|:---|:---|:---|
| `BSDB_WAREHOUSE` | Base directory or URI for tables | `./data` |
| `BSDB_GPU_DEVICE` | Compute target (`auto`, `cpu`, `cuda`, `mps`) | `auto` |
| `BSDB_CONFIG` | Catalog configuration file path | Unset |

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
