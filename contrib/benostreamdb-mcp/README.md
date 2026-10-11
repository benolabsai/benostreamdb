# BenoStreamDB MCP Server

`benostreamdb-mcp` is an AI agent tool server implementing the open [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) for **BenoStreamDB**.

It exposes analytical querying, vector similarity search, schema inspection, and live change feeds directly to autonomous AI agents and IDEs (such as Claude Desktop, Cursor, and Antigravity) over standard input/output (stdio JSON-RPC).

---

## Quick Start with `uvx`

You can run `benostreamdb-mcp` instantly without any manual compilation or installation using [`uv`](https://github.com/astral-sh/uv):

```bash
uvx benostreamdb-mcp
```

### Configuring Claude Desktop / Cursor / Antigravity

Add the following to your MCP client configuration (e.g. `claude_desktop_config.json`):

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

---

## Alternative Installation Methods

### Via Pip

```bash
pip install benostreamdb-mcp
benostreamdb-mcp
```

### From Source (Rust)

```bash
cargo install --path .
# Or run directly from the repository root:
cargo run --release -p benostreamdb-mcp
```

---

## Configuration & Environment Variables

| Variable | Description | Default |
|:---|:---|:---|
| `BSDB_WAREHOUSE` | Base directory or URI for tables | System temporary directory |
| `BSDB_GPU_DEVICE` | Hardware accelerator target (`auto`, `cpu`, `cuda`, `mps`) | `auto` |
| `BSDB_CONFIG` | Catalog configuration file path | None |

---

## Exposed MCP Tools

- `list_graph_tables`: Discover all analytical tables, node tables, and edge tables.
- `execute_sql`: Run analytical SQL queries, pgvector distance calculations, and graph traversals.
- `resources/read`: Inspect Arrow and Iceberg table schemas at `benostreamdb://tables/{namespace}/{table}`.
- `subscribe_table`: Subscribe to real-time change data capture (CDC) streams.
