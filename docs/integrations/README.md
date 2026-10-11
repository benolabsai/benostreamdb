# Connectors & Ecosystem Integrations

BenoStreamDB provides native connectors, API bindings, and network gateways across the data engineering and AI ecosystems.

---

## Available Connectors & Interfaces

| Integration | Type | Primary Use Case | Guide |
|:---|:---|:---|:---|
| **Python Bindings** | In-Process Library | Data science, PyArrow/Pandas pipelines, interactive notebooks, and embedding workflows | [Python Guide](PYTHON.md) |
| **Apache Spark** | DataSource V2 Plugin | Large-scale batch ETL, structured streaming, catalog procedures, and distributed vector scoring | [Spark Guide](SPARK.md) |
| **Trino** | Trino SPI Plugin | Interactive distributed SQL analytics, index-accelerated predicate pushdown, and graph traversal | [Trino Guide](TRINO.md) |
| **dbt Adapter** | dbt Plugin (`dbt-benostreamdb`) | Analytics engineering, transformation pipelines, and incremental Iceberg model materializations | [dbt Guide](DBT.md) |
| **Arrow Flight SQL** | gRPC Network Gateway | Standard database endpoint for polyglot clients (Go, C++, JDBC/ODBC), DBeaver, and BI tools | [Flight SQL Guide](FLIGHT_SQL.md) |
| **Search Gateway** | HTTP REST Gateway | Drop-in REST compatibility for OpenSearch / ES 7.10 (port 9200) and Qdrant v1.x (port 6333) | [Search Gateway Guide](SEARCH_GATEWAY.md) |
| **Model Context Protocol** | AI Agent Tool Server | JSON-RPC tool server for LLMs and autonomous agents (Claude Desktop, Cursor, Antigravity) | [MCP Guide](MCP.md) |
| **Java JNI Bridge** | JVM Native Interface | High-performance Arrow C Data Interface bridge powering JVM query engines | [Java JNI Guide](JAVA_JNI.md) |

---

## Architectural Principles

1. **Serverless by Default**: Applications with native bindings (Python, Rust, JVM) embed the engine in-process with zero operational overhead, operating directly on object storage.
2. **Network Optional**: Remote server interfaces (Arrow Flight SQL, REST Search Gateway) provide single-process endpoints when network access is required, without imposing a mandatory cluster runtime.
3. **Format Compatibility**: Tables are stored as standard Apache Iceberg datasets. Any external tool that reads Parquet or Iceberg can query the underlying data files without lock-in.
