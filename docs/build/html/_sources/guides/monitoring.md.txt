# HyperStreamDB Monitoring & Operations

HyperStreamDB is designed to be highly observable and easy to operate in serverless, ephemeral environments as well as traditional long-running daemon setups.

## Telemetry & Tracing

HyperStreamDB leverages a push-based OpenTelemetry (OTLP) pipeline to export distributed traces. Because it often runs in serverless functions (like AWS Lambda) where instances may be frozen between invocations, the `Table` API explicitly flushes traces on teardown (via the `Drop` trait) to ensure no data is lost.

### Configuration

Tracing is disabled by default. To enable tracing, set the `JAEGER_ENABLED` environment variable. When enabled, traces are automatically pushed to an OTLP-compatible endpoint.

| Environment Variable | Description | Default |
|----------------------|-------------|---------|
| `JAEGER_ENABLED` | Set to `true` to enable OpenTelemetry exporting. | `false` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | The destination URL for OTLP traces. | `http://localhost:4317` (OTLP/gRPC default) |
| `RUST_LOG` | The log level filter (e.g., `info`, `hyperstreamdb=debug`). | `info` |

### Core Instrumentation
We instrument key paths to provide visibility into latency and bottlenecks:
- **Write Path:** `write_async`, `commit_async`
- **Read Path:** `read_async`, `stream_all`, `vector_search_index`, `vector_search_flat`
- **Manifest Orchestration:** Optimistic concurrency loops, schema updates, and conflict resolution in the `ManifestManager`.

## Stateless Operational CLI (`hdb`)

The `hdb` binary is a standalone CLI tool that performs administrative actions directly against the object storage tier without requiring a long-running database server to be active.

### Usage

```bash
# Start an interactive SQL REPL
hdb repl

# Execute a single SQL query
hdb query --query "SELECT * FROM my_table LIMIT 10"

# Register a table in the session
hdb register --name my_table --uri s3://my-bucket/my-table
```

### Table Management

You can perform routine maintenance using the `hdb table` subcommand:

```bash
# Inspect table metadata and statistics
hdb table inspect --uri s3://my-bucket/my-table

# Compact small data files to optimize read performance
hdb table compact --uri s3://my-bucket/my-table

# Vacuum (delete) data files that are no longer referenced and older than N days
hdb table vacuum --uri s3://my-bucket/my-table --older-than-days 7
```

*Note: The CLI is stateless; it interacts directly with the storage URI.*
