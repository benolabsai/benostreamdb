# BenoStreamDB Arrow Flight & Flight SQL Gateway

`benostreamdb-flight` provides a high-performance native Apache Arrow Flight and Flight SQL interface over BenoStreamDB.

> **Architecture Note:** BenoStreamDB is **serverless-first**. Arrow Flight SQL is an optional server interface for network-accessible database access. Flight SQL currently provides **single-process server access**; distributed or clustered execution across multiple nodes is not currently supported or implied.

## Overview

Arrow Flight SQL is the standard, language-agnostic data transport and database protocol for analytical engines. It enables zero-copy, highly parallelized stream transfers of Arrow RecordBatches between clients and BenoStreamDB over gRPC/HTTP/2.

### Architecture

```
Client Application (Python pyarrow.flight / JDBC / Go / C++ / Rust)
                    │
                    ▼
         Arrow Flight SQL (gRPC / HTTP2)
                    │
                    ▼
         BenoStreamFlightSqlService
                    │
                    ▼
           BenoStreamSession
                    │
                    ▼
       DataFusion Query Engine + BenoStreamDB Core
```

## Features

- **Standard Flight SQL Protocol**: Supports `CommandStatementQuery`, `CommandPreparedStatementQuery`, and metadata introspection (`CommandGetTables`, `CommandGetCatalogs`, `CommandGetDbSchemas`, etc.).
- **Zero-Copy Arrow Transport**: Streams record batches directly into memory without row-wise serialization or deserialization penalties.
- **DataFusion Integration**: Direct query execution backed by DataFusion with vector search and indexing pushdowns.
- **Ecosystem Compatibility**: Works out-of-the-box with any Flight SQL compliant client (DBeaver, JDBC Flight SQL driver, PyArrow, DuckDB, Apache Spark, etc.).

## Running the Server

```bash
cargo run -p benostreamdb-flight --bin bsdb-flight -- --port 50051 --table-uri file:///path/to/table
```

## Connecting with Python (PyArrow Flight SQL)

```python
from pyarrow import flight
import pyarrow.flight as flight_sql

client = flight.FlightClient("grpc://localhost:50051")
# Execute query via Flight SQL
info = client.get_flight_info(flight.FlightDescriptor.for_command(b"SELECT * FROM t LIMIT 10"))
reader = client.do_get(info.endpoints[0].ticket)
table = reader.read_all()
print(table.to_pandas())
```
