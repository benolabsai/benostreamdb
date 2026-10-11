# Arrow Flight SQL Gateway (`benostreamdb-flight`)

`benostreamdb-flight` provides a high-performance Apache Arrow Flight and Flight SQL network interface over BenoStreamDB.

---

## Architecture & Deployment Model

BenoStreamDB is **serverless-first**: applications with native language support (Python, Rust, JVM via Spark/Trino) embed the engine directly in-process without needing a database server.

For network clients, polyglot applications (Go, C++, Node.js), and BI tools that communicate via JDBC/ODBC/ADBC, the optional Flight SQL server exposes the engine over standard gRPC.

> **Architecture Note:** Arrow Flight SQL currently provides **single-process server access**. Distributed coordinator scheduling or multi-node clustered execution is not implied by Flight SQL support.

```
Client Application (JDBC / ODBC / ADBC / Go / C++ / BI tools)
                    │
                    ▼
         Arrow Flight SQL (gRPC / HTTP/2)
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

---

## ⚠️ Security & Authentication

* **Stateless Authentication**: The server supports stateless authentication via static API keys (`BSDB_API_KEY`) and JWT tokens (HS256 shared secret or RS256 IdP public keys). Tokens are transmitted via the gRPC `authorization: Bearer <token>` metadata header.
* **No Native TLS**: The server binary does **not** terminate TLS natively. When exposed across network boundaries, deploy behind a reverse proxy (e.g., Envoy, NGINX, AWS ALB) that handles TLS termination and certificates.
* **Binding**: Defaults to `127.0.0.1:50051`. For hardened environments, set `BSDB_AUTH_REQUIRED=true` to fail closed if unauthenticated requests arrive.

---

## Configuration

Configure the server via environment variables:

| Variable | Description | Default |
|:---|:---|:---|
| `BSDB_FLIGHT_BIND` | Bind address for gRPC listener | `127.0.0.1` |
| `BSDB_FLIGHT_PORT` | Port for Flight SQL listener | `50051` |
| `BSDB_WAREHOUSE` | Default storage warehouse for created tables | `./data` |
| `BSDB_CONFIG` | Path to catalog configuration TOML | Unset |
| `BSDB_API_KEY` | Shared secret API key for client auth | Unset |
| `BSDB_JWT_SECRET` | Secret for HMAC-SHA256 JWT tokens | Unset |
| `BSDB_AUTH_REQUIRED` | Fail closed on unauthenticated requests | `false` |
| `BSDB_METRICS_BIND` | HTTP metrics listener address | `127.0.0.1` |
| `BSDB_METRICS_PORT` | HTTP Prometheus metrics port | `9090` |

---

## Running the Server

Build and start the server:

```bash
cargo run --release -p benostreamdb-flight
```

Or run via Docker:

```bash
docker run -p 50051:50051 -p 9090:9090 \
  -e BSDB_API_KEY="your-secret-key" \
  -e BSDB_WAREHOUSE="s3://my-bucket/warehouse" \
  benostreamdb-flight:latest
```

---

## Client Usage Examples

### 1. Python via ADBC (Arrow Database Connectivity)

```python
import pyarrow.flight as flight
from adbc_driver_flightsql import dbapi

# Connect with Bearer authentication
conn = dbapi.connect(
    uri="grpc://127.0.0.1:50051",
    db_kwargs={
        "adbc.flight.sql.rpc.call_header.authorization": "Bearer your-secret-key"
    }
)

with conn.cursor() as cur:
    cur.execute("SELECT id, title FROM documents ORDER BY id LIMIT 10")
    results = cur.fetchallarrow()
    print(results)
```

### 2. DuckDB

```sql
INSTALL flight;
LOAD flight;

-- Query BenoStreamDB via Flight SQL
SELECT * FROM flight_sql_query('grpc://127.0.0.1:50051', 'SELECT * FROM documents');
```

### 3. BI & SQL Tools (DBeaver, Tableau, Superset)

Use the standard Apache Arrow Flight SQL JDBC driver. Configure the connection URL as:
```
jdbc:arrow-flight-sql://127.0.0.1:50051?useEncryption=false
```
