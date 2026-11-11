# Multi-Catalog Usage Guide

BenoStreamDB supports enterprise-grade data catalogs to provide table discovery, atomic commits, and snapshot isolation across your data lake. 

## Supported Catalogs

| Catalog | Protocol | Optimistic CAS Commit Behavior | Use Case |
|---|---|---|---|
| **Memory Catalog (`memory`)** | In-process `RwLock` | Validates Iceberg `TableRequirement` predicates (`assert-table-uuid`, `assert-ref-snapshot-id`, schema/spec/field ids) and emits HTTP-409 equivalent conflicts. | Zero-infrastructure unit/integration tests, shared-URI (`memory://<name>`) multi-writer convergence testing. |
| **JDBC Catalog (`jdbc`)** | `sqlx::AnyPool` (PostgreSQL / SQLite) | Validates `TableRequirement` predicates against persisted `TableMetadata` and executes `UPDATE ... WHERE metadata_location = ?` (`$N` on Postgres) compare-and-swap, returning `409` on zero rows updated. | Embedded SQLite single-node catalogs (`sqlite::memory:` / `sqlite:///...`) and relational PostgreSQL Iceberg metadata stores. |
| **Iceberg REST (`rest`)** | REST v1 (Bearer / OAuth2 / AWS SigV4) | Sends full Iceberg REST `requirements` + `updates` payload; discovers `/v1/config` prefixes when `warehouse` is set; retries up to 8 times on HTTP `409 Conflict`. | Vendor-neutral interoperable standard (Apache Polaris, Lakekeeper, Snowflake Open Catalog, generic SigV4 REST endpoints). |
| **Amazon S3 Tables (`s3tables`)** | Iceberg REST v1 (`SigV4`, `s3tables` service) | Auto-extracts region from table bucket ARN, signs every request with `x-amz-content-sha256` AWS SigV4, discovers prefix via `/v1/config?warehouse=<arn>`, and uses native REST `409` CAS retries. | Managed Amazon S3 Tables buckets. |
| **AWS Glue (`glue`)** | Native SDK or Iceberg REST (`mode = "rest"`) | Validates `TableRequirement`s and commits with Glue `version_id` conditional lock (`ConcurrentModificationException` mapped to `409`), or delegates to Glue Iceberg REST (`service = "glue"`) in `mode = "rest"`. | AWS serverless metadata management. |
| **Hive Metastore (`hive`)** | Thrift | Validates `TableRequirement`s and executes `alter_table_with_environment_context` with `expected_parameter_key = "metadata_location"`, mapping mismatch exceptions to `409`. | Enterprise Hadoop ecosystem integration. |
| **Project Nessie (`nessie`)** | REST v2 (optional Bearer token) | Validates `TableRequirement`s, handles `set-metadata-location` and `DELETE`, commits against branch `expectedHash`, and maps HTTP `409 ReferenceConflictException` to retryable `409`. | Branching, tagging, and time-travel data engineering workflows. |
| **Unity Catalog (`unity`)** | Databricks Unity REST & Iceberg REST | Supports both standard Unity Tables REST (`PATCH` with `previous_metadata_location` + `TableRequirement` verification) and **Databricks Unity Iceberg REST** (`/api/2.1/unity-catalog/iceberg-rest` full `/v1` `409` CAS). | Databricks Lakehouse integration. |

> [!NOTE]
> Multi-writer optimistic concurrency convergence relies on `commit_table_with_requirements` returning `BenoStreamError::CatalogRequest { status: Some(409), .. }` when a concurrent commit advances `metadata_location` or `current-snapshot-id`. All catalog adapters enforce this CAS contract (`MAX_CATALOG_RETRIES = 8` in [`maintenance.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/table/maintenance.rs)). All remote HTTP/object-store URIs entered via `create_catalog_async` are checked by the built-in SSRF guard against loopback, private, and link-local hosts.

---

## 0. Memory and JDBC Catalogs (Zero-Infrastructure & SQL Backends)

### In-Process Memory Catalog (`memory`)
Use `memory` for deterministic tests and multi-writer conflict simulations without standing up an external metadata server. Naming two clients with the same `memory://<name>` URI shares one process-wide `RwLock` catalog state so concurrent writers observe each other's commits and `409` conflicts:

```python
import pyarrow as pa
import benostreamdb as bsdb

catalog = bsdb.create_catalog(
    "memory",
    {"uri": "memory://dev_catalog", "warehouse": "file:///tmp/bsdb_wh"},
)
catalog.create_namespace("analytics")

schema = bsdb.Schema.from_arrow(pa.schema([pa.field("id", pa.int32(), nullable=False)]))
catalog.create_table("analytics", "events", schema)

table = catalog.load_table("analytics", "events")
table.write(pa.table({"id": [1, 2, 3]}))
table.commit()
```

### JDBC Catalog (`jdbc`: SQLite and PostgreSQL)
BenoStreamDB's JDBC catalog routes PostgreSQL (`postgres://`, `$1..$N` parameters) and SQLite (`sqlite://`, `sqlite::memory:`, `?` parameters) through `sqlx::AnyPool`, using `VARCHAR(1024)` for `metadata_location` and conditional `WHERE metadata_location = <expected>` updates:

```python
import benostreamdb as bsdb

# Embedded single-connection SQLite in-memory catalog
mem_jdbc = bsdb.create_catalog(
    "jdbc",
    {"uri": "sqlite::memory:", "warehouse": "file:///tmp/bsdb_jdbc_wh", "catalog_name": "default"},
)

# File-backed SQLite or PostgreSQL catalog
pg_jdbc = bsdb.create_catalog(
    "jdbc",
    {
        "uri": "postgres://iceberg:secret@pg-host:5432/iceberg_catalog",
        "warehouse": "s3://my-lakehouse-bucket/tables",
        "catalog_name": "prod",
    },
)
```

---

## 1. Hive Metastore (Detailed Example)

The Hive Metastore (HMS) is the industry standard for metadata management in Hadoop-compatible environments.

### Connection
```python
import benostreamdb as bsdb

# Connect to HMS via Thrift (no auth example)
catalog = bsdb.create_catalog("hive", {"url": "thrift://metastore-host:9083"})

# Load a table from the catalog (returns a Table)
table = catalog.load_table("default", "events_analytics")
```

### How it Works
When you load a table from Hive, BenoStreamDB:
1.  Queries the HMS for the `metadata_location` parameter in the table properties.
2.  Loads the corresponding Iceberg manifest from storage (S3/GCS/FS).
3.  On `commit()`, it writes a new manifest version and atomically updates the `metadata_location` in HMS using a CAS (Compare-and-Swap) operation on the backend database.

---

## 2. Project Nessie

Nessie provides Git-like semantics for your data lake, allowing you to branch and merge table states.

### Setup
Run Nessie locally via Docker:
```bash
docker run -p 19120:19120 projectnessie/nessie
```

### Python API
```python
import benostreamdb as bsdb

# Connect to Nessie
catalog = bsdb.create_catalog("nessie", {"url": "http://localhost:19120"})

# Create a branch for experimentation
catalog.create_branch("etl-job-v2", source_ref="main")

# Load a table from the specific branch
table = catalog.load_table("etl-job-v2", "users")
```

---

## 3. AWS Glue Catalog & Amazon S3 Tables

For AWS deployments, BenoStreamDB supports both **Amazon S3 Tables** (`s3tables`, Iceberg REST over AWS SigV4) and **AWS Glue** (native `aws-sdk-glue` with `version_id` optimistic locking or managed Glue Iceberg REST via `mode = "rest"`).

### Amazon S3 Tables (`s3tables`)
```python
import benostreamdb as bsdb

# Derives region (`us-east-1`) from the bucket ARN, signs requests via AWS SigV4
# using the default AWS credential chain, and auto-discovers prefix via `/v1/config`.
catalog = bsdb.create_catalog(
    "s3tables",
    {"warehouse": "arn:aws:s3tables:us-east-1:123456789012:bucket/analytics-tables"},
)

catalog.create_namespace("events")
table = catalog.load_table("events", "clickstream_data")
```

### AWS Glue (`glue` SDK & Iceberg REST)
```python
import benostreamdb as bsdb

# Native AWS Glue SDK (version_id CAS)
sdk_catalog = bsdb.create_catalog("glue", {"region": "us-east-1"})

# Managed AWS Glue Iceberg REST endpoint (SigV4 + /v1/config)
rest_catalog = bsdb.create_catalog(
    "glue",
    {"mode": "rest", "region": "us-east-1", "catalog_id": "123456789012"},
)

table = sdk_catalog.load_table("production_db", "clickstream_data")
```

### Databricks Unity Catalog (`unity`)
```python
import benostreamdb as bsdb

# Recommended: Databricks Unity Iceberg REST endpoint (full Iceberg CAS & 409 retries)
unity_iceberg = bsdb.create_catalog(
    "unity",
    {
        "url": "https://<workspace>.cloud.databricks.com/api/2.1/unity-catalog/iceberg-rest",
        "token": "<DATABRICKS_PAT>",
        "warehouse": "main",
    },
)

# Standard Unity Tables REST API
unity_tables = bsdb.create_catalog(
    "unity",
    {
        "url": "https://<workspace>.cloud.databricks.com",
        "token": "<DATABRICKS_PAT>",
        "warehouse": "main",
    },
)
```

---

## 4. Iceberg REST Catalog (Snowflake / Apache Polaris & Lakekeeper)

The vendor-neutral REST catalog is the most interoperable way to manage Iceberg tables across different engines (Trino, Spark, Snowflake, BenoStreamDB). BenoStreamDB natively supports the official Iceberg REST OpenAPI specification, including OAuth2 client credentials (`credential` / `client_id` + `client_secret`), static Bearer tokens (`token`), AWS SigV4 signing (`auth = "sigv4"`), and `/v1/config?warehouse=...` auto-discovery.

### Snowflake / Apache Polaris with OAuth2 Client Credentials
```python
import benostreamdb as bsdb

# Connect to an Apache Polaris REST catalog and load a table
catalog = bsdb.create_catalog("rest", {
    "url": "https://polaris.example.com/api/catalog/v1",
    "prefix": "my_warehouse",
    "credential": "<POLARIS_CLIENT_ID>:<POLARIS_CLIENT_SECRET>",
    "scope": "PRINCIPAL_ROLE:ALL",
})
table = catalog.load_table("production", "campaign_results")
```

All `benostreamdb` engines and servers (`benostreamdb-search`, `benostreamdb-flight`, and Python bindings) also accept catalog parameters from `benostream.toml` or `BSDB_SEARCH_CATALOG_*` / `BSDB_CATALOG_*` environment variables (`TYPE`, `URL`/`URI`, `TOKEN`, `CREDENTIAL`, `PREFIX`, `ID`, `REGION`, `WAREHOUSE`, `MODE`, `AUTH`, `SERVICE`, `SCOPE`). For OAuth2 credentials, the engine executes `/v1/oauth/tokens` exchanges and caches bearer tokens, refreshing automatically within 60 seconds of expiration.

### Static Token (Tabular / Nessie REST)
```python
catalog = bsdb.create_catalog("rest", {"url": "https://api.tabular.io/v1/"})
table = catalog.load_table("marketing", "campaign_results")
```

---

## Next Steps
* For zero-copy querying in Snowflake via Apache Polaris, see the [Snowflake + Polaris Integration Guide](CATALOG_USAGE.md).
* For cluster configuration or setting up catalog properties via `benostream.toml`, see the [Configuration Guide](CONFIGURATION.md).


---

## Snowflake Polaris

_(merged from the former `catalog_usage.md`)_


This guide walks through configuring a **Zero-Copy Open Data Lakehouse** connecting **BenoStreamDB**, **Apache Polaris**, and **Snowflake**. 

With this architecture:
- **BenoStreamDB** provides sub-2ms hybrid vector (HNSW) and full-text (BM25) search.
- **Apache Polaris** acts as the centralized open Iceberg REST catalog.
- **Snowflake** queries the exact same Parquet data natively using standard Snowflake SQL without data duplication, ETL pipelines, or storage lock-in.

---

## Architecture Overview

```mermaid
flowchart LR
    subgraph S3 ["Cloud Storage (AWS S3 / GCS / Azure)"]
        Parquet["Parquet Data Segments"]
        HNSW["HNSW Vector Sidecars"]
        Meta["metadata/vX.metadata.json"]
    end

    subgraph Polaris ["Apache Polaris (Iceberg REST Catalog)"]
        CatalogPointer["Table: customer_reviews<br>Active: vX.metadata.json"]
    end

    subgraph Workloads ["Multi-Engine Execution"]
        BSDB["BenoStreamDB / bsdb-search<br>• Real-time REST ingestion<br>• Sub-2ms Vector / BM25 Search"]
        Snowflake["Snowflake Cloud Data Warehouse<br>• Native SQL queries & aggregations<br>• Zero-copy joins with enterprise data<br>• BI dashboards & Snowpark ML"]
    end

    BSDB -->|1. Writes Parquet & Commits| S3
    BSDB -->|2. Iceberg Atomic Swap| Polaris
    Polaris -.->|3. Synchronizes Snapshot| Snowflake
    Snowflake -->|4. Zero-Copy Query Execution| S3
```

---

## Prerequisites

1. An **S3 Bucket** (or GCS/Azure container) storing your BenoStreamDB tables.
2. An active **Apache Polaris** instance (Open-source Polaris or Snowflake-managed Open Catalog).
3. A **Snowflake** account with `ACCOUNTADMIN` or privileges to create storage and catalog integrations.

---

## Step 1: Configure BenoStreamDB to Sync with Polaris

Configure `benostreamdb-search` using environment variables or a `benostream.toml` configuration file.

### Option A: Environment Variables

```bash
# Storage location for Parquet data and Iceberg metadata
export BSDB_SEARCH_STORAGE_URI=s3://my-lakehouse-bucket/tables

# Polaris Iceberg REST catalog configuration
export BSDB_SEARCH_CATALOG_TYPE=rest
export BSDB_SEARCH_CATALOG_URL=https://polaris.example.com/api/catalog/v1
export BSDB_SEARCH_CATALOG_CREDENTIAL="<POLARIS_CLIENT_ID>:<POLARIS_CLIENT_SECRET>"
export BSDB_SEARCH_CATALOG_PREFIX="my_warehouse"
export BSDB_SEARCH_CATALOG_NAMESPACE="production"
```

### Option B: `benostream.toml`

```toml
[storage]
type = "s3"
bucket = "my-lakehouse-bucket"
region = "us-east-1"

[catalog]
catalog_type = "rest"

[catalog.config]
url = "https://polaris.example.com/api/catalog/v1"
credential = "POLARIS_CLIENT_ID:POLARIS_CLIENT_SECRET"
prefix = "my_warehouse"
namespace = "production"
```

When you start `bsdb-search`, startup diagnostics confirm Polaris catalog synchronization:
```
INFO Initialized external Iceberg catalog from environment catalog_type=Rest namespace=production
```

---

## Step 2: Configure Storage Access in Snowflake (External Volume)

Create an `EXTERNAL VOLUME` in Snowflake pointing to the same S3 bucket where BenoStreamDB writes Parquet files:

```sql
CREATE OR REPLACE EXTERNAL VOLUME benostream_s3_volume
  STORAGE_LOCATIONS =
    (
      (
        NAME = 'my-s3-us-east-1'
        STORAGE_PROVIDER = 'S3'
        STORAGE_BASE_URL = 's3://my-lakehouse-bucket/tables/'
        STORAGE_AWS_ROLE_ARN = 'arn:aws:iam::123456789012:role/snowflake_s3_read_role'
      )
    );

-- Retrieve Snowflake IAM user ARN to authorize trust relationship in AWS IAM
DESCRIBE EXTERNAL VOLUME benostream_s3_volume;
```

---

## Step 3: Configure Polaris Catalog Integration in Snowflake

Create a `CATALOG INTEGRATION` in Snowflake that tells Snowflake how to query Polaris:

```sql
CREATE OR REPLACE CATALOG INTEGRATION polaris_iceberg_catalog
  CATALOG_SOURCE = ICEBERG_REST
  TABLE_FORMAT = ICEBERG
  CATALOG_NAMESPACE = 'production'
  REST_CONFIG = (
    CATALOG_URI = 'https://polaris.example.com/api/catalog/v1'
    WAREHOUSE = 'my_warehouse'
  )
  REST_AUTHENTICATION = (
    TYPE = OAUTH2
    OAUTH_CLIENT_ID = '<POLARIS_CLIENT_ID>'
    OAUTH_CLIENT_SECRET = '<POLARIS_CLIENT_SECRET>'
    OAUTH_ALLOWED_SCOPES = ('PRINCIPAL_ROLE:ALL')
  )
  ENABLED = TRUE;
```

---

## Step 4: Create the Table in Snowflake

Create an Iceberg table in Snowflake referencing the Polaris catalog table name:

```sql
USE DATABASE my_analytics_db;
USE SCHEMA public;

CREATE OR REPLACE ICEBERG TABLE customer_reviews
  EXTERNAL_VOLUME = 'benostream_s3_volume'
  CATALOG = 'polaris_iceberg_catalog'
  CATALOG_TABLE_NAME = 'customer_reviews';
```

---

## Step 5: Ingest via BenoStreamDB and Query in Snowflake

### 1. Ingest Data via BenoStreamDB Search API

Ingest documents via the OpenSearch-compatible REST API:

```bash
curl -X POST "http://localhost:9200/customer_reviews/_doc" \
  -H "Content-Type: application/json" \
  -d '{
    "review_id": 1001,
    "product": "Wireless Headphones",
    "rating": 5,
    "review_text": "Incredible sound isolation and deep bass.",
    "embedding": [0.12, -0.45, 0.88, 0.23]
  }'
```

* **What happens immediately:**
  1. BenoStreamDB persists the row to Parquet on S3.
  2. BenoStreamDB indexes the vector in the local HNSW sidecar for sub-2ms ANN search.
  3. BenoStreamDB triggers an **Iceberg Atomic Swap** to Apache Polaris, advancing the snapshot ID.

### 2. Query Natively in Snowflake

Run queries in Snowflake against `customer_reviews` immediately:

```sql
-- Standard SQL analytics & aggregations
SELECT 
    product,
    AVG(rating) AS avg_rating,
    COUNT(*) AS total_reviews
FROM customer_reviews
GROUP BY product
ORDER BY total_reviews DESC;

-- Native join with Snowflake enterprise data
SELECT 
    r.review_id,
    r.product,
    r.rating,
    o.order_total,
    o.customer_segment
FROM customer_reviews r
JOIN snowflake_internal.sales.orders o 
  ON r.review_id = o.review_reference_id
WHERE r.rating <= 2;
```

---

## Key Benefits

1. **Zero-Copy Architecture**: No `COPY INTO`, no Kafka-to-Snowflake connectors, and no duplicate storage charges.
2. **Dual-Speed Workloads**:
   * **Sub-2ms Interactive Search**: Applications query `bsdb-search` over HTTP for instant vector & full-text retrieval.
   * **Large-Scale BI & SQL**: Data analysts and data scientists query the exact same data in Snowflake.
3. **Open Standards**: If you ever migrate away from Snowflake, all data remains in vanilla Apache Iceberg / Parquet format with Apache Polaris as the vendor-neutral catalog.
