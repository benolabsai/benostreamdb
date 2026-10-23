# Snowflake Integration Guide

> ⚠️ **Developer Preview / Untested Against Live Snowflake Accounts**  
>  
> This integration architecture is built on the **Apache Iceberg Table Specification (V2/V3)** and the **Iceberg REST Catalog Specification** (compatible with Snowflake Open Catalog / Apache Polaris).  
> While BenoStreamDB's underlying Iceberg engine, detached overlay catalog, and REST client are verified through automated integration tests, **end-to-end certification against live, production Snowflake Enterprise accounts is currently in development and untested in this release**.  
> Evaluate this workflow in a sandbox or non-production environment.

---

## 📖 Overview

Enterprises that store analytical datasets in Snowflake can use BenoStreamDB to layer reconstructible, high-performance secondary indexes—**vector ANN (HNSW), full-text BM25, and scalar bitmap indexes**—over their Snowflake data with **zero data duplication** and **zero Snowflake compute credits**.

BenoStreamDB integrates with Snowflake via **Snowflake Iceberg Tables** residing in customer-managed object storage (Amazon S3, Google Cloud Storage, or Azure Blob Storage).

```text
┌─────────────────────────────────────────────────┐
│              Snowflake Data Cloud               │
│  • Writes Parquet data via External Volume      │
│  • Emits standard Iceberg metadata snapshots    │
└───────────────────────┬─────────────────────────┘
                        │
                        ▼ (writes Parquet + metadata)
┌─────────────────────────────────────────────────┐          ┌─────────────────────────────────────────┐
│     Customer Object Storage (S3 / GCS / Azure)  │          │     BenoStreamDB Overlay Storage        │
│             External Volume Prefix              │          │            (Separate Bucket)            │
│              (Strictly Read-Only)               │          │           (Writable Sidecars)           │
├─────────────────────────────────────────────────┤          ├─────────────────────────────────────────┤
│ • metadata/v1.metadata.json, snap-*.avro        │          │ • detached_catalog.json (lineage/sync)  │
│ • data/00001.parquet                            │◀─────────┤ • manifest.json (PK & index definitions)│
│ • data/00002.parquet                            │          │ • data/00001.bsdb_idx (HNSW / BM25)     │
│ (Managed & compacted by Snowflake)              │          │ • data/00002.bsdb_idx (HNSW / BM25)     │
└─────────────────────────────────────────────────┘          └─────────────────────────────────────────┘
```

---

## 🔄 Two Integration Approaches

### Approach A: Direct Storage Mounting (Detached Overlay Mode — Recommended)
**No Snowflake catalog credentials or API tokens needed.**

When Snowflake manages an Iceberg table on an External Volume, it writes standard Iceberg `v{N}.metadata.json` files and Parquet files into your cloud storage. BenoStreamDB mounts the storage path directly.

#### Step 1: Create an External Volume & Iceberg Table in Snowflake
```sql
-- 1. Create External Volume pointing to your S3 bucket
CREATE EXTERNAL VOLUME my_s3_ext_vol
   STORAGE_LOCATIONS =
      (
         (
            NAME = 'my-s3-us-east-1'
            STORAGE_PROVIDER = 'S3'
            STORAGE_BASE_URL = 's3://my-company-lakehouse/snowflake_warehouse/'
            STORAGE_AWS_ROLE_ARN = 'arn:aws:iam::123456789012:role/snowflake_role'
         )
      );

-- 2. Create a Snowflake-managed Iceberg Table
CREATE ICEBERG TABLE customer_interactions (
    interaction_id INT,
    customer_id INT,
    transcript STRING,
    embedding ARRAY,
    status STRING
)
EXTERNAL_VOLUME = 'my_s3_ext_vol'
CATALOG = 'SNOWFLAKE'
BASE_LOCATION = 'customer_interactions/';
```

#### Step 2: Mount the Table in BenoStreamDB
Point BenoStreamDB to the Snowflake external volume base location:

```python
import benostreamdb as bsdb

# Mount external Snowflake table in read-only mode
table = bsdb.Table.mount_external_iceberg(
    table_uri="s3://my-company-lakehouse/snowflake_warehouse/customer_interactions/",
    overlay_storage_uri="s3://my-index-overlays/customer_interactions/"
)

# Declare primary key for O(1) point lookups
table.set_primary_key(["interaction_id"])

# Build BM25 Okapi full-text search over conversation transcripts
table.add_index("transcript", "bm25")

# Build TurboQuant 8-bit quantized HNSW vector index
table.add_index("embedding", "hnsw_tq8")

# Wait for initial index build to complete
table.wait_for_background_tasks()
```

#### Step 3: Keep Indexes Synchronized with Snowflake Ingestion
When Snowflake pipelines insert rows or run background table maintenance (`OPTIMIZE`):

```python
# Option 1: Start background sync worker (polls metadata pointer every 60s)
table.start_detached_sync_worker(interval_seconds=60)

# Option 2: Trigger sync on demand (e.g. from an S3 ObjectCreated event)
synced = table.sync_detached_overlay()
if synced:
    print("New Snowflake snapshot detected; index overlays synchronized.")
```

#### Step 4: Query with Sub-Millisecond Search
```python
# 1. Full-Text BM25 Search
results = table.sql("""
    SELECT interaction_id, transcript
    FROM t
    WHERE transcript = 'account cancellation refund'
    LIMIT 10
""")

# 2. Vector ANN Search
results = table.vector_search("embedding", query_vector, k=10)
```

---

### Approach B: Via Snowflake Open Catalog (Polaris / Iceberg REST API)
If your organization requires table resolution through the **Iceberg REST Catalog** protocol rather than direct object storage URLs:

Snowflake supports the Iceberg REST Catalog standard via **Snowflake Open Catalog** (Apache Polaris).

```python
import benostreamdb as bsdb

# Connect via Iceberg REST Catalog endpoint
catalog = bsdb.create_catalog("rest", {
    "url": "https://<account_locator>.snowflakecomputing.com/polaris/api/catalog",
    "prefix": "MY_DATABASE",
})

table = catalog.load_table("MY_SCHEMA", "CUSTOMER_INTERACTIONS")
```

> The embedded Python REST wrapper takes `url` and an optional `prefix`; OAuth2
> client credentials are configured on the engine/search server through
> `benostream.toml` or the `BSDB_SEARCH_CATALOG_*` environment variables.

---

## ❓ What About Proprietary Snowflake Internal Tables?

Standard Snowflake tables (not created with `CREATE ICEBERG TABLE`) reside in Snowflake's proprietary, encrypted micro-partition format in internal stages. No third-party engine can directly scan these files from disk.

### Solution: Zero-ETL Iceberg Projection in Snowflake
Enterprises can project proprietary Snowflake tables into Iceberg format using a Snowflake CTAS or Dynamic Table:

```sql
CREATE ICEBERG TABLE customer_interactions_iceberg
  EXTERNAL_VOLUME = 'my_s3_ext_vol'
  CATALOG = 'SNOWFLAKE'
  BASE_LOCATION = 'customer_interactions/'
  AS SELECT * FROM proprietary_interactions;
```
Once written to the external volume, BenoStreamDB immediately mounts and indexes the data without ongoing migration pipelines.

---

## 🔒 Enterprise Security & Invariants

1. **Zero Snowflake Compute Costs**: All queries, vector ANN searches, BM25 text scans, and index builds run on BenoStreamDB compute. Zero Snowflake virtual warehouse credits are billed.
2. **Strict Read-Only Access**: BenoStreamDB requires only `s3:GetObject` on the Snowflake external volume prefix. It cannot alter, delete, or rewrite Snowflake data files.
3. **Reconstructible State**: All index sidecars live in the isolated `overlay_storage_uri`. If the overlay bucket is lost, all indexes can be rebuilt from the authoritative Iceberg Parquet files.
