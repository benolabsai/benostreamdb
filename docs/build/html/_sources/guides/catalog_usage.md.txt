# Multi-Catalog Usage Guide

HyperStreamDB supports enterprise-grade data catalogs to provide table discovery, atomic commits, and snapshot isolation across your data lake. 

## Supported Catalogs

| Catalog | Protocol | Use Case |
|---------|----------|----------|
| **Hive Metastore** | Thrift | Enterprise standard, Hadoop ecosystem. |
| **Project Nessie** | REST v2 | Git-like versioning (branching, merging). |
| **AWS Glue** | Native SDK | AWS cloud-native metadata management. |
| **Iceberg REST** | REST v1 | Vendor-neutral, interoperable standard. |
| **Unity Catalog** | REST | Databricks ecosystem integration. |

---

## 1. Hive Metastore (Detailed Example)

The Hive Metastore (HMS) is the industry standard for metadata management in Hadoop-compatible environments.

### Connection
```python
import hyperstreamdb as hdb

# Connect to HMS via Thrift (no auth example)
table = hdb.Table.from_hive(
    address="thrift://metastore-host:9083",
    namespace="default",
    table="events_analytics"
)
```

### How it Works
When you load a table from Hive, HyperStreamDB:
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
# Connect to Nessie
catalog = hdb.NessieCatalog("http://localhost:19120")

# Create a branch for experimentation
catalog.create_branch("etl-job-v2", source_ref="main")

# Load table from the specific branch
table = hdb.Table.from_nessie(
    "http://localhost:19120", 
    namespace="prod", 
    table="users",
    ref="etl-job-v2"
)
```

---

## 3. AWS Glue Catalog

For AWS users, the Glue Data Catalog provides a managed, serverless metadata store.

### Usage
```python
# HyperStreamDB uses your local AWS credentials (IAM/Env)
table = hdb.Table.from_glue(
    namespace="production_db", 
    table="clickstream_data"
)
```

---

## 4. Iceberg REST Catalog

The vendor-neutral REST catalog is the most interoperable way to manage Iceberg tables across different engines (Trino, Spark, HyperStreamDB).

### Usage
```python
table = hdb.Table.from_rest(
    url="https://api.tabular.io/v1/",
    namespace="marketing",
    table="campaign_results",
    token="YOUR_OAUTH_TOKEN"  # Optional OAuth2 token
)
```

---

## Next Steps
More detailed guides for authentication (Kerberos, OAuth2, IAM Roles) and advanced branching workflows are coming in future releases.
