# HyperStreamDB Catalog Usage Guide

HyperStreamDB integrates with the **Nessie Project** (Iceberg-compatible REST Catalog) to provide Git-like semantics for your data. This allows you to specific snapshots, create branches for experimentation, and merge changes atomically.

## 1. Setup

You need a running Nessie server. You can run one locally using Docker:

```bash
docker run -p 19120:19120 projectnessie/nessie
```

The Nessie API will be available at `http://localhost:19120`.

## 2. Python API Overview

HyperStreamDB exposes the `PyNessieCatalog` class for interacting with the catalog.

### Initialization

```python
import hyperstreamdb as hdb

# Connect to local Nessie instance
catalog = hdb.PyNessieCatalog("http://localhost:19120")
```

### creating a Table (Iceberg-compatible)

To create a table, you need to define its schema and provide a location. Note that in Nessie, tables are tracked on specific **branches** (default is `main`).

```python
import json

# Define Schema (Iceberg JSON format)
schema_json = json.dumps({
    "type": "struct",
    "schema-id": 0,
    "fields": [
        {"id": 0, "name": "user_id", "type": "int", "required": True},
        {"id": 1, "name": "event_ts", "type": "timestamp", "required": False},
        {"id": 2, "name": "value", "type": "double", "required": False}
    ]
})

# Create Table on 'main' branch
# Note: For S3 usage, location would be "s3://my-bucket/tables/events"
catalog.create_table(
    branch="main",
    table_name="events",
    location="file:///tmp/hyperstream/events", 
    schema_json=schema_json
)
```

### Writing Data

Currently, writes are performed directly via `PyTable` or `PyWriter`. The catalog integration tracks the *metadata* pointers.

```python
import pandas as pd

# 1. Connect to the table location directly for writing
# (Future versions will load configuration from Catalog)
table = hdb.PyTable("file:///tmp/hyperstream/events")

# 2. Write Data
df = pd.DataFrame({
    "user_id": [1, 2, 3],
    "value": [10.5, 20.0, 15.2]
})
table.write_pandas(df)

# Note: Automatic committing to Nessie is not yet transparently hooked into 'write_pandas'.
# In the current phase, you manage the data writes and catalog commits separately or use the forthcoming 'transaction' API.
```

### Branching

You can create branches to isolate changes (e.g., for ETL jobs or testing).

```python
# Create 'dev' branch from 'main'
catalog.create_branch("dev", source_ref="main")

# Now you can create tables or commit changes to 'dev' without affecting 'main'
catalog.create_table(
    branch="dev", 
    table_name="experimental_table",
    location="...",
    schema_json=schema_json
)
```

### Time Travel / Reading Specific References

The `PyReader` supports reading from specific table locations. The Catalog is used to *resolve* which location (snapshot) corresponds to a specific branch or time.

*(Note: Full `load_table` resolution logic is currently being finalized in Phase 2)*.

## 3. Example Workflow

1.  **Ingest**: ETL job writes raw data to `raw_data` table on `main`.
2.  **Branch**: Data Scientist creates `experiment-1` branch from `main`.
3.  **Tests**: DS modifies data or merges new segments on `experiment-1`.
4.  **Validate**: Verify results on branch.
5.  **Merge**: (Upcoming) Merge `experiment-1` back to `main`.
