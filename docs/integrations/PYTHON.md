# Python Bindings

BenoStreamDB provides high-performance Python bindings using [PyO3](https://github.com/PyO3/pyo3). This allows you to use BenoStreamDB directly from Python scripts, Jupyter notebooks, and AI pipelines.

## Installation

```bash
pip install benostreamdb
```

*(Note: Ensure you have the Rust toolchain installed if building from source)*

## Usage

### Catalog Selection and Configuration

BenoStreamDB supports multiple catalog backends (Nessie, Hive, REST, Glue, Unity). You can configure the catalog using a factory method or a TOML configuration file.

#### 1. Direct Instantiation

```python
import benostreamdb as bsdb

# Create a Nessie catalog
catalog = bsdb.create_catalog("nessie", {"url": "http://localhost:19120"})

# Create a Hive catalog
# Note: Requires Thrift connection
catalog = bsdb.create_catalog("hive", {
    "url": "thrift://localhost:9083", 
    "warehouse": "s3://bucket/warehouse"
})

# Create a Unity Catalog
catalog = bsdb.create_catalog("unity", {
    "url": "https://<host>.cloud.databricks.com",
    "token": "dapi123..."
})
```

#### 2. Default Configuration (Recommended)

You can define your catalog configuration in a standard location. BenoStreamDB searches in the following order:

1.  `BSDB_CONFIG` (Environment Variable path)
2.  `./benostream.toml` (Current Directory)
3.  `~/.benostream/config.toml` (Home Directory)

**Load Default Catalog:**
```python
# Automatically loads from the first found config file
catalog = bsdb.load_default_catalog()
```

**Example TOML Config (`benostream.toml`):**
```toml
catalog_type = "nessie"

[config]
url = "http://localhost:19120"
branch = "main"
```

#### 3. Load Config from Specific File

```python
catalog = bsdb.create_catalog_from_config("/path/to/my_config.toml")
```

See [examples/configs/](../../examples/configs/) for example configuration files for each catalog type.

### Table Operations (Writing and Reading)

BenoStreamDB centers around the `Table` class, which manages underlying Apache Iceberg Parquet files and rebuildable index overlays.

```python
import benostreamdb as bsdb
import pandas as pd
import numpy as np

# Open or initialize a table directly from object storage or local path
table = bsdb.Table("s3://my-bucket/dataset")

# Prepare data
df = pd.DataFrame({
    "id": [1, 2, 3],
    "text": ["hello", "world", "benostream"],
    "vector": [[0.1, 0.2, 0.3], [0.3, 0.4, 0.5], [0.5, 0.6, 0.7]]
})

# Append rows (accepts pandas DataFrame or pyarrow Table/RecordBatch)
table.append(df)
table.commit()

# Read data back into Arrow or Pandas
arrow_table = table.to_arrow()
pdf = table.to_pandas()
```

### Vector Search & Index Overlays

You can configure vector, full-text (BM25), or bitmap indexes over table columns and perform similarity searches:

```python
# Add an HNSW vector index on the 'vector' column
# Index algorithms: "hnsw" (default float32), "hnsw_tq8", "hnsw_tq4", "bm25", "bitmap"
table.add_index(["vector"], index_type="hnsw")
table.build_indexes()

# Top-k vector search
query_vector = [0.1, 0.2, 0.3]
results = table.vector_search(query_vector, column="vector", k=10)
```

### SQL Analytics & Filtering

Execute ANSI SQL queries with pgvector distance operators directly through `Table.sql()` or an engine `Session`:

```python
# Ad-hoc query on the table (referenced as 'self' or table name)
matches = table.sql("SELECT id, text FROM self WHERE id > 1 AND text LIKE '%world%'")
print(matches.to_pandas())

# Full DataFusion session across multiple tables
session = bsdb.Session()
session.register_table("documents", table)
res = session.sql("""
    SELECT id, text, embedding <-> '[0.1, 0.2, 0.3]'::vector AS dist
    FROM documents
    ORDER BY dist ASC
    LIMIT 5
""")
print(res.to_arrow())
```

### Live Subscriptions (change feed)

`Table.subscribe()` returns a live subscription to the table's committed changes. `recv()` blocks (with an optional timeout); `try_recv()` is non-blocking. `subscribe_filtered("age > 30")` yields only matching rows. `close()` (or dropping the object / using it as a context manager) unsubscribes.

```python
sub = table.subscribe()                 # or table.subscribe_filtered("age > 30")
ev = sub.recv(timeout_ms=1000)          # {"event_type": "batch", "rows": 3, "data": <pyarrow.Table>}
ev = sub.try_recv()                     # non-blocking; None when idle
sub.close()                             # unsubscribe (idempotent)

with table.subscribe() as sub:          # context manager
    ...
```

The change feed is **in-process**: it observes commits made by writers in the same process. The same primitive is available as the `subscribe_events` SQL table function (reachable from every connector) and as a Flight SQL streaming ticket.

### Compute Device Configuration

BenoStreamDB provides dynamic hardware acceleration via the `Device` class, with automatic detection and graceful CPU fallback:

```python
# Auto-detect best available device (CUDA, Apple Silicon Metal, or CPU)
device = bsdb.Device("auto")
print(f"Active backend: {device.backend}")

# Explicit device selection
device = bsdb.Device("cuda")   # NVIDIA GPU (or AMD ROCm via HIP alignment)
device = bsdb.Device("mps")    # Apple Silicon Metal
device = bsdb.Device("cpu")    # CPU fallback
```

For stand-alone batch vector distance computations outside tables, see the [Python Vector API Guide](python_vector_api.md).

## Architecture

The Python binding is built with PyO3. It leverages the **Arrow C Data Interface** to exchange data between the Rust engine and Python runtimes (PyArrow/Pandas) with **zero-copy** memory sharing wherever possible, avoiding unnecessary serialization overhead.
