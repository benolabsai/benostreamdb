# Python Bindings

HyperStreamDB provides high-performance Python bindings using [PyO3](https://github.com/PyO3/pyo3). This allows you to use HyperStreamDB directly from Python scripts, Jupyter notebooks, and AI pipelines.

## Installation

```bash
pip install hyperstreamdb
```

*(Note: Ensure you have the Rust toolchain installed if building from source)*

## Usage

### Catalog Selection and Configuration

HyperStreamDB supports multiple catalog backends (Nessie, Hive, REST, Glue, Unity). You can configure the catalog using a factory method or a TOML configuration file.

#### 1. Direct Instantiation

```python
import hyperstreamdb as hdb

# Create a Nessie catalog
catalog = hdb.create_catalog("nessie", {"url": "http://localhost:19120"})

# Create a Hive catalog
# Note: Requires Thrift connection
catalog = hdb.create_catalog("hive", {
    "url": "thrift://localhost:9083", 
    "warehouse": "s3://bucket/warehouse"
})

# Create a Unity Catalog
catalog = hdb.create_catalog("unity", {
    "url": "https://<host>.cloud.databricks.com",
    "token": "dapi123..."
})
```

#### 2. Default Configuration (Recommended)

You can define your catalog configuration in a standard location. HyperStreamDB searches in the following order:

1.  `HYPERSTREAM_CONFIG` (Environment Variable path)
2.  `./hyperstream.toml` (Current Directory)
3.  `~/.hyperstream/config.toml` (Home Directory)

**Load Default Catalog:**
```python
# Automatically loads from the first found config file
catalog = hdb.load_default_catalog()
```

**Example TOML Config (`hyperstream.toml`):**
```toml
catalog_type = "nessie"

[config]
url = "http://localhost:19120"
branch = "main"
```

#### 3. Load Config from Specific File

```python
catalog = hdb.create_catalog_from_config("/path/to/my_config.toml")
```

See [examples/configs/](../examples/configs/) for example configuration files for each catalog type.

### Writing Data

```python
import hyperstreamdb as hyperstream
import pandas as pd

# Create a writer
writer = hyperstream.Writer("s3://my-bucket/dataset")

# Create a dataframe
df = pd.DataFrame({
    "id": [1, 2, 3],
    "text": ["hello", "world", "hyperstream"],
    "vector": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
})

# Write the segment
writer.write_dataframe(df)
writer.commit()
```

### Reading Data (with Predicate Pushdown)

```python
reader = hyperstream.Reader("s3://my-bucket/dataset")

# Filter logic is pushed down to Rust and uses Inverted Indexes
# Only relevant rows are materialized into Pandas
df = reader.query(
    filter="id > 1 AND text LIKE '%world%'",
    columns=["id", "text", "vector"]
)

print(df)
```

## Architecture

The Python binding is a thin wrapper around the Rust core. It uses **Arrow C Data Interface** to transfer data between Rust (Apache Arrow) and Python (Pandas/PyArrow) with **zero-copy**. This ensures that reading data in Python is as fast as reading it in Rust.
