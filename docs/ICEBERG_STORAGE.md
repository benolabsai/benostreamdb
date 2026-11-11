# Apache Iceberg Storage & Zero-Copy Interoperability

BenoStreamDB writes 100% spec-compliant **Apache Iceberg (V2/V3) tables stored as Apache Parquet**. 

Because the storage format is standard Iceberg Parquet and the secondary index overlays (HNSW, BM25, Roaring Bitmaps) are decoupled into persistent Puffin sidecars, external query engines—such as **Apache Spark, Trino, DuckDB, Polars, and PyIceberg**—can read BenoStreamDB tables directly with **zero copy, zero export, and zero vendor lock-in**.

---

## 1. The Indexed Lakehouse Storage Architecture

In BenoStreamDB, table state is decoupled into three structural layers:

```
                          Iceberg Table Root
                                   │
       ┌───────────────────────────┴───────────────────────────┐
       │                                                       │
1. Authoritative Columnar Data                  2. Advisory Secondary Index Overlays
       │                                                       │
  data/*.parquet                                     metadata/*.puffin
  • Standard ZSTD Parquet                             • Compound Puffin bundles (ADR-001)
  • Explicit field-ids on all columns                 • Roaring Bitmaps (scalar filters)
  • Readable by Spark, Trino, DuckDB                  • HNSW / TurboQuant (TQ8/TQ4) graphs
                                                      • BM25 Okapi term frequencies
                                                               │
                                       3. Snapshot & Catalog Metadata
                                                               │
                                            • metadata/v{N}.metadata.json
                                            • _manifest/snap-*.avro
                                            • _manifest/*-m*.avro (data manifests)
                                            • _manifest/*-d*.avro (delete manifests)
```

### The Zero-Copy Contract
1. **No Data Rewrites for Indexing**: Adding or dropping an index never rewrites the underlying Parquet files. Indexes are advisory mathematical overlays that map queries to row offsets within data files.
2. **The Overlay Invariant**: If an index sidecar is absent, corrupted, or stale, queries safely degrade to Parquet columnar scanning without failing or returning incorrect results.
3. **Full Spec Adherence**: External engines reading via the Iceberg catalog read authoritative data files and ignore advisory Puffin sidecars.

---

## 2. Reading & Writing Iceberg Tables (Python API)

### 2.1 Basic Table Creation and Appends

```python
import pyarrow as pa
import benostreamdb as bsdb

# Open or create an Iceberg table on local filesystem or object storage (S3/GCS/Azure)
table = bsdb.Table("s3://lakehouse/warehouse/telemetry_events")

# Define an explicit Arrow schema with an embedding column
schema = pa.schema([
    pa.field("id", pa.uint64()),
    pa.field("device_id", pa.string()),
    pa.field("category", pa.string()),
    pa.field("timestamp", pa.timestamp("ns")),
    pa.field("embedding", pa.list_(pa.float32(), 384)),
])

# Write Arrow RecordBatches or Tables
records = pa.Table.from_arrays(
    [
        pa.array([1, 2, 3], type=pa.uint64()),
        pa.array(["dev-01", "dev-02", "dev-03"]),
        pa.array(["edge", "gateway", "edge"]),
        pa.array([1700000000000000000, 1700000001000000000, 1700000002000000000], type=pa.timestamp("ns")),
        pa.FixedSizeListArray.from_arrays(pa.array([0.1] * (3 * 384), type=pa.float32()), 384),
    ],
    schema=schema,
)

table.write(records)
table.commit()
```

---

### 2.2 Physical Layout Control: Sort Orders (V2)

Controlling the physical sort order enables Parquet row-group range pruning, dramatically accelerating scalar filter evaluations.

```python
# Set table sort order applied during writes and compaction
table.replace_sort_order(
    columns=["timestamp", "device_id"],
    ascending=[False, True]  # timestamp DESC, device_id ASC
)

# Convenience alias
table.set_sort_order(["timestamp", "device_id"], [False, True])
```

- **Range Pruning**: External engines and BenoStreamDB evaluate min/max bounds in Parquet file metadata to skip unreferenced row groups.
- **Compression**: Sorted data clusters identical values, maximizing dictionary encoding and ZSTD run-length compression.

---

### 2.3 Partition Spec Evolution (V2)

Evolve table partitioning over time without rewriting historical data:

```python
from benostreamdb import PartitionField

# Initial temporal partitioning
table.update_spec([
    PartitionField(source_ids=[3], name="date", transform="day")
])

# Evolve partition strategy to add region partitioning
table.update_spec([
    PartitionField(source_ids=[3], name="date", transform="day"),
    PartitionField(source_ids=[2], name="category", transform="identity"),
])
```

Supported partition transforms:
- `identity`: Exact value partitioning.
- `year`, `month`, `day`, `hour`: Temporal partitioning on timestamps.
- `bucket(N)`: Consistent hash bucketing into $N$ buckets.
- `truncate(W)`: Truncates string or binary values to length $W$.

---

### 2.4 Column Statistics & NDV (Number of Distinct Values)

BenoStreamDB writes exact column statistics into data manifest files automatically during each commit to accelerate cost-based query planning and row-group pruning:

```python
# Inspect table metrics and index coverage
stats = table.statistics
print(f"Row count: {stats.row_count}")
print(f"File count: {stats.file_count}")
print(f"Total size: {stats.total_size_bytes} bytes")
print(f"Index coverage: {stats.index_coverage}")
```

Written automatically per field ID in data manifests:
- `null_value_counts`: Null presence per column.
- `lower_bounds` / `upper_bounds`: Min/max values for row-group pruning.
- `distinct_value_counts`: Exact NDV for cost-based join and filter estimation.

---

### 2.5 Row Lineage & V3 Capabilities

BenoStreamDB supports Iceberg V3 metadata primitives:
- `next-row-id`: Monotonically increasing 64-bit row identifier in table metadata.
- `first-row-id` & `added-rows`: Tracked on each snapshot for immutable row lineage.
- `_last_updated_sequence_number`: Tracks which transaction sequence modified each data file.

---

## 3. Zero-Copy External Engine Compatibility

To ensure external readers (Trino, DuckDB, Spark, PyIceberg) load BenoStreamDB tables without modification, the engine strictly enforces Iceberg structural metadata invariants:

| Metadata Component | Requirement for External Readers | Implementation in BenoStreamDB |
| :--- | :--- | :--- |
| `v{N}.metadata.json` | `TableMetadataParser`-compliant JSON | Mandatory kebab-case `schema-id`, non-empty `sort-orders`, and root snapshot IDs. |
| Manifest Lists (`snap-*.avro`) | Absolute URIs & typed Avro entries | Absolute paths to manifests; typed `partitions.element-id` metadata. |
| Data Manifests (`*-m*.avro`) | Field-IDs on every column | Avro user metadata carries `schema`, `partition-spec`, and `format-version`. |
| Parquet Data Files | Column `field_id` metadata | Every Parquet column (including nested elements and vector lists) carries an explicit Iceberg `field_id`. |
| Position Deletes | Reserved delete IDs | `del-pos-*.parquet` files store `file_path` and `pos` naming absolute data-file URIs. |

### Validating with PyIceberg

Verify zero-copy interoperability using `pyiceberg`:

```python
from pyiceberg.catalog import load_catalog

# Point PyIceberg directly at the BenoStreamDB table directory
catalog = load_catalog("default", **{"type": "filesystem"})
iceberg_table = catalog.load_table("s3://lakehouse/warehouse/telemetry_events")

# Read directly into a PyArrow Table via PyIceberg
arrow_data = iceberg_table.scan().to_arrow()
print(f"Successfully loaded {len(arrow_data)} rows via PyIceberg!")
```

---

## 4. Feature Conformance Matrix (Iceberg V2 / V3)

The table below reflects machine-checked verification via `tests/test_iceberg_conformance.rs` and `tests/integration/test_bsdb_to_pyiceberg.py` against **PyIceberg 0.12.0** and **PyArrow 25.0.1**:

| Specification Feature | Target Version | Conformance Status | Technical Notes |
| :--- | :---: | :---: | :--- |
| **Table Metadata JSON** (`v{N}.metadata.json`) | V2 | ✅ Conformant | Validated by independent PyIceberg and Trino parser loads. |
| **Manifest List** (`snap-*.avro`) | V2 | ✅ Conformant | All fields carry spec `field-id`s; manifests use absolute URIs. |
| **Data Manifest** (`*-m*.avro`) | V2 | ✅ Conformant | Maps carry `logicalType: map`; arrays carry `element-id`. |
| **Sort Orders** | V2 | ✅ Conformant | Persisted in metadata JSON (`sort-orders`, `default-sort-order-id`). |
| **Partition Spec Evolution** | V2 | ✅ Conformant | Supported via `update_spec`; preserves manifest lists. |
| **Column Statistics (NDV / Bounds)** | V2 | ✅ Conformant | `null_value_counts`, `lower_bounds`, `upper_bounds` per field ID. |
| **Nanosecond Timestamps** | V2 | ✅ Conformant | `Timestamp(Nanosecond)` round-trips without precision loss. |
| **Row Lineage** (`next-row-id`) | V3 | ✅ Conformant | `next-row-id` in metadata; `first-row-id` on snapshots. |
| **Position Deletes** | V2 | ✅ Conformant | Registered in delete manifests; applied during scan planning (used across V2 and V3 tables today). |
| **Deletion Vectors** | V3 | ✅ Written & read (format-version 3) | `Table::delete_async` on a `format_version = 3` table emits a spec-exact `deletion-vector-v1` Puffin blob and a manifest delete entry with `content = 1`, `file_format = "PUFFIN"`, `referenced_data_file`, `content_offset`, and `content_size_in_bytes`; V2 tables keep Parquet position deletes. Certified by [`tests/test_iceberg_v3_conformance.rs`](https://github.com/benolabsai/benostreamdb/blob/main/tests/test_iceberg_v3_conformance.rs) (independent Avro reader) and by `apache/iceberg-rust` 0.11 in [`tests/iceberg_rust_oracle`](https://github.com/benolabsai/benostreamdb/blob/main/tests/iceberg_rust_oracle/README.md). Merge-on-read deletes share the same codec. |
| **Compound Puffin Index Blobs** | Spec | ⚠️ Extension | Secondary index overlays (HNSW, BM25, Bitmaps) stored in `.puffin`. Standard readers ignore and fall back to Parquet scans. |
| **Staging & WAL Directories** (`_staging`, `_wal`) | Spec | ⚠️ Extension | Durability artifacts stored alongside table root; ignored by external readers. |

---

## 5. Summary & Related Guides

- **[Catalog Integration Guide](CATALOG_USAGE.md)**: Configuring production catalogs (Apache Polaris, Project Nessie, AWS Glue, Unity).
- **[Concurrency & Durability](CONCURRENCY.md)**: Multi-writer OCC, `_manifest/` commits, and distributed locking.
- **[pgvector SQL Guide](PGVECTOR_SQL_GUIDE.md)**: Querying vector embeddings in Iceberg tables using ANSI SQL and DataFusion.
- **[ADR-001: Tightened Index Overlay Contract](https://www.benostreamdb.org/guides/adr_001.html)**: Compound Puffin sidecar architecture.
