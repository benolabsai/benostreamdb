# Iceberg V2/V3 API Documentation

## Table of Contents
- [Sort Orders (V2)](#sort-orders-v2)
- [Partition Spec Evolution (V2)](#partition-spec-evolution-v2)
- [Statistics with NDV (V2)](#statistics-with-ndv-v2)
- [Row Lineage (V3)](#row-lineage-v3)
- [Default Values (V3)](#default-values-v3)

---

## Sort Orders (V2)

Control the physical layout of data for optimal query performance.

### API

```python
import hyperstreamdb as hdb

table = hdb.Table("s3://bucket/table")

# Set sort order (applied during writes)
table.replace_sort_order(
    columns=["timestamp", "user_id"],
    ascending=[False, True]  # timestamp DESC, user_id ASC
)

# Get current sort order
# sort_order = table.get_sort_order() 
# print(sort_order)
```

### Benefits
- **Range Pruning**: Sorted data enables efficient min/max filtering
- **Compression**: Better compression ratios for sorted columns
- **Query Performance**: Faster scans when filtering on sort columns

---

## Partition Spec Evolution (V2)

Change partitioning strategy without rewriting data.

### API

```python
from hyperstreamdb import PartitionField

# Initial partition spec
table.update_spec([
    PartitionField(
        source_ids=[1],   # Column ID(s)
        name="date",
        transform="day"   # day, month, year, hour, bucket, truncate
    )
])

# Evolve to add new partition
table.update_spec([
    PartitionField(source_ids=[1], name="date", transform="day"),
    PartitionField(source_ids=[2], name="region", transform="identity")
])
```

### Supported Transforms
- `identity`: No transformation
- `year`, `month`, `day`, `hour`: Temporal partitioning
- `bucket(N)`: Hash bucketing with N buckets
- `truncate(W)`: Truncate strings/numbers to width W

---

## Statistics with NDV (V2)

Distinct count estimation using HyperLogLog for query optimization.

### Implementation

Statistics are automatically computed during writes:

```python
# NDV is computed transparently
table.write_pandas(df)

# Statistics include:
# - min/max values
# - null_count
# - distinct_count (HyperLogLog, ~1% error)
```

### Technical Details
- **Algorithm**: HyperLogLogPlus with precision 14
- **Memory**: O(1) space (~1.5KB per column)
- **Accuracy**: ~1-2% error rate
- **Types**: Int32, Int64, Utf8

---

## Row Lineage (V3)

Track individual row identity and update history.

### Automatic Metadata Columns

When `format_version >= 3`, two metadata columns are automatically added:

| Column | Type | Description |
|--------|------|-------------|
| `_row_id` | String | UUID v4 unique identifier |
| `_last_updated_sequence_number` | Int64 | Manifest version when row was written |

### Usage

```python
# V3 tables automatically include row lineage
table = hdb.Table("s3://bucket/v3-table")
table.write_pandas(df)

# Query with metadata columns
result = table.to_pandas()
print(result.columns)
# ['id', 'name', '_row_id', '_last_updated_sequence_number']

# Filter by sequence number (time travel)
recent = table.to_pandas(filter="_last_updated_sequence_number > 100")
```

### Benefits
- **Change Data Capture**: Track which rows changed
- **Deduplication**: Use `_row_id` for exact row matching
- **Time Travel**: Query data as of specific sequence numbers

---

## Default Values (V3)

Define default values for schema evolution.

### Schema Fields

```python
# Default values are stored in schema metadata
# initial_default: Value for existing rows when column is added
# write_default: Value for new rows when column is null
```

### Example

```python
# When adding a new column to existing table:
# - Existing rows get initial_default
# - New rows with null get write_default
```

---

## Migration Guide: V2 → V3

### Upgrading Tables

1. **No Breaking Changes**: V3 is backward compatible
2. **Automatic Metadata**: `_row_id` and `_last_updated_sequence_number` added transparently
3. **No Data Rewrite**: Existing data files remain unchanged

### Compatibility Matrix

| Writer Version | Reader Version | Compatible |
|----------------|----------------|------------|
| V2 | V2 | ✅ Yes |
| V2 | V3 | ✅ Yes |
| V3 | V2 | ⚠️ Metadata columns ignored |
| V3 | V3 | ✅ Yes |

---

## Performance Characteristics

### Sort Orders
- **Write Overhead**: ~5-10% (sorting cost)
- **Read Speedup**: 2-10x for range queries on sorted columns

### HyperLogLog NDV
- **Memory**: 1.5KB per column (vs MB-GB for exact counting)
- **Accuracy**: 98-99% (acceptable for query optimization)
- **Computation**: ~10% write overhead

### V3 Row Lineage
- **Storage Overhead**: ~40 bytes per row (UUID + i64)
- **Write Overhead**: <1% (UUID generation)
- **Query Impact**: None (columns are optional in queries)

---

## Examples

### Complete V2/V3 Workflow

```python
import hyperstreamdb as hdb
import pandas as pd

# Create table with V2 features
table = hdb.Table("s3://bucket/analytics")

# Configure sort order for time-series data
table.replace_sort_order(
    columns=["event_time", "user_id"],
    ascending=[False, True]
)

# Set partition spec
table.update_spec([
    PartitionField(source_ids=[1], name="date", transform="day")
])

# Write data (V3 row lineage added automatically)
df = pd.DataFrame({
    "event_time": pd.date_range("2024-01-01", periods=1000),
    "user_id": range(1000),
    "action": ["click"] * 1000
})
table.write_pandas(df)

# Run maintenance/optimization (Standard name)
table.rewrite_data_files()

# Query with automatic index usage
recent_clicks = table.to_pandas(
    filter="event_time > '2024-01-15' AND user_id < 100"
)

# Access V3 metadata
print(recent_clicks[["_row_id", "_last_updated_sequence_number"]].head())

# Time Travel: Rollback to previous state
table.rollback_to_snapshot(123456789)
```

