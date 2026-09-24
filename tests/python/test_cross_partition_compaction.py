"""Cross-partition compaction and partition transforms.

Cross-partition compaction merges small files from *different* partitions into
one bin and re-partitions the merged batch. That is only correct when every
partition transform is a deterministic function of the row data, so these tests
pin down:

1. `bucket(N)` produces a bounded integer partition value (not the raw value).
2. `truncate(W)` floors integers / cuts strings.
3. Compaction across partitions preserves every row and re-partitions the
   output correctly (one file per partition).

Partition values are observed through the Hive-style data-file paths
(`category=7/seg_*.parquet`), which is where the engine materialises them.
"""
import re

import pyarrow as pa
import pytest

import benostreamdb as bsdb


def _schema():
    return pa.schema([("id", pa.int64()), ("category", pa.large_string())])


def _spec(transform, name="category"):
    return {
        "fields": [
            {"name": name, "transform": transform, "source_id": 1, "field_id": 1000}
        ]
    }


def _write(t, ids, cats):
    t.write(pa.table({
        "id": pa.array(ids, pa.int64()),
        "category": pa.array(cats, pa.large_string()),
    }))
    t.commit()


def _partition_values(t, field="category"):
    """Extract the partition value from each data file's Hive path."""
    vals = []
    for f in t._inner.list_data_files():
        m = re.search(rf"{field}=([^/]+)/", f.file_path)
        if m:
            vals.append(m.group(1))
    return vals


def test_bucket_partition_values_are_transformed(tmp_path):
    uri = f"file://{tmp_path}/t"
    t = bsdb.Table.create_partitioned(uri, _schema(), _spec("bucket(8)"))
    _write(t, [1, 2, 3], ["a", "b", "c"])

    vals = _partition_values(t)
    assert vals, "expected at least one partitioned data file"
    for v in vals:
        assert v.isdigit(), f"bucket value must be an integer, got {v!r}"
        assert 0 <= int(v) < 8, f"bucket value must be < 8, got {v}"


def test_truncate_partition_values(tmp_path):
    uri = f"file://{tmp_path}/t"
    schema = pa.schema([("id", pa.int64()), ("n", pa.int64())])
    spec = {"fields": [{"name": "n", "transform": "truncate(10)", "source_id": 2, "field_id": 1000}]}
    t = bsdb.Table.create_partitioned(uri, schema, spec)
    t.write(pa.table({"id": pa.array([1, 2], pa.int64()), "n": pa.array([123, 127], pa.int64())}))
    t.commit()

    vals = sorted(_partition_values(t, field="n"))
    assert vals == ["120"], f"123 and 127 both truncate to 120, got {vals}"


def test_cross_partition_compaction_preserves_rows_and_partitions(tmp_path):
    uri = f"file://{tmp_path}/t"
    t = bsdb.Table.create_partitioned(uri, _schema(), _spec("identity"))

    # Five tiny files spread across two partitions.
    for i in range(5):
        _write(t, [i], [f"c{i % 2}"])

    before = t.to_pandas()
    assert len(before) == 5

    # Force compaction: every tiny segment is a candidate, and the 1 GB target
    # puts all of them in a single cross-partition bin.
    t.rewrite_data_files(min_file_size_bytes=1_000_000_000)

    after = t.to_pandas()
    assert len(after) == 5, f"compaction lost rows: {len(before)} -> {len(after)}"
    assert set(after["id"].tolist()) == set(before["id"].tolist())

    # The merged bin must have been re-partitioned back into one file per
    # partition (2 distinct categories).
    files = t._inner.list_data_files()
    part_vals = set(_partition_values(t))
    assert part_vals == {"c0", "c1"}, f"unexpected partitions after compaction: {part_vals}"
    assert len(files) == 2, f"expected one file per partition, got {len(files)}"