"""MVCC manifest commits: snapshot versions and concurrent writers.

Covers the lock-free commit path:

1. `snapshot_version()` is a monotonically increasing snapshot id.
2. Two independent writer handles appending concurrently must not lose
   updates (the OCC retry loop rebases onto the latest snapshot).
3. Schema evolution (which previously took a global `commit.lock`) must not
   clobber a concurrent append.
"""
import pyarrow as pa
import pytest

import hyperstreamdb as hdb


def _schema():
    return pa.schema([("id", pa.int64()), ("val", pa.large_string())])


def _batch(ids):
    return pa.table({
        "id": pa.array(ids, pa.int64()),
        "val": pa.array([f"v{i}" for i in ids], pa.large_string()),
    })


def test_snapshot_version_is_monotonic(tmp_path):
    uri = f"file://{tmp_path}/t"
    t = hdb.Table.create(uri, _schema())
    t.write(_batch([0]))
    t.commit()
    v1 = t.snapshot_version()

    t.write(_batch([1]))
    t.commit()
    v2 = t.snapshot_version()

    t.write(_batch([2]))
    t.commit()
    v3 = t.snapshot_version()

    assert v1 < v2 < v3, f"snapshot versions must increase: {v1}, {v2}, {v3}"


def test_concurrent_writers_do_not_lose_updates(tmp_path):
    uri = f"file://{tmp_path}/t"
    t = hdb.Table.create(uri, _schema())
    t.write(_batch([0]))
    t.commit()

    # Two independent handles == two writers racing on the same table.
    a = hdb.Table(uri)
    b = hdb.Table(uri)
    a.write(_batch([1]))
    b.write(_batch([2]))
    a.commit()
    b.commit()

    ids = set(hdb.Table(uri).to_pandas()["id"].tolist())
    assert {0, 1, 2} <= ids, f"lost an update under concurrency: {sorted(ids)}"


def test_schema_evolution_does_not_clobber_concurrent_append(tmp_path):
    """`update_schema` is now pure OCC (no global commit.lock)."""
    uri = f"file://{tmp_path}/t"
    t = hdb.Table.create(uri, _schema())
    t.write(_batch([0]))
    t.commit()

    writer = hdb.Table(uri)
    writer.write(_batch([1]))
    writer.commit()

    # Schema evolution on the original handle.
    t.add_column("extra", pa.int64())

    ids = set(hdb.Table(uri).to_pandas()["id"].tolist())
    assert {0, 1} <= ids, f"schema evolution clobbered a concurrent append: {sorted(ids)}"