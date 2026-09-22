"""Index-config persistence and no-index graceful degradation.

Two behaviours that were broken and cost a whole-site load:

1. `add_index` on a fresh table (before the first commit) only mutated
   in-memory state, so a *second process* opening the table wrote unindexed
   segments (queries then flat-scanned 98 GB). The config must persist in the
   manifest schema and be restored on open.
2. A table with no index configuration must behave like plain Iceberg:
   correct results via scan, no errors.
"""
import glob
import os

import numpy as np
import pyarrow as pa
import pytest

import hyperstreamdb as hdb

DIM = 8


def _table(uri, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, DIM), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    schema = pa.schema([
        ("id", pa.int64()),
        ("title", pa.large_string()),
        ("embedding", pa.list_(pa.float32(), DIM)),
    ])
    t = hdb.Table.create(uri, schema)
    t.write(pa.table({
        "id": pa.array(np.arange(n), pa.int64()),
        "title": pa.array([f"page {i}" for i in range(n)], pa.large_string()),
        "embedding": pa.FixedSizeListArray.from_arrays(
            pa.array(vecs.reshape(-1), pa.float32()), DIM),
    }))
    t.commit()
    t.wait_for_background_tasks()
    return t, vecs


def test_index_config_persists_and_is_inherited(tmp_path):
    """A second process/instance must inherit the index config and index its writes."""
    uri = f"file://{tmp_path}/tbl"
    t, vecs = _table(uri)
    t.add_index("embedding", "hnsw")
    t.add_index("title", "inverted")
    # force a commit so the schema (with index specs) is written
    t.write(pa.table({
        "id": pa.array([9999], pa.int64()),
        "title": pa.array(["extra"], pa.large_string()),
        "embedding": pa.FixedSizeListArray.from_arrays(
            pa.array(vecs[0].reshape(-1), pa.float32()), DIM),
    }))
    t.commit()
    t.wait_for_background_tasks()

    # Reopen: the config must come back from the manifest
    t2 = hdb.Table(uri)
    assert t2.index_all is not None  # property exists / table usable
    # Write through the reopened instance and confirm the new segment is indexed
    before = set(glob.glob(os.path.join(str(tmp_path), "tbl", "*.embedding.*")))
    t2.write(pa.table({
        "id": pa.array([10000], pa.int64()),
        "title": pa.array(["second"], pa.large_string()),
        "embedding": pa.FixedSizeListArray.from_arrays(
            pa.array(vecs[1].reshape(-1), pa.float32()), DIM),
    }))
    t2.commit()
    t2.wait_for_background_tasks()
    after = set(glob.glob(os.path.join(str(tmp_path), "tbl", "*.embedding.*")))
    assert after - before, "reopened table wrote a segment with no vector index files"


def test_no_index_table_degrades_gracefully(tmp_path):
    """No index config -> plain Iceberg behaviour: correct results, no errors."""
    uri = f"file://{tmp_path}/plain"
    t, vecs = _table(uri, n=500, seed=7)

    # scalar filter (scan path)
    df = t.execute_sql("SELECT count(*) c FROM t WHERE id < 100").to_pandas()
    assert int(df["c"].iloc[0]) == 100

    # vector search without any index -> flat scan, still correct
    res = t.vector_search("embedding", vecs[3].tolist(), k=3, columns=["id", "title"])
    assert len(res) == 3
    assert int(res["id"].iloc[0]) == 3, "nearest neighbour wrong on scan path"

    # keyword search without an inverted index must not error
    try:
        kw = t.execute_sql("SELECT count(*) c FROM t WHERE title LIKE '%page 1%'").to_pandas()
        assert int(kw["c"].iloc[0]) > 0
    except Exception as e:  # pragma: no cover - must not happen
        pytest.fail(f"plain scan query failed: {e}")