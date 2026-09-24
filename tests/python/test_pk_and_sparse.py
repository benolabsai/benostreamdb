"""Primary-key operations and sparse-vector construction.

Two bugs fixed here:

1. `add_primary_key` panicked with "Cannot start a runtime from within a
   runtime": the async `_validate_pk_uniqueness` called the *sync*
   `read_with_columns`, which re-entered the Tokio runtime. It now uses the
   async read path.
2. `SparseVector(indices, values, dim)` rejected plain Python lists (it only
   accepted NumPy arrays), contradicting its own docstring.
"""
import numpy as np
import pyarrow as pa
import pytest

import benostreamdb as bsdb


def test_add_primary_key_and_filter(tmp_path):
    uri = f"file://{tmp_path}/pk"
    schema = pa.schema([("a", pa.int64()), ("b", pa.int64()), ("v", pa.large_string())])
    t = bsdb.Table.create(uri, schema)

    # Must not panic.
    t.add_primary_key("a")
    assert t.primary_key == ["a"]

    t.write(pa.table({
        "a": pa.array([1, 2, 3], pa.int64()),
        "b": pa.array([1, 1, 1], pa.int64()),
        "v": pa.array(["p", "q", "r"], pa.large_string()),
    }))
    t.commit()

    df = t.execute_sql("SELECT * FROM t WHERE (a, b) IN ((1,1),(3,1))").to_pandas()
    assert sorted(df["a"].tolist()) == [1, 3]


def test_primary_key_rejects_duplicates(tmp_path):
    uri = f"file://{tmp_path}/pk_dup"
    schema = pa.schema([("a", pa.int64()), ("v", pa.large_string())])
    t = bsdb.Table.create(uri, schema)
    t.add_primary_key("a")
    t.write(pa.table({
        "a": pa.array([1, 2], pa.int64()),
        "v": pa.array(["x", "y"], pa.large_string()),
    }))
    t.commit()

    with pytest.raises(Exception):
        t.write(pa.table({
            "a": pa.array([1], pa.int64()),
            "v": pa.array(["dup"], pa.large_string()),
        }))
        t.commit()


def test_row_value_in_list_read_path(tmp_path):
    """Row-value IN-list over an indexed composite PK is pushed to the scan.

    With every PK column inverted-indexed, `scan` prunes segments that cannot
    match before reading them. This asserts correctness of that path (and, by
    construction, that pruning never drops a matching row).
    """
    uri = f"file://{tmp_path}/pkread"
    schema = pa.schema([("a", pa.int64()), ("b", pa.int64()), ("v", pa.large_string())])
    t = bsdb.Table.create(uri, schema)
    t.add_primary_key("a")
    t.add_index("a", "inverted")
    t.add_index("b", "inverted")

    # Several segments so there is something to prune.
    for chunk in range(5):
        t.write(pa.table({
            "a": pa.array([chunk * 10 + i for i in range(10)], pa.int64()),
            "b": pa.array([1] * 10, pa.int64()),
            "v": pa.array([f"v{chunk * 10 + i}" for i in range(10)], pa.large_string()),
        }))
        t.commit()
    t.wait_for_background_tasks()

    df = t.execute_sql("SELECT a, b FROM t WHERE (a, b) IN ((1,1),(23,1))").to_pandas()
    assert sorted(df["a"].tolist()) == [1, 23], df

    # A tuple that matches nothing must return no rows (not a false prune of
    # everything, and not a leak of unrelated rows).
    empty = t.execute_sql("SELECT a FROM t WHERE (a, b) IN ((999,999))").to_pandas()
    assert empty.empty


def test_sparse_vector_accepts_lists_and_arrays():
    from_list = bsdb.SparseVector([0, 5, 10], [1.0, 2.0, 3.0], 100)
    assert from_list.dim == 100
    assert list(from_list.indices) == [0, 5, 10]
    assert list(from_list.values) == [1.0, 2.0, 3.0]

    from_np = bsdb.SparseVector(
        np.array([0, 5, 10], dtype=np.uint32),
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
        100,
    )
    assert list(from_np.indices) == [0, 5, 10]

    # Identical vectors -> zero L2 distance.
    assert bsdb.l2_sparse(from_list, from_np) == pytest.approx(0.0)


def test_sparse_vector_validates_input():
    with pytest.raises(Exception):
        # indices not sorted
        bsdb.SparseVector([5, 0], [1.0, 2.0], 10)
    with pytest.raises(Exception):
        # length mismatch
        bsdb.SparseVector([0, 1], [1.0], 10)
    with pytest.raises(Exception):
        # out of bounds
        bsdb.SparseVector([0, 99], [1.0, 2.0], 10)