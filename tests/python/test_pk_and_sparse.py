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

import hyperstreamdb as hdb


def test_add_primary_key_and_filter(tmp_path):
    uri = f"file://{tmp_path}/pk"
    schema = pa.schema([("a", pa.int64()), ("b", pa.int64()), ("v", pa.large_string())])
    t = hdb.Table.create(uri, schema)

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
    t = hdb.Table.create(uri, schema)
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


def test_sparse_vector_accepts_lists_and_arrays():
    from_list = hdb.SparseVector([0, 5, 10], [1.0, 2.0, 3.0], 100)
    assert from_list.dim == 100
    assert list(from_list.indices) == [0, 5, 10]
    assert list(from_list.values) == [1.0, 2.0, 3.0]

    from_np = hdb.SparseVector(
        np.array([0, 5, 10], dtype=np.uint32),
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
        100,
    )
    assert list(from_np.indices) == [0, 5, 10]

    # Identical vectors -> zero L2 distance.
    assert hdb.l2_sparse(from_list, from_np) == pytest.approx(0.0)


def test_sparse_vector_validates_input():
    with pytest.raises(Exception):
        # indices not sorted
        hdb.SparseVector([5, 0], [1.0, 2.0], 10)
    with pytest.raises(Exception):
        # length mismatch
        hdb.SparseVector([0, 1], [1.0], 10)
    with pytest.raises(Exception):
        # out of bounds
        hdb.SparseVector([0, 99], [1.0, 2.0], 10)