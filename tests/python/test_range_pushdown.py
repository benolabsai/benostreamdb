"""OR-over-ranges pushdown (A1.6).

A disjunction of ranges on one indexed column is used to prune whole segments
before they are read:

    WHERE (id BETWEEN 1 AND 5) OR (id BETWEEN 50 AND 55) OR (id BETWEEN 90 AND 95)

DataFusion post-filters this correctly on its own; the pushdown is a
performance optimisation, so the property that matters is that pruning never
drops a matching row and never leaks a non-matching one.
"""
import pyarrow as pa
import pytest

import hyperstreamdb as hdb


@pytest.fixture()
def table(tmp_path):
    uri = f"file://{tmp_path}/oranges"
    schema = pa.schema([("id", pa.int64()), ("v", pa.large_string())])
    t = hdb.Table.create(uri, schema)
    t.add_index("id", "inverted")
    # 10 segments of 10 rows: ids 0..99 — enough segments that pruning matters.
    for chunk in range(10):
        t.write(pa.table({
            "id": pa.array([chunk * 10 + i for i in range(10)], pa.int64()),
            "v": pa.array([f"v{chunk * 10 + i}" for i in range(10)], pa.large_string()),
        }))
        t.commit()
    t.wait_for_background_tasks()
    return t


def _ids(t, where):
    df = t.execute_sql(f"SELECT id FROM t WHERE {where} ORDER BY id").to_pandas()
    return df["id"].tolist()


def test_between_disjunction(table):
    got = _ids(
        table,
        "(id BETWEEN 1 AND 5) OR (id BETWEEN 50 AND 55) OR (id BETWEEN 90 AND 95)",
    )
    assert got == list(range(1, 6)) + list(range(50, 56)) + list(range(90, 96))


def test_comparison_disjunction(table):
    """`BETWEEN` is often lowered to an AND of two comparison bounds."""
    got = _ids(table, "(id >= 20 AND id <= 22) OR (id >= 70 AND id <= 71)")
    assert got == [20, 21, 22, 70, 71]


def test_single_range_unchanged(table):
    assert _ids(table, "id BETWEEN 30 AND 32") == [30, 31, 32]


def test_non_matching_disjunction_returns_nothing(table):
    assert _ids(table, "(id BETWEEN 500 AND 505) OR (id BETWEEN 600 AND 605)") == []


def test_mixed_columns_is_not_merged(table):
    """Ranges on different columns must not be treated as one disjunction."""
    got = _ids(table, "(id BETWEEN 1 AND 2) OR (v = 'v55')")
    assert got == [1, 2, 55]