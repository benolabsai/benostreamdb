"""EXPLAIN reports *why* segments were pruned.

`explain()` used to say only "N segments pruned via Partition/Stats mapping",
which tells you nothing actionable. It now names the rule that ruled each
segment out.

Note: only the partition path is exercised here. Stats-based pruning is
currently inert because written segments carry empty `column_stats`, so
`Stats*` reasons cannot fire yet (tracked separately).
"""
import pyarrow as pa
import pytest

import hyperstreamdb as hdb


@pytest.fixture()
def partitioned(tmp_path):
    uri = f"file://{tmp_path}/part"
    schema = pa.schema([("id", pa.int64()), ("cat", pa.large_string())])
    spec = {
        "fields": [
            {"name": "cat", "transform": "identity", "source_id": 2, "field_id": 1000}
        ]
    }
    t = hdb.Table.create_partitioned(uri, schema, spec)
    for c in ["a", "b", "c"]:
        t.write(pa.table({
            "id": pa.array([1, 2], pa.int64()),
            "cat": pa.array([c, c], pa.large_string()),
        }))
        t.commit()
    t.wait_for_background_tasks()
    return t


def _plan(t, filter_str):
    # `Table.explain` (the bool constructor flag) shadows the method on the
    # Python wrapper, so reach the engine method directly.
    return t._inner.explain(filter_str, None)


def test_explain_names_the_pruning_rule(partitioned):
    plan = _plan(partitioned, "cat = 'a'")
    assert "Pruning Activity" in plan
    # Two of the three partitions are ruled out, and the plan says why.
    assert "2 segment(s)" in plan
    assert "partition value" in plan


@pytest.fixture()
def ranged(tmp_path):
    """Five segments with disjoint id ranges, no partitioning involved."""
    uri = f"file://{tmp_path}/ranged"
    t = hdb.Table.create(uri, pa.schema([("id", pa.int64()), ("v", pa.large_string())]))
    for c in range(5):
        t.write(pa.table({
            "id": pa.array([c * 10 + i for i in range(10)], pa.int64()),
            "v": pa.array([f"v{c * 10 + i}" for i in range(10)], pa.large_string()),
        }))
        t.commit()
    t.wait_for_background_tasks()
    return t


def test_stats_pruning_fires_on_column_statistics(ranged):
    """Column min/max must actually reach the manifest and prune segments.

    This was inert for a long time: parquet statistics were disabled, the
    manifest writer hardcoded the bounds to null, and the reader failed to
    unwrap them from Avro's nullable-union wrapper. Each of those alone was
    enough to leave `column_stats` empty.
    """
    plan = _plan(ranged, "id >= 100")
    assert "Pruning Activity" in plan
    assert "column max < filter min" in plan
    assert "Execution Scope: 0 rows in 0 segments" in plan


def test_stats_pruning_keeps_matching_segment(ranged):
    plan = _plan(ranged, "id >= 40")
    assert "column max < filter min" in plan
    # Only the 40-49 segment survives.
    assert "Execution Scope: 10 rows in 1 segments" in plan
    # ...and the query still returns the right answer.
    got = ranged.execute_sql("SELECT count(*) c FROM t WHERE id >= 45").to_pandas()["c"].tolist()
    assert got == [5]


def test_stats_pruning_on_equality(ranged):
    plan = _plan(ranged, "id = 15")
    assert "column max < filter min" in plan or "column min > filter max" in plan
    assert "Execution Scope: 10 rows in 1 segments" in plan


def test_explain_without_pruning_has_no_breakdown(partitioned):
    # A predicate that matches every partition prunes nothing, so there is no
    # reason breakdown to print.
    plan = _plan(partitioned, "id >= 0")
    assert "Pruning Activity" not in plan
    assert "Execution Scope: 6 rows in 3 segments" in plan
