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


def test_explain_without_pruning_has_no_breakdown(partitioned):
    # A predicate that matches every partition prunes nothing, so there is no
    # reason breakdown to print.
    plan = _plan(partitioned, "id >= 0")
    assert "Pruning Activity" not in plan
    assert "Execution Scope: 6 rows in 3 segments" in plan
