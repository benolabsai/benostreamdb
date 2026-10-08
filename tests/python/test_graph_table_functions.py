# Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

"""Functional tests for the graph traversal **table functions**
(`FROM graph_neighbors(...)` etc.).

These are the `FROM`-clause counterparts to the graph UDAFs. They follow the
same in-memory / out-of-core `GraphMode` pattern, so the tests also assert mode
invariance (`auto` == `in_memory`).
"""

import pyarrow as pa
import pytest

import benostreamdb as b


@pytest.fixture()
def session(tmp_path):
    # A line graph 0-1-2-3-4-5 (undirected). A unique URI per test avoids the
    # "table already exists" error from reusing a fixed location.
    edges = pa.table({"source": [0, 1, 2, 3, 4], "target": [1, 2, 3, 4, 5]})
    table = b.Table.from_arrow(f"file://{tmp_path}/edges", edges)
    s = b.Session()
    s.register("edges", table)
    return s


def rows(session, sql):
    return session.sql(sql).to_pandas().to_dict("records")


def test_neighbors_are_hop_and_seed_annotated(session):
    got = rows(session, "select * from graph_neighbors('edges', '0', 2, 'auto')")
    assert got == [
        {"node": 0, "hop": 0, "seed": 0},
        {"node": 1, "hop": 1, "seed": 0},
        {"node": 2, "hop": 2, "seed": 0},
    ]


def test_shortest_path_is_ordered_and_inclusive(session):
    got = rows(session, "select * from graph_shortest_path('edges', 0, 4, 'auto')")
    assert [r["node"] for r in got] == [0, 1, 2, 3, 4]
    assert [r["hop"] for r in got] == [0, 1, 2, 3, 4]


def test_all_shortest_paths(session):
    got = rows(session, "select * from graph_all_shortest_paths('edges', 0, 4, 'auto')")
    assert [list(r["path"]) for r in got] == [[0, 1, 2, 3, 4]]


def test_subgraph_edges_are_induced(session):
    got = rows(session, "select * from graph_subgraph('edges', '2', 1, 'auto')")
    assert {(r["source"], r["target"]) for r in got} == {(2, 3)}


def test_connecting_paths_union_pairwise(session):
    got = rows(session, "select * from graph_connecting_paths('edges', '0,5', 'auto')")
    assert [(r["source"], r["target"]) for r in got] == [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
    ]


def test_mode_invariance_auto_equals_in_memory(session):
    for sql in (
        "select * from graph_neighbors('edges', '0', 3, '{mode}')",
        "select * from graph_shortest_path('edges', 0, 5, '{mode}')",
        "select * from graph_subgraph('edges', '2', 2, '{mode}')",
    ):
        auto = rows(session, sql.format(mode="auto"))
        in_mem = rows(session, sql.format(mode="in_memory"))
        assert auto == in_mem, f"mode invariance broken for: {sql}"


def test_missing_table_is_a_table_error(session):
    with pytest.raises(Exception) as exc:
        session.sql("select * from graph_neighbors('__no_such_table__', '1', 1)")
    message = str(exc.value)
    assert "not found" in message and "table" in message.lower()


def test_custom_endpoint_columns(tmp_path):
    """Non-standard endpoint column names are named explicitly (the trailing
    `source`, `target` args), rather than being hard-coded to source/target."""
    edges = pa.table({"from_node": [0, 1, 2], "to_node": [1, 2, 3]})
    table = b.Table.from_arrow(f"file://{tmp_path}/custom_edges", edges)
    s = b.Session()
    s.register("custom_edges", table)

    got = rows(
        s,
        "select * from graph_neighbors('custom_edges', '0', 2, 'auto', 'from_node', 'to_node')",
    )
    assert [r["node"] for r in got] == [0, 1, 2]
