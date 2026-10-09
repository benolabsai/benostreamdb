# Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

"""Table-type metadata: the declarative node/edge-table convention.

A table can declare its role and endpoints in its properties (`table_type`,
`src_col`, `dst_col`, `id_col`, …) instead of relying on column-name
auto-detection. This is what lets the graph functions resolve a table's
endpoints from metadata, and what an MCP agent uses to discover the graph
(`Session.list_graph_tables`).
"""

import pyarrow as pa
import pytest

import benostreamdb as b


def test_create_edge_table_stamps_metadata(tmp_path):
    t = b.Table.create_edge_table(f"file://{tmp_path}/edges")
    assert t.table_type() == "edge"
    assert t.is_edge_table()
    assert t.edge_endpoints() == ("source", "target")
    meta = t.graph_metadata()
    assert meta["table_type"] == "edge"
    assert meta["source_column"] == "source"
    assert meta["target_column"] == "target"
    assert meta["relation_column"] == "relation"
    assert meta["weight_column"] == "weight"


def test_create_node_table_stamps_metadata(tmp_path):
    n = b.Table.create_node_table(f"file://{tmp_path}/nodes")
    assert n.table_type() == "node"
    meta = n.graph_metadata()
    assert meta["id_column"] == "id"
    assert meta["label_column"] == "name"


def test_set_property_persists(tmp_path):
    t = b.Table.from_arrow(f"file://{tmp_path}/t", pa.table({"a": [1]}))
    t.set_property("table_type", "edge")
    t.set_property("src_col", "a")
    props = t.properties()
    assert props["table_type"] == "edge"
    assert props["src_col"] == "a"
    assert t.table_type() == "edge"
    # Internal keys (e.g. the WAL commit marker) must survive a property set.
    assert any(k.startswith("benostream.") for k in props)


def test_alter_table_set_tblproperties(tmp_path):
    t = b.Table.from_arrow(
        f"file://{tmp_path}/t", pa.table({"source": [0], "target": [1]})
    )
    s = b.Session()
    s.register("t", t)
    s.sql(
        "ALTER TABLE t SET TBLPROPERTIES "
        "('table_type'='edge', 'src_col'='source', 'dst_col'='target')"
    )
    assert t.table_type() == "edge"
    assert t.edge_endpoints() == ("source", "target")


def test_metadata_first_endpoint_resolution(tmp_path):
    """Columns named `u`/`v` (non-standard); the metadata names them, so the
    graph functions resolve the endpoints without being told."""
    t = b.Table.from_arrow(
        f"file://{tmp_path}/edges", pa.table({"u": [0, 1, 2], "v": [1, 2, 3]})
    )
    t.set_properties({"table_type": "edge", "src_col": "u", "dst_col": "v"})
    s = b.Session()
    s.register("edges", t)
    got = (
        s.sql("select * from graph_neighbors('edges', '0', 2, 'auto')")
        .to_pandas()
        .to_dict("records")
    )
    assert [r["node"] for r in got] == [0, 1, 2]


def test_list_graph_tables(tmp_path):
    e = b.Table.create_edge_table(f"file://{tmp_path}/edges")
    n = b.Table.create_node_table(f"file://{tmp_path}/nodes")
    p = b.Table.from_arrow(f"file://{tmp_path}/plain", pa.table({"x": [1]}))
    s = b.Session()
    s.register("edges", e)
    s.register("nodes", n)
    s.register("plain", p)
    got = {t["name"].split(".")[-1]: t["table_type"] for t in s.list_graph_tables()}
    assert got == {"edges": "edge", "nodes": "node"}
