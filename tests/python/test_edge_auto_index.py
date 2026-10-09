# Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

"""Declaring an edge table materialises its forward/reverse CSR overlays.

`table_type = 'edge'` (with `src_col`/`dst_col`) is the declarative form of
"this is a graph"; setting it must *configure* the two CSR graph indexes the
traversal fast paths need, idempotently. (The physical CSR files are built on
write/backfill, so these tests assert the configured index columns.)
"""

import pyarrow as pa

import benostreamdb


def test_create_edge_table_configures_forward_and_reverse_csr(tmp_path):
    table = benostreamdb.Table.create_edge_table(f"file://{tmp_path}/edges")
    # `create_edge_table` stamps the edge metadata, which triggers the CSR config.
    assert table.is_edge_table()
    cols = set(table.index_columns)
    assert {"source", "target"} <= cols, f"CSR not configured: {sorted(cols)}"


def test_set_property_edge_is_idempotent(tmp_path):
    schema = pa.schema([pa.field("src", pa.uint64()), pa.field("dst", pa.uint64())])
    table = benostreamdb.Table.create(f"file://{tmp_path}/e2", schema)
    table.set_properties({"table_type": "edge", "src_col": "src", "dst_col": "dst"})
    assert {"src", "dst"} <= set(table.index_columns)
    # Re-declaring must not error or duplicate.
    table.set_properties({"table_type": "edge", "src_col": "src", "dst_col": "dst"})
    assert {"src", "dst"} <= set(table.index_columns)


def test_plain_table_gets_no_graph_index(tmp_path):
    table = benostreamdb.Table.from_arrow(
        f"file://{tmp_path}/plain", pa.table({"a": [1, 2]})
    )
    assert not table.is_edge_table()
    assert "a" not in set(table.index_columns)
