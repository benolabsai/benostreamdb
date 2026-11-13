# Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

"""Python catalog binding tests that need no external service.

`memory` and `jdbc` (SQLite in-memory) are self-contained. Networked catalogs are
covered against a real server in `test_catalog_real_services.py` rather than an
embedded HTTP mock: a mock can drift from the catalog's actual protocol and then
happily pass while the client is wrong (or fail while it is right).
"""

import pyarrow as pa
import pytest

import benostreamdb as bsdb


def _schema() -> bsdb.Schema:
    return bsdb.Schema.from_arrow(
        pa.schema(
            [
                pa.field("id", pa.int32(), nullable=False),
                pa.field("label", pa.string(), nullable=False),
            ]
        )
    )


def test_memory_catalog_python_roundtrip(tmp_path):
    warehouse = f"file://{tmp_path}/wh"
    cat = bsdb.create_catalog(
        "memory", {"uri": f"memory://py_test_{tmp_path.name}", "warehouse": warehouse}
    )
    assert isinstance(cat, bsdb.PyMemoryCatalog)

    cat.create_namespace("analytics")
    assert cat.namespace_exists("analytics") is True
    assert cat.list_tables("analytics") == []

    cat.create_table("analytics", "events", _schema())
    assert cat.table_exists("analytics", "events") is True
    assert cat.list_tables("analytics") == ["events"]

    tbl = cat.load_table("analytics", "events")
    tbl.write(pa.table({"id": [1, 2, 3], "label": ["a", "b", "c"]}))
    tbl.commit()

    tbl2 = cat.load_table("analytics", "events")
    res = tbl2.sql("SELECT COUNT(*) AS cnt FROM t")
    assert res["cnt"].to_pylist() == [3]

    with pytest.raises(RuntimeError):
        cat.drop_namespace("analytics")

    cat.drop_table("analytics", "events")
    assert cat.table_exists("analytics", "events") is False
    cat.drop_namespace("analytics")
    assert cat.namespace_exists("analytics") is False


def test_jdbc_sqlite_catalog_python_roundtrip(tmp_path):
    warehouse = f"file://{tmp_path}/wh"
    cat = bsdb.create_catalog(
        "jdbc",
        {
            "uri": "sqlite::memory:",
            "warehouse": warehouse,
            "catalog_name": "py_jdbc",
        },
    )

    cat.create_namespace("prod")
    assert cat.namespace_exists("prod") is True

    cat.create_table("prod", "metrics", _schema())
    assert cat.table_exists("prod", "metrics") is True
    assert cat.list_tables("prod") == ["metrics"]

    tbl = cat.load_table("prod", "metrics")
    tbl.write(pa.table({"id": [10, 20], "label": ["x", "y"]}))
    tbl.commit()

    tbl2 = cat.load_table("prod", "metrics")
    res = tbl2.sql("SELECT SUM(id) AS total FROM t")
    assert res["total"].to_pylist() == [30]

    cat.drop_table("prod", "metrics")
    assert cat.table_exists("prod", "metrics") is False
    cat.drop_namespace("prod")
    assert cat.namespace_exists("prod") is False
