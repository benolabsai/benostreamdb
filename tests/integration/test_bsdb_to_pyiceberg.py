"""
WS4: BenoStreamDB -> PyIceberg interoperability.

Validates that a table written by BenoStreamDB can be read by an *independent*
Iceberg implementation (PyIceberg), not just by BenoStreamDB itself. This is the
reverse direction of `test_pyiceberg_compat.py` (which reads a PyIceberg table
with BenoStreamDB).

Requires: pyiceberg + pyarrow (installed in the project `.venv`).

Run with:
    .venv/bin/python -m pytest tests/integration/test_bsdb_to_pyiceberg.py -v
"""

import glob
import os
import tempfile

import pyarrow as pa
import pytest

import benostreamdb as bsdb

pyiceberg = pytest.importorskip("pyiceberg")
from pyiceberg.table import StaticTable  # noqa: E402


def _latest_metadata(table_path: str) -> str:
    mdir = os.path.join(table_path, "metadata")
    files = sorted(
        f for f in os.listdir(mdir) if f.endswith(".metadata.json")
    )
    assert files, f"no metadata JSON written under {mdir}"
    return os.path.join(mdir, files[-1])


def test_bsdb_table_is_readable_by_pyiceberg():
    """BenoStreamDB -> PyIceberg: schema + data must round-trip."""
    tmp = tempfile.mkdtemp(prefix="bsdb_pyiceberg_")
    table_path = os.path.join(tmp, "t")
    uri = f"file://{table_path}"

    data = pa.table(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["alpha", "beta", "gamma", "delta", "epsilon"],
            "value": [1.5, 2.5, 3.5, 4.5, 5.5],
        }
    )
    bsdb.Table.from_arrow(uri, data)

    metadata_path = _latest_metadata(table_path)

    # PyIceberg loads the table purely from the on-disk metadata/manifests.
    ice = StaticTable.from_metadata(metadata_path)
    arrow = ice.scan().to_arrow()

    assert arrow.num_rows == 5, f"expected 5 rows, got {arrow.num_rows}"
    assert arrow.schema.names == ["id", "name", "value"], arrow.schema
    assert arrow.column("id").to_pylist() == [1, 2, 3, 4, 5]
    assert arrow.column("name").to_pylist() == [
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
    ]
    assert arrow.column("value").to_pylist() == [1.5, 2.5, 3.5, 4.5, 5.5]


def test_bsdb_multi_commit_manifest_list_is_readable():
    """Multiple commits -> the manifest list must remain readable by PyIceberg."""
    tmp = tempfile.mkdtemp(prefix="bsdb_pyiceberg_multi_")
    table_path = os.path.join(tmp, "t")
    uri = f"file://{table_path}"

    # Two separate commits (each adds a segment + a manifest list).
    bsdb.Table.from_arrow(uri, pa.table({"id": [1, 2, 3]}))
    t = bsdb.Table(uri)
    t.write(pa.table({"id": [4, 5, 6]}))
    t.commit()

    metadata_path = _latest_metadata(table_path)
    ice = StaticTable.from_metadata(metadata_path)
    arrow = ice.scan().to_arrow()
    ids = sorted(arrow.column("id").to_pylist())
    assert ids == [1, 2, 3, 4, 5, 6], f"ids mismatch: {ids}"


if __name__ == "__main__":
    test_bsdb_table_is_readable_by_pyiceberg()
    test_bsdb_multi_commit_manifest_list_is_readable()
    print("=== BenoStreamDB -> PyIceberg interop OK ===")
