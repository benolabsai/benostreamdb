import os
import shutil
import tempfile
import pyarrow as pa
import pytest
from pyiceberg.catalog.sql import SqlCatalog
import benostreamdb as bsdb

def test_pyiceberg_transformation_and_detached_sync():
    tmp = tempfile.mkdtemp(prefix="bsdb_test_detached_")
    warehouse = os.path.join(tmp, "warehouse")
    catalog_db = os.path.join(tmp, "catalog.db")
    overlay_dir = os.path.join(tmp, "overlay")
    os.makedirs(warehouse, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    try:
        # 1. Setup PyIceberg SQL catalog
        catalog = SqlCatalog(
            "test_catalog",
            uri=f"sqlite:///{catalog_db}",
            warehouse=f"file://{warehouse}",
        )
        catalog.create_namespace("default")

        schema = pa.schema([
            ("id", pa.int64()),
            ("title", pa.string()),
            ("category", pa.string()),
        ])

        table_name = "default.articles"
        iceberg_table = catalog.create_table(table_name, schema=schema)

        # Append batch 1 via PyIceberg
        batch1 = pa.Table.from_pydict({
            "id": [1, 2, 3],
            "title": ["Database Design Principles", "Vector Search with HNSW", "Graph Analytics with CSR"],
            "category": ["database", "vector", "graph"],
        })
        iceberg_table.append(batch1)
        snap1_id = iceberg_table.current_snapshot().snapshot_id
        assert snap1_id is not None

        # 2. Mount in BenoStreamDB
        bsdb_table = bsdb.Table.mount_external_iceberg(
            table_uri=f"file://{warehouse}/default/articles",
            overlay_storage_uri=f"file://{overlay_dir}"
        )

        # Declare primary key
        bsdb_table.set_primary_key(["id"])
        assert bsdb_table.get_primary_key() == ["id"]

        # Declare BM25 index
        bsdb_table.add_index("title", "bm25")

        # Query initial data via SQL
        q1 = bsdb_table.sql("SELECT id, title, category FROM t WHERE id = 2")
        assert q1.num_rows == 1
        assert q1.column("title")[0].as_py() == "Vector Search with HNSW"

        # 3. External transformation via PyIceberg (Append batch 2)
        batch2 = pa.Table.from_pydict({
            "id": [4, 5],
            "title": ["Lakehouse Indexing Overlays", "Autonomous AI Agents"],
            "category": ["lakehouse", "ai"],
        })
        iceberg_table.append(batch2)
        snap2_id = iceberg_table.current_snapshot().snapshot_id
        assert snap2_id != snap1_id

        # 4. Sync & reconcile overlay in BenoStreamDB
        synced = bsdb_table.sync_detached_overlay()
        assert synced is True

        # 5. Verify query results over full data
        full_data = bsdb_table.to_arrow()
        assert full_data.num_rows == 5

        # Query new row added in PyIceberg
        q2 = bsdb_table.sql("SELECT id, title, category FROM t WHERE id = 5")
        assert q2.num_rows == 1
        assert q2.column("title")[0].as_py() == "Autonomous AI Agents"

    finally:
        shutil.rmtree(tmp, ignore_errors=True)
