import os
import random
import shutil
import tempfile
import pyarrow as pa
import pytest
from pyiceberg.catalog.sql import SqlCatalog
import benostreamdb as bsdb

CATEGORIES = ["database", "vector", "graph", "lakehouse", "search", "streaming"]
WORDS = [
    "indexing", "compaction", "snapshot", "arrow", "parquet",
    "retrieval", "clustering", "overlay", "lineage", "durability"
]

def _make_title(rng, id_val):
    w1 = rng.choice(WORDS)
    w2 = rng.choice(WORDS)
    return f"Doc {id_val} about {w1} and {w2}"

def test_pyiceberg_transformation_and_detached_sync():
    """Verify that external PyIceberg tables with rich types map correctly,

    and that after external mutations and sync, queries find and use the synced index.
    """
    tmp = tempfile.mkdtemp(prefix="bsdb_test_detached_")
    warehouse = os.path.join(tmp, "warehouse")
    catalog_db = os.path.join(tmp, "catalog.db")
    overlay_dir = os.path.join(tmp, "overlay")
    os.makedirs(warehouse, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    try:
        # 1. Setup PyIceberg SQL catalog with rich schema (primitives, floats, vectors, booleans)
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
            ("embedding", pa.list_(pa.float32())),
            ("score", pa.float64()),
            ("active", pa.bool_()),
        ])

        table_name = "default.articles"
        iceberg_table = catalog.create_table(table_name, schema=schema)

        # Append batch 1 via PyIceberg
        batch1 = pa.Table.from_pydict({
            "id": [1, 2, 3],
            "title": ["Database Design Principles", "Vector Search with HNSW", "Graph Analytics with CSR"],
            "category": ["database", "vector", "graph"],
            "embedding": [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]],
            "score": [95.5, 98.0, 89.2],
            "active": [True, True, False],
        }, schema=schema)
        iceberg_table.append(batch1)
        snap1_id = iceberg_table.current_snapshot().snapshot_id
        assert snap1_id is not None

        # 2. Mount in BenoStreamDB - verify schema maps all complex types cleanly
        bsdb_table = bsdb.Table.mount_external_iceberg(
            table_uri=f"file://{warehouse}/default/articles",
            overlay_storage_uri=f"file://{overlay_dir}"
        )

        # Declare primary key
        bsdb_table.set_primary_key(["id"])
        assert bsdb_table.get_primary_key() == ["id"]

        # Declare BM25 index on title
        bsdb_table.add_index("title", "bm25")
        bsdb_table.wait_for_background_tasks()

        # Query initial data via SQL
        q1 = bsdb_table.sql("SELECT id, title, category, score, active FROM t WHERE id = 2")
        assert q1.num_rows == 1
        assert q1.column("title")[0].as_py() == "Vector Search with HNSW"
        assert q1.column("score")[0].as_py() == 98.0
        assert q1.column("active")[0].as_py() is True

        # 3. External transformation via PyIceberg (Append batch 2)
        batch2 = pa.Table.from_pydict({
            "id": [4, 5],
            "title": ["Lakehouse Indexing Overlays", "Autonomous AI Agents"],
            "category": ["lakehouse", "ai"],
            "embedding": [[0.2, 0.4, 0.6, 0.8], [0.3, 0.6, 0.9, 1.2]],
            "score": [99.1, 97.4],
            "active": [True, True],
        }, schema=schema)
        iceberg_table.append(batch2)
        snap2_id = iceberg_table.current_snapshot().snapshot_id
        assert snap2_id != snap1_id

        # 4. Sync & reconcile overlay in BenoStreamDB
        synced = bsdb_table.sync_detached_overlay()
        assert synced is True
        bsdb_table.wait_for_background_tasks()

        # 5. Verify query results over full data
        full_data = bsdb_table.to_arrow()
        assert full_data.num_rows == 5

        # Query new row added in PyIceberg
        q2 = bsdb_table.sql("SELECT id, title, category, score, active FROM t WHERE id = 5")
        assert q2.num_rows == 1
        assert q2.column("title")[0].as_py() == "Autonomous AI Agents"
        assert q2.column("score")[0].as_py() == 97.4

        # 6. CRITICAL VERIFICATION: Explain must find and use the synced index
        plan = bsdb_table.explain("title = 'Autonomous AI Agents'")
        assert ("Inverted Index" in plan or "BM25" in plan), (
            f"Query plan must use the synced index, got:\n{plan}"
        )
        assert "access: Full Scan" not in plan, (
            f"Query plan must not fall back to full scan when index is synced, got:\n{plan}"
        )

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.mark.parametrize("seed", [42, 123])
def test_pyiceberg_randomized_actions_workload(seed):
    """Randomized differential workload comparing external PyIceberg writes + BenoStreamDB detached syncs

    against an in-memory ground-truth oracle model, verifying synced index usage throughout.
    """
    rng = random.Random(seed)
    steps = 20

    tmp = tempfile.mkdtemp(prefix=f"bsdb_rand_test_{seed}_")
    warehouse = os.path.join(tmp, "warehouse")
    catalog_db = os.path.join(tmp, "catalog.db")
    overlay_dir = os.path.join(tmp, "overlay")
    os.makedirs(warehouse, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    try:
        catalog = SqlCatalog(
            f"rand_catalog_{seed}",
            uri=f"sqlite:///{catalog_db}",
            warehouse=f"file://{warehouse}",
        )
        catalog.create_namespace("default")

        schema = pa.schema([
            ("id", pa.int64()),
            ("title", pa.string()),
            ("category", pa.string()),
            ("score", pa.float64()),
            ("active", pa.bool_()),
        ])

        table_name = "default.random_articles"
        iceberg_table = catalog.create_table(table_name, schema=schema)

        # Ground truth in-memory oracle model
        model = {}
        next_id = 1
        pending_upstream_commits = 0

        # Initial seed append in PyIceberg
        initial_count = 3
        init_ids = list(range(next_id, next_id + initial_count))
        next_id += initial_count
        init_titles = [_make_title(rng, i) for i in init_ids]
        init_cats = [rng.choice(CATEGORIES) for _ in init_ids]
        init_scores = [round(rng.uniform(50.0, 100.0), 1) for _ in init_ids]
        init_actives = [rng.choice([True, False]) for _ in init_ids]

        for i, t, c, s, a in zip(init_ids, init_titles, init_cats, init_scores, init_actives):
            model[i] = {"id": i, "title": t, "category": c, "score": s, "active": a}

        batch0 = pa.Table.from_pydict({
            "id": init_ids,
            "title": init_titles,
            "category": init_cats,
            "score": init_scores,
            "active": init_actives,
        }, schema=schema)
        iceberg_table.append(batch0)

        # Mount in BenoStreamDB
        bsdb_table = bsdb.Table.mount_external_iceberg(
            table_uri=f"file://{warehouse}/default/random_articles",
            overlay_storage_uri=f"file://{overlay_dir}"
        )
        bsdb_table.set_primary_key(["id"])
        bsdb_table.add_index("title", "bm25")
        bsdb_table.wait_for_background_tasks()

        for step in range(1, steps + 1):
            action = rng.choice(["append", "append", "sync", "sync_and_assert", "assert"])

            if action == "append":
                batch_size = rng.randint(1, 5)
                batch_ids = list(range(next_id, next_id + batch_size))
                next_id += batch_size
                batch_titles = [_make_title(rng, i) for i in batch_ids]
                batch_cats = [rng.choice(CATEGORIES) for _ in batch_ids]
                batch_scores = [round(rng.uniform(50.0, 100.0), 1) for _ in batch_ids]
                batch_actives = [rng.choice([True, False]) for _ in batch_ids]

                for i, t, c, s, a in zip(batch_ids, batch_titles, batch_cats, batch_scores, batch_actives):
                    model[i] = {"id": i, "title": t, "category": c, "score": s, "active": a}

                batch = pa.Table.from_pydict({
                    "id": batch_ids,
                    "title": batch_titles,
                    "category": batch_cats,
                    "score": batch_scores,
                    "active": batch_actives,
                }, schema=schema)
                iceberg_table.append(batch)
                pending_upstream_commits += 1

            elif action in ("sync", "sync_and_assert"):
                synced = bsdb_table.sync_detached_overlay()
                if pending_upstream_commits > 0:
                    assert synced is True, f"Step {step}: expected sync to detect pending snapshot"
                    pending_upstream_commits = 0
                    bsdb_table.wait_for_background_tasks()
                else:
                    assert synced is False, f"Step {step}: expected no-op sync when no upstream commits"

            if action in ("assert", "sync_and_assert") and pending_upstream_commits == 0:
                # 1. Total row count assertion
                all_data = bsdb_table.to_arrow()
                assert all_data.num_rows == len(model), (
                    f"Step {step}: Row count mismatch. BenoStreamDB={all_data.num_rows}, Model={len(model)}"
                )

                # 2. Point lookup assertion (existing key)
                picked_id = rng.choice(list(model.keys()))
                expected = model[picked_id]
                res = bsdb_table.sql(f"SELECT id, title, category, score, active FROM t WHERE id = {picked_id}")
                assert res.num_rows == 1, f"Step {step}: point lookup id={picked_id} returned {res.num_rows} rows"
                assert res.column("title")[0].as_py() == expected["title"]
                assert res.column("category")[0].as_py() == expected["category"]
                assert res.column("score")[0].as_py() == expected["score"]

                # 3. Non-existent point lookup assertion
                ghost_id = next_id + 9999
                ghost_res = bsdb_table.sql(f"SELECT id FROM t WHERE id = {ghost_id}")
                assert ghost_res.num_rows == 0, f"Step {step}: phantom row returned for non-existent id={ghost_id}"

                # 4. Scalar predicate category filter assertion
                test_cat = rng.choice(CATEGORIES)
                expected_ids = sorted([k for k, v in model.items() if v["category"] == test_cat])
                cat_res = bsdb_table.sql(f"SELECT id FROM t WHERE category = '{test_cat}'")
                got_ids = sorted(cat_res.column("id").to_pylist()) if cat_res.num_rows > 0 else []
                assert got_ids == expected_ids, (
                    f"Step {step}: Category '{test_cat}' filter diverged. Got={got_ids}, Expected={expected_ids}"
                )

                # 5. Synced index verification on the external Iceberg table
                plan = bsdb_table.explain(f"title = '{expected['title']}'")
                assert ("Inverted Index" in plan or "BM25" in plan), (
                    f"Step {step}: Query plan must use synced index, got:\n{plan}"
                )

        # Final reconciliation assertion
        if pending_upstream_commits > 0:
            bsdb_table.sync_detached_overlay()
            bsdb_table.wait_for_background_tasks()

        final_data = bsdb_table.to_arrow()
        assert final_data.num_rows == len(model)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)
