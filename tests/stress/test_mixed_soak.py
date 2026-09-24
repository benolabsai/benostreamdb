# Copyright (c) 2026 Richard Albright. All rights reserved.

"""Mixed-workload soak: continuous write/search/read churn for a wall-clock
budget. The pass criteria are the GA-relevant ones — the process stays alive,
memory does not run away, and every operation keeps returning results.
"""

import time

import pytest

pa = pytest.importorskip("pyarrow")
np = pytest.importorskip("numpy")
bsdb = pytest.importorskip("benostreamdb")

DIM = 64
ROWS = 1_000


def _table(offset: int):
    ids = pa.array(range(offset, offset + ROWS), type=pa.int64())
    vec = np.random.rand(ROWS * DIM).astype(np.float32)
    emb = pa.FixedSizeListArray.from_arrays(pa.array(vec), DIM)
    return pa.Table.from_batches(
        [pa.RecordBatch.from_arrays([ids, emb], names=["id", "embedding"])]
    )


def test_mixed_workload_soak(monkeypatch, table_uri, rows_of, rss_gb, rss_ceiling_gb, soak_seconds):
    monkeypatch.setenv("BSDB_INDEX_BUILD_CONCURRENCY", "2")

    table = bsdb.Table.create(
        table_uri,
        pa.schema([("id", pa.int64()), ("embedding", pa.list_(pa.float32(), DIM))]),
    )
    table.add_index("embedding", {"type": "hnsw", "device": "cpu"})

    query = np.random.rand(DIM).astype(np.float32).tolist()
    deadline = time.time() + soak_seconds
    writes = 0
    while time.time() < deadline:
        table.write(_table(writes * ROWS))
        writes += 1

        if writes % 3 == 0:
            table.commit()
        if writes % 2 == 0:
            hits = table.search("embedding", query, k=5)
            assert rows_of(hits) >= 1, "search returned nothing mid-soak"
        if writes % 5 == 0:
            assert rows_of(table.read(columns=["id"])) >= ROWS

    table.commit()
    table.wait_for_background_tasks()

    assert writes > 1, "soak did not perform any writes"
    rss = rss_gb()
    assert rss < rss_ceiling_gb, (
        f"RSS grew to {rss:.2f} GB over a {soak_seconds:.0f}s soak "
        f"(ceiling {rss_ceiling_gb:.2f} GB) — likely a leak or unbounded queue"
    )
