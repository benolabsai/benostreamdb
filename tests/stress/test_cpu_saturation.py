# Copyright (c) 2026 Richard Albright. All rights reserved.

"""CPU saturation: gated index builds must not starve concurrent reads.

The `BSDB_INDEX_BUILD_CONCURRENCY` gate exists so the runtime cannot fan out
`nproc` multi-GB index builds. This checks the other side of the bargain:
while builds are gated and queued, queries still complete in bounded time
rather than blocking indefinitely behind them.
"""

import time

import pytest

pa = pytest.importorskip("pyarrow")
np = pytest.importorskip("numpy")
bsdb = pytest.importorskip("benostreamdb")

DIM = 64
ROWS = 5_000
SEARCH_DEADLINE_S = 60.0


def _table(offset: int):
    ids = pa.array(range(offset, offset + ROWS), type=pa.int64())
    vec = np.random.rand(ROWS * DIM).astype(np.float32)
    emb = pa.FixedSizeListArray.from_arrays(pa.array(vec), DIM)
    return pa.Table.from_batches(
        [pa.RecordBatch.from_arrays([ids, emb], names=["id", "embedding"])]
    )


def test_gated_builds_do_not_starve_queries(monkeypatch, table_uri, rows_of):
    monkeypatch.setenv("BSDB_INDEX_BUILD_CONCURRENCY", "2")

    table = bsdb.Table.create(
        table_uri,
        pa.schema([("id", pa.int64()), ("embedding", pa.list_(pa.float32(), DIM))]),
    )
    table.add_index("embedding", {"type": "hnsw", "device": "cpu"})

    # Queue several index builds by flushing repeatedly; with a gate of 2 these
    # cannot all run at once.
    for i in range(6):
        table.write(_table(i * ROWS))
        table.commit()

    # Queries must still make progress while builds drain.
    query = np.random.rand(DIM).astype(np.float32).tolist()
    deadline = time.time() + SEARCH_DEADLINE_S
    results = 0
    while time.time() < deadline:
        hits = table.search("embedding", query, k=5)
        assert rows_of(hits) >= 1
        results += 1
        if results >= 3:
            break

    assert results >= 3, "queries did not complete while index builds were gated"

    table.wait_for_background_tasks()
    assert rows_of(table.read(columns=["id"])) == 6 * ROWS
