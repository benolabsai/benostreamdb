# Copyright (c) 2026 Richard Albright. All rights reserved.

"""Memory saturation: a low ingest RAM high-water mark must pause, not OOM.

Regression guard for the OOM that killed the Wikipedia load: writers must
back-pressure on `BSDB_MAX_INGEST_RAM_GB` and resume once background index builds
release memory, rather than growing RSS until the OS kills the process.
"""

import pytest

pa = pytest.importorskip("pyarrow")
np = pytest.importorskip("numpy")
bsdb = pytest.importorskip("benostreamdb")

DIM = 128
ROWS = 20_000


def _table(offset: int):
    ids = pa.array(range(offset, offset + ROWS), type=pa.int64())
    vec = np.random.rand(ROWS * DIM).astype(np.float32)
    emb = pa.FixedSizeListArray.from_arrays(pa.array(vec), DIM)
    return pa.Table.from_batches(
        [pa.RecordBatch.from_arrays([ids, emb], names=["id", "embedding"])]
    )


def test_low_ram_limit_pauses_without_oom(monkeypatch, table_uri, rows_of, rss_gb, rss_ceiling_gb):
    # Force back-pressure hard: a tiny limit plus a single build slot means the
    # writer will have to wait for the background build to release memory.
    monkeypatch.setenv("BSDB_MAX_INGEST_RAM_GB", "0.5")
    monkeypatch.setenv("BSDB_INDEX_BUILD_CONCURRENCY", "1")

    table = bsdb.Table.create(
        table_uri,
        pa.schema([("id", pa.int64()), ("embedding", pa.list_(pa.float32(), DIM))]),
    )
    table.add_index("embedding", {"type": "hnsw", "device": "cpu"})

    writes = 4
    for i in range(writes):
        # Must return (possibly after waiting for memory to be reclaimed), not
        # abort. A panic or OOM here fails the test by killing the process.
        table.write(_table(i * ROWS))
    table.commit()
    table.wait_for_background_tasks()

    # Graceful degradation: the process survived and the data is readable.
    assert rows_of(table.read(columns=["id"])) == writes * ROWS

    rss = rss_gb()
    assert rss < rss_ceiling_gb, (
        f"RSS grew to {rss:.2f} GB with a 0.5 GB ingest limit "
        f"(ceiling {rss_ceiling_gb:.2f} GB) — back-pressure is not holding"
    )
