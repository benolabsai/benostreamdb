# Copyright (c) 2026 Richard Albright. All rights reserved.

"""Disk I/O saturation: many commit cycles (WAL + manifest + parquet) must
complete without losing or duplicating rows.

Each `commit` writes a WAL record, a manifest generation, and a small parquet
file, so a tight loop of them is a cheap, deterministic way to hammer disk I/O
and the manifest CAS without needing a large dataset.
"""

import pytest

pa = pytest.importorskip("pyarrow")
np = pytest.importorskip("numpy")
bsdb = pytest.importorskip("benostreamdb")

DIM = 32
ROWS = 2_000
CYCLES = 20


def _table(offset: int):
    ids = pa.array(range(offset, offset + ROWS), type=pa.int64())
    vec = np.random.rand(ROWS * DIM).astype(np.float32)
    emb = pa.FixedSizeListArray.from_arrays(pa.array(vec), DIM)
    return pa.Table.from_batches(
        [pa.RecordBatch.from_arrays([ids, emb], names=["id", "embedding"])]
    )


def test_many_commit_cycles_are_durable(table_uri, rows_of):
    table = bsdb.Table.create(
        table_uri,
        pa.schema([("id", pa.int64()), ("embedding", pa.list_(pa.float32(), DIM))]),
    )

    for i in range(CYCLES):
        table.write(_table(i * ROWS))
        table.commit()

    table.wait_for_background_tasks()

    # Integrity: every committed row is present exactly once.
    assert rows_of(table.read(columns=["id"])) == CYCLES * ROWS
