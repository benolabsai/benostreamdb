# Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

"""`Table.subscribe()` — the live change feed, exposed natively in Python.

The core broadcast is in-process, so a subscription observes commits made by
writers in the same process (which is exactly what these tests exercise).
"""

import pyarrow as pa

import benostreamdb


def _table(tmp_path, name="t"):
    return benostreamdb.Table.from_arrow(
        f"file://{tmp_path}/{name}", pa.table({"v": [1, 2, 3]})
    )


def test_subscribe_receives_committed_batch(tmp_path):
    table = _table(tmp_path)
    sub = table.subscribe()

    table.write(pa.table({"v": [4, 5]}))
    table.commit()

    # Drain until we see the committed batch (a commit marker may arrive first).
    seen_batch = None
    for _ in range(5):
        ev = sub.recv(timeout_ms=2000)
        if ev is None:
            break
        if ev["event_type"] == "batch" and ev["rows"] > 0:
            seen_batch = ev
            break
    assert seen_batch is not None, "no committed batch was delivered"
    assert seen_batch["data"].num_rows == seen_batch["rows"]


def test_subscribe_filtered_only_delivers_matching_rows(tmp_path):
    table = _table(tmp_path, "f")
    sub = table.subscribe_filtered("v >= 10")

    # First commit matches nothing -> must not be delivered.
    table.write(pa.table({"v": [1, 2, 3]}))
    table.commit()
    # Second commit has matching rows.
    table.write(pa.table({"v": [5, 10, 20]}))
    table.commit()

    ev = sub.recv(timeout_ms=2000)
    assert ev is not None
    assert ev["event_type"] == "batch"
    assert ev["rows"] == 2, "only v>=10 rows should be delivered"


def test_try_recv_is_non_blocking(tmp_path):
    table = _table(tmp_path, "n")
    sub = table.subscribe()
    # Nothing committed yet.
    assert sub.try_recv() is None


def test_subscription_close_and_context_manager(tmp_path):
    table = _table(tmp_path, "c")
    sub = table.subscribe()
    assert not sub.is_closed()
    sub.close()
    assert sub.is_closed()
    # recv after close raises rather than blocking.
    import pytest

    with pytest.raises(Exception):
        sub.recv(timeout_ms=10)

    with table.subscribe() as ctx_sub:
        assert not ctx_sub.is_closed()
    assert ctx_sub.is_closed()


def test_subscribe_events_sql_table_function(tmp_path):
    """`subscribe_events(...)` is a DataFusion table function, so it is
    reachable from every SQL surface (this is the universal path)."""
    table = _table(tmp_path, "sql")
    session = benostreamdb.Session()
    session.register("s", table)
    # Resolves and drains (no commits in this process -> zero rows, no error).
    session.sql("SELECT * FROM subscribe_events('s', '', 5, 50)")
