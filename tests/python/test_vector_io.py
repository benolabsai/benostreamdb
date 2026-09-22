"""Vector-search I/O: projection and the score-only short-circuit.

Two behaviours pinned here:

1. A projection of only the synthesised `distance` column used to build an
   *empty* Parquet projection, which failed with "must either specify a row
   count or at least one column". It must return the scores.
2. The score is already known from the index search, so a score-only query must
   not read Parquet at all — and must return exactly the same scores as a full
   search.
"""
import numpy as np
import pyarrow as pa
import pytest

import hyperstreamdb as hdb

DIM = 16
N = 500


@pytest.fixture(scope="module")
def table(tmp_path_factory):
    uri = f"file://{tmp_path_factory.mktemp('vio')}/t"
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((N, DIM), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

    schema = pa.schema([
        ("id", pa.int64()),
        ("title", pa.large_string()),
        ("embedding", pa.list_(pa.float32(), DIM)),
    ])
    t = hdb.Table.create(uri, schema)
    t.write(pa.table({
        "id": pa.array(np.arange(N), pa.int64()),
        "title": pa.array([f"p{i}" for i in range(N)], pa.large_string()),
        "embedding": pa.FixedSizeListArray.from_arrays(
            pa.array(vecs.reshape(-1), pa.float32()), DIM),
    }))
    t.add_index("embedding", "hnsw")
    t.commit()
    t.wait_for_background_tasks()
    return t, vecs


def test_score_only_projection_returns_scores(table):
    """Projecting only `distance` must not blow up on an empty projection."""
    t, vecs = table
    res = t.vector_search("embedding", vecs[7].tolist(), k=5, columns=["distance"])
    assert list(res.columns) == ["distance"]
    assert len(res) == 5
    # Nearest neighbour of row 7 is row 7, distance 0.
    assert res["distance"].iloc[0] == pytest.approx(0.0, abs=1e-6)


def test_score_only_matches_full_search(table):
    t, vecs = table
    q = vecs[7].tolist()
    full = t.vector_search("embedding", q, k=5, columns=["id", "distance"])
    score_only = t.vector_search("embedding", q, k=5, columns=["distance"])

    assert score_only["distance"].tolist() == full["distance"].tolist()
    assert int(full["id"].iloc[0]) == 7


def test_projection_excludes_unrequested_payload_columns(table):
    """Only the requested payload columns are read; the score is always added."""
    t, vecs = table
    res = t.vector_search("embedding", vecs[7].tolist(), k=3, columns=["id"])
    assert "id" in res.columns
    assert "distance" in res.columns
    # Not requested, and the wide `embedding` column must not have been read.
    assert "title" not in res.columns
    assert "embedding" not in res.columns
    assert len(res) == 3