import tempfile
import pytest
import numpy as np
import benostreamdb as hs
import os
import shutil

@pytest.fixture
def table():
    path = tempfile.mkdtemp(prefix="hs_tq_test_")
    
    # Create a table with an embedding column
    table = hs.Table.create(
        path,
        schema=hs.Schema([
            hs.Field("id", hs.DataType.int64()),
            hs.Field("embedding", hs.DataType.vector(128))
        ])
    )
    yield table
    if os.path.exists(path):
        shutil.rmtree(path)

def test_turboquant_indexing(table):
    # 1. Ingest some vectors
    np.random.seed(42)
    n = 1000
    embeddings = np.random.randn(n, 128).astype(np.float32)
    data = [
        {"id": i, "embedding": embeddings[i].tolist()}
        for i in range(n)
    ]
    table.insert(data)
    table.commit()
    
    # 2. Add HNSW_TQ8 index explicitly using new intuitive names.
    #
    # This test used to also add `hnsw_tq4` here. A vector column carries one
    # vector index family, so the second `add_index` REPLACES the first: the
    # query was then answered by the 4-bit index while the assertions still
    # assumed TQ8's error scale. That is not a stable test -- the 8-bit
    # self-match sits at ~0.12 and the 4-bit one at ~1.92, so the distance bound
    # below passed or failed depending on which index won the race. TQ4 has its
    # own test at the bottom of this file, with a scale-free assertion.
    table.add_index("embedding", {"type": "hnsw_tq8", "complexity": 16, "quality": 200})

    # Wait for the background index build tasks to complete
    table.wait_for_background_tasks()

    # 2. Search using TQ8 (Default)
    query = embeddings[0].tolist()
    results = table.search("embedding", query, k=5)
    assert len(results) > 0
    # TurboQuant is a lossy, *approximate* index. Rank-1 is not guaranteed, so
    # assert recall@k instead: the exact nearest must be in the top-k.
    ids = list(results["id"])
    assert 0 in ids, f"exact nearest (id 0) missing from top-{len(ids)}: {ids}"
    print("TurboQuant Search Success!")
    # The self-match should be very close under 8-bit TurboQuant. The bound is
    # TQ8-specific (measured ~0.12); it would not hold for the 4-bit index.
    assert results.loc[results["id"] == 0, "distance"].iloc[0] < 0.5


def test_turboquant4_indexing(table):
    """TQ4 keeps recall; its distance scale is its own and must not be compared
    to TQ8's. 4-bit codes are coarser, so the self-match distance is ~1.92
    rather than ~0.12 -- asserting a fixed magnitude here is what made the TQ8
    test order-dependent."""
    np.random.seed(7)
    n = 500
    embeddings = np.random.randn(n, 128).astype(np.float32)
    table.insert([{"id": i, "embedding": embeddings[i].tolist()} for i in range(n)])
    table.commit()

    table.add_index("embedding", {"type": "hnsw_tq4", "complexity": 16})
    table.wait_for_background_tasks()

    results = table.search("embedding", embeddings[0].tolist(), k=5)
    assert len(results) > 0
    ids = list(results["id"])
    assert 0 in ids, f"exact nearest (id 0) missing from top-{len(ids)}: {ids}"
    # Scale-free ordering check: the self-match must be the closest returned row.
    self_distance = results.loc[results["id"] == 0, "distance"].iloc[0]
    assert self_distance == results["distance"].min(), (
        f"self-match distance {self_distance} is not the minimum "
        f"{results['distance'].min()}"
    )

def test_tq_auto_threshold(table):
    # Verify that we can still manually trigger TQ8
    # Default without params should use our new default (HnswTq8)
    table.add_index("embedding", "hnsw_tq8")
    
    # Check if index exists in manifest (verify snake_case type)
    manifest = table.manifest()
    embedding_field = next(f for f in manifest["schemas"][-1]["fields"] if f["name"] == "embedding")
    algos = [idx["type"] for idx in embedding_field["indexes"]]
    assert "hnsw_tq8" in algos

if __name__ == "__main__":
    # Manual run
    hs.init()
    # ...
