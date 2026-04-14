import pytest
import numpy as np
import hyperstreamdb as hs
import os
import shutil

@pytest.fixture
def table():
    path = "/tmp/hs_tq_test"
    if os.path.exists(path):
        shutil.rmtree(path)
    
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
    
    # 2. Add HNSW_TQ8 index explicitly using new intuitive names
    table.add_index("embedding", {"type": "hnsw_tq8", "complexity": 16, "quality": 200})
    
    # 3. Add HNSW_TQ4 index explicitly
    table.add_index("embedding", {"type": "hnsw_tq4", "complexity": 16})
    
    # 2. Search using TQ8 (Default)
    query = embeddings[0].tolist()
    results = table.search("embedding", query, k=5)
    assert len(results) > 0
    assert results.iloc[0]["id"] == 0
    print("TurboQuant Search Success!")
    assert results.iloc[0]["id"] == 0
    assert results.iloc[0]["distance"] < 0.1 # Should be very close for TQ8

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
