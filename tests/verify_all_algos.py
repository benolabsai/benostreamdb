import pytest
import numpy as np
import hyperstreamdb as hs
import os
import shutil

def setup_table(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    
    table = hs.Table.create(
        path,
        schema=hs.Schema([
            hs.Field("id", hs.DataType.int64()),
            hs.Field("embedding", hs.DataType.vector(128)),
            hs.Field("text", hs.DataType.string())
        ])
    )
    return table

def test_all_algorithms():
    path = "/tmp/hs_all_algos_test"
    table = setup_table(path)
    
    # 1. Ingest baseline data
    n = 100
    embeddings = np.random.randn(n, 128).astype(np.float32)
    data = [
        {"id": i, "embedding": embeddings[i].tolist(), "text": f"document {i}"}
        for i in range(n)
    ]
    table.insert(data)
    
    # 2. Test hnsw_tq8 (The new default)
    print("Testing hnsw_tq8...")
    table.add_index("embedding", "hnsw_tq8")
    manifest = table.manifest()
    algos = [idx["type"] for idx in next(f for f in manifest["schemas"][-1]["fields"] if f["name"] == "embedding")["indexes"]]
    assert "hnsw_tq8" in algos
    
    # 3. Test hnsw_tq4 (Higher compression)
    print("Testing hnsw_tq4...")
    table.add_index("embedding", {"type": "hnsw_tq4", "complexity": 16})
    manifest = table.manifest()
    algos = [idx["type"] for idx in next(f for f in manifest["schemas"][-1]["fields"] if f["name"] == "embedding")["indexes"]]
    assert "hnsw_tq4" in algos
    
    # 4. Test hnsw_pq (Legacy PQ)
    print("Testing hnsw_pq...")
    table.add_index("embedding", {"type": "hnsw_pq", "complexity": 16, "compression": 32})
    manifest = table.manifest()
    algos = [idx["type"] for idx in next(f for f in manifest["schemas"][-1]["fields"] if f["name"] == "embedding")["indexes"]]
    assert "hnsw_pq" in algos
    
    # 5. Test hnsw (Uncompressed)
    print("Testing hnsw (uncompressed)...")
    table.add_index("embedding", {"type": "hnsw", "complexity": 32, "quality": 100})
    manifest = table.manifest()
    algos = [idx["type"] for idx in next(f for f in manifest["schemas"][-1]["fields"] if f["name"] == "embedding")["indexes"]]
    assert "hnsw" in algos
    
    # 6. Test bm25 (Keyword)
    print("Testing bm25...")
    table.add_index("text", "bm25")
    manifest = table.manifest()
    algos = [idx["type"] for idx in next(f for f in manifest["schemas"][-1]["fields"] if f["name"] == "text")["indexes"]]
    assert "bm25" in algos
    
    # 7. Test Search accuracy for TQ8
    query = embeddings[0].tolist()
    results = table.search("embedding", query, k=5)
    assert len(results) > 0
    assert results.iloc[0]["id"] == 0
    print("TurboQuant Search Success!")

if __name__ == "__main__":
    try:
        test_all_algorithms()
        print("\nALL ALGORITHM TESTS PASSED!")
    finally:
        path = "/tmp/hs_all_algos_test"
        if os.path.exists(path):
            shutil.rmtree(path)
