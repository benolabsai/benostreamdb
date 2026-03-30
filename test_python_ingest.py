import os
import pandas as pd
import numpy as np
import hyperstreamdb as hdb
import shutil
import pytest

DB_PATH = "test_unified_ingest_db"

@pytest.fixture
def setup_db():
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
    yield
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

def test_unified_ingestion(setup_db):
    table = hdb.Table.create(DB_PATH, hdb.Schema([
        hdb.Field("id", hdb.DataType.int64()),
        hdb.Field("val", hdb.DataType.float32()),
        hdb.Field("embedding", hdb.DataType.vector(3))
    ]))

    print("Testing Pandas ingest...")
    df = pd.DataFrame({
        "id": [1, 2],
        "val": [10.5, 20.5],
        # Each row gets a numpy array (float32)
        "embedding": [np.array([0.1, 0.2, 0.3], dtype=np.float32), 
                      np.array([0.4, 0.5, 0.6], dtype=np.float32)]
    })
    table.write(df)
    
    # Verify duplicates logic (Standard write appends)
    table.write(df)
    table.commit()
    assert table.row_count == 4

    # Test OVERWRITE
    print("Testing OVERWRITE (clearing duplicates)...")
    table.write(df, mode="overwrite")
    table.commit()
    assert len(table.to_pandas()) == 2
    
    # Test UPSERT (Deduplication)
    print("Testing UPSERT (deduplication)...")
    df_upsert = pd.DataFrame({
        "id": [1],
        "val": [99.9],
        "embedding": [np.array([0.1, 0.2, 0.3], dtype=np.float32)]
    })
    # This should update existing id=1 rows or mark them as deleted
    table.upsert(df_upsert, key_column="id")
    table.commit()
    
    # Exact row count depends on merge-on-read vs merge-on-write
    # In MoR, we add new data + delete files. Total *active* rows should stay same or change.
    results = table.to_pandas()
    # id=1 should now be 99.9
    assert 99.9 in results["val"].values
    
    print("Testing auto-vectorization with Search...")
    class FakeEmbedder(hdb.EmbeddingFunction):
        def __call__(self, texts):
            return np.array([[0.1, 0.2, 0.3]] * len(texts), dtype=np.float32)

    hdb.registry.register("fake", FakeEmbedder())
    table.define_embedding("text", "fake", vector_column="embedding")
    table.commit()
    table.wait_for_background_tasks()

    # This should trigger auto-vectorization of "hello" into [0.1, 0.2, 0.3]
    search_res = table.query().vector_search("hello", column="embedding", k=1).to_pandas()
    assert len(search_res) == 1
    assert search_res.iloc[0]["id"] == 1
    assert search_res.iloc[0]["val"] == 99.9
    
    print("Test passed!")

if __name__ == "__main__":
    pytest.main([__file__])
