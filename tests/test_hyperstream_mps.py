import hyperstreamdb as hs
import numpy as np
import pyarrow as pa
import os
import shutil
import pytest
import tempfile

@pytest.fixture
def table_uri():
    with tempfile.TemporaryDirectory() as tmp_dir:
        uri = "file://" + os.path.abspath(tmp_dir)
        yield uri

def test_mps_search(table_uri):
    # 1. Create Table
    dim = 128
    schema = hs.Schema([
        hs.Field("id", hs.DataType.int64()),
        hs.Field("embedding", hs.DataType.vector(dim))
    ])
    
    table = hs.Table.create(table_uri, schema)
    
    # 2. Add Index
    table.add_index_columns(["embedding"])
    
    # 3. Write Data
    n_rows = 1000
    ids = pa.array(np.arange(n_rows), type=pa.int64())
    embeddings_raw = np.random.randn(n_rows, dim).astype(np.float32)
    embeddings = pa.FixedSizeListArray.from_arrays(embeddings_raw.flatten(), dim)
    
    batch = pa.RecordBatch.from_arrays([ids, embeddings], names=["id", "embedding"])
    
    table.write([batch])
    table.commit()
    
    # Wait for indexes to build
    table.wait_for_indexes()
    
    # 4. Search
    query = np.random.randn(dim).astype(np.float32).tolist()
    results = table.search("embedding", query, k=5)
    
    # 5. Assertions
    assert len(results) == 5
    assert "id" in results.columns
    assert "embedding" in results.columns
    assert len(results["embedding"].iloc[0]) == dim
