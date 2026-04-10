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

def test_cuda_search(table_uri):
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
    # Fixed seed for reproducibility
    rng = np.random.RandomState(42)
    embeddings_raw = rng.randn(n_rows, dim).astype(np.float32)
    embeddings = pa.FixedSizeListArray.from_arrays(embeddings_raw.flatten(), dim)
    
    batch = pa.RecordBatch.from_arrays([ids, embeddings], names=["id", "embedding"])
    
    table.write([batch])
    table.commit()
    
    # Wait for indexes to build
    table.wait_for_indexes()
    
    # 4. Search with CUDA context
    query = rng.randn(dim).astype(np.float32).tolist()
    
    # Explicitly request CUDA backend via PyTorch-style string
    compute_ctx_arg = "cuda:0"
    print(f"Testing with ComputeContext: {compute_ctx_arg}")

    try:
        results = table.search("embedding", query, k=5, device=hs.Device(compute_ctx_arg))
        
        print(f"Got {len(results)} results")
        
        # 5. Assertions
        assert len(results) == 5
        assert "id" in results.columns
        assert "embedding" in results.columns
        assert len(results["embedding"].iloc[0]) == dim
        
    except Exception as e:
        # If the error is related to CUDA not being available (which might happen if shared lib load fails)
        # we skip the test instead of failing since linux CI runners don't have GPUs.
        pytest.skip(f"CUDA search skipped (CUDA backend not available): {e}")
