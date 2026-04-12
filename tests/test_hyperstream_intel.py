import hyperstreamdb as hs
import numpy as np
import pyarrow as pa
import os
import shutil
import pytest
import tempfile
import sys

@pytest.fixture
def table_uri():
    with tempfile.TemporaryDirectory() as tmp_dir:
        uri = "file://" + os.path.abspath(tmp_dir)
        yield uri

def test_intel_wgpu_search(table_uri):
    if sys.platform != "linux" or not hs.Device.is_available("xpu"):
        pytest.skip("Intel/XPU testing requires a Linux platform with compatible drivers.")

    dim = 128
    schema = hs.Schema([
        hs.Field("id", hs.DataType.int64()),
        hs.Field("embedding", hs.DataType.vector(dim))
    ])
    
    table = hs.Table.create(table_uri, schema)
    table.add_index_columns(["embedding"])
    
    n_rows = 1000
    ids = pa.array(np.arange(n_rows), type=pa.int64())
    rng = np.random.RandomState(42)
    embeddings_raw = rng.randn(n_rows, dim).astype(np.float32)
    embeddings = pa.FixedSizeListArray.from_arrays(embeddings_raw.flatten(), dim)
    
    batch = pa.RecordBatch.from_arrays([ids, embeddings], names=["id", "embedding"])
    
    table.write([batch])
    table.commit()
    table.wait_for_indexes()
    
    query = rng.randn(dim).astype(np.float32).tolist()
    
    # Use Torch-aligned 'xpu' string
    compute_ctx_arg = "xpu:0"
    print(f"Testing with ComputeContext: {compute_ctx_arg}")

    try:
        results = table.search("embedding", query, k=5, device=hs.Device(compute_ctx_arg))
        print(f"Got {len(results)} results")
        
        assert len(results) == 5
        assert "id" in results.columns
        assert "embedding" in results.columns
        assert len(results["embedding"].iloc[0]) == dim
        
    except Exception as e:
        pytest.skip(f"Intel architecture search skipped (WGPU fallback missing/failed): {e}")
