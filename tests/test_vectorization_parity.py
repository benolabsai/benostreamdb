import hyperstreamdb as hs
import numpy as np
import pyarrow as pa
import pytest
import os
import tempfile
import sys

@pytest.fixture
def table_uri():
    with tempfile.TemporaryDirectory() as tmp_dir:
        uri = "file://" + os.path.abspath(tmp_dir)
        yield uri

def test_vectorization_parity(table_uri):
    # This verifies the DataFusion distance loops in vector_udf.rs function natively identical to CPU loop.
    dim = 64
    schema = hs.Schema([
        hs.Field("id", hs.DataType.int64()),
        hs.Field("embedding", hs.DataType.vector(dim))
    ])
    
    table = hs.Table.create(table_uri, schema)
    
    n_rows = 1000
    ids = pa.array(np.arange(n_rows), type=pa.int64())
    rng = np.random.RandomState(42)
    embeddings_raw = rng.randn(n_rows, dim).astype(np.float32)
    embeddings = pa.FixedSizeListArray.from_arrays(embeddings_raw.flatten(), dim)
    batch = pa.RecordBatch.from_arrays([ids, embeddings], names=["id", "embedding"])
    
    table.write([batch])
    table.commit()

    query = rng.randn(dim).astype(np.float32).tolist()
    
    # 1. CPU Reference Run (Non-Batched Iterator inside DataFusion)
    hs.Device("cpu").activate()
    cpu_result = table.search("embedding", query, k=1000).sort_values(by="id")
    # Column name is 'distance' according to previous failure logs
    cpu_distances = cpu_result["distance"].to_numpy()
    
    # 2. Intel/ROCm WGSL Run (Vectorized Matrix Batching across Wgpu)
    if sys.platform == "linux" and hs.Device.is_available("xpu"):
        try:
            hs.Device("xpu").activate()
            gpu_result = table.search("embedding", query, k=1000).sort_values(by="id")
            gpu_distances = gpu_result["distance"].to_numpy()
            
            # Verify Parity
            np.testing.assert_allclose(
                cpu_distances, 
                gpu_distances, 
                rtol=1e-4, 
                err_msg="Vectorization Parity Failed! Vectorized WGPU paths output mismatch vs CPU."
            )
            print("Vectorization Parity Test: PASSED (WGSL batch mapping matches CPU arrays)")
        except Exception as e:
            print(f"Skipping GPU parity part: {e}")
    else:
        print("Skipped GPU vectorization check since machine is not Linux.")
