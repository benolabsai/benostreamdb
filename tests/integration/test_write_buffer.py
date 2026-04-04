import pytest
import hyperstreamdb as hdb
import pyarrow as pa
import numpy as np
import uuid
import os
import shutil
import time

def test_write_buffer_and_index():
    # Setup
    uri = f"/tmp/test_hyperstream_{uuid.uuid4()}"
    if os.path.exists(uri):
        shutil.rmtree(uri)
    
    # Set Cache Limit to something small to test triggering (or we trigger manually)
    # Even better, we test the buffering behavior specifically.
    os.environ["HYPERSTREAM_CACHE_GB"] = "1" 
    
    table = hdb.Table(uri)
    table.index_all_columns() # Enable indexing
    
    # 1. Create Data (Vector + Scalar)
    dim = 128
    rows = 1000
    
    # Vectors: [1.0, 1.0, ...]
    vectors = []
    vector = [1.0] * dim
    for _ in range(rows):
        vectors.append(vector)
        
    ids = range(rows)
    
    schema = pa.schema([
        ('id', pa.int32()),
        ('embedding', pa.list_(pa.float32(), dim))
    ])
    
    batch = pa.RecordBatch.from_arrays([
        pa.array(ids),
        pa.array(vectors, type=pa.list_(pa.float32(), dim))
    ], schema=schema)
    
    # 2. Write to Buffer
    print("Writing 1000 rows to buffer...")
    table.write_arrow(pa.Table.from_batches([batch]))
    
    # 3. Verify NOT on disk (Checking manifest or file count via internal verify, 
    # but user can check by listing files if they want. Here we rely on behavior)
    # Actually, we can check basic read.
    
    print("Reading back (should come from buffer)...")
    results = table.to_pandas()
    assert len(results) > 0
    total_rows = len(results)
    assert total_rows == rows
    print(f"Read success: {total_rows} rows found in buffer.")
    
    # 4. Verify Vector Search in Buffer
    print("Executing Vector Search on Buffer...")
    # Query with exact match
    query_vec = vector 
    search_results = table.to_pandas(
        vector_filter={"column": "embedding", "query": query_vec, "k": 10}
    )
    
    assert len(search_results) > 0
    print(f"Buffer Vector Search returned {len(search_results)} rows.")
    # Verify ID is correct (should be 0..9)
    res_ids = search_results["id"].tolist()
    print("Result IDs:", res_ids)
    assert 0 in res_ids
    
    # 5. Flush / Commit
    print("Committing (Flushing to Disk)...")
    table.commit()
    
    # 6. Verify Data persisted and Memory cleared
    # We can check by reading again (should still find data, but from disk now)
    print("Reading after commit...")
    results_disk = table.read()
    total_rows_disk = results_disk.num_rows if hasattr(results_disk, "num_rows") else sum(b.num_rows for b in results_disk)
    assert total_rows_disk == rows
    
    # 7. Verify Vector Search on Disk
    print("Executing Vector Search on Disk...")
    search_results_disk = table.vector_search(
        "embedding",
        query_vec,
        k=10
    )
    assert len(search_results_disk) > 0
    print("Disk Vector Search success.")

    # Cleanup
    if os.path.exists(uri):
        shutil.rmtree(uri)

if __name__ == "__main__":
    test_write_buffer_and_index()
