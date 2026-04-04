import hyperstreamdb as hs
import pyarrow as pa
import numpy as np
import os
import shutil
import pytest
import tempfile

@pytest.fixture
def table_uri():
    # Use a temporary directory for the table URI
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Use file:// protocol for local tables
        uri = "file://" + os.path.abspath(tmp_dir)
        yield uri

def test_pyarrow_schema_consistency(table_uri):
    # 1. Setup Table with Vector column using pyarrow.schema
    dim = 128
    schema = pa.schema([
        ("id", pa.int64()),
        ("embedding", pa.list_(pa.float32(), dim))
    ])
    
    table = hs.Table.create(table_uri, schema)
    
    # 2. Add Index (needed for vector search)
    table.add_index_columns(["embedding"])
    
    # 3. Write Data
    n_rows = 100
    ids = pa.array(np.arange(n_rows), type=pa.int64())
    rng = np.random.RandomState(42)
    embeddings_raw = rng.randn(n_rows, dim).astype(np.float32)
    embeddings = pa.FixedSizeListArray.from_arrays(embeddings_raw.flatten(), dim)
    
    batch = pa.RecordBatch.from_arrays([ids, embeddings], names=["id", "embedding"])
    table.write(pa.Table.from_batches([batch]))
    table.commit()
    
    # 4. Wait for indexing to complete
    table.wait_for_indexes()
    
    # 5. Test Normal Search Result Schema
    query_vec = rng.randn(dim).astype(np.float32).tolist()
    results = table.to_arrow(vector_filter={"column": "embedding", "query": query_vec, "k": 10})
    
    assert len(results) > 0, "Should have results"
    assert "distance" in results.column_names, "Schema should contain 'distance' column"
    # PyArrow table schema check
    assert results.schema.get_field_index("distance") >= 0
    assert results.schema.field("distance").type == pa.float32()
    
    # 6. Test EMPTY Search Result Schema (The Regression Case)
    # Search with a filter that returns no results
    results_empty = table.to_arrow(
        vector_filter={"column": "embedding", "query": query_vec, "k": 10},
        filter="id > 1000"
    )
    
    assert len(results_empty) == 0, "Should have NO results"
    # CRITICAL: Even if empty, the schema should still contain 'distance'
    assert "distance" in results_empty.column_names, "EMPTY search result should STILL contain 'distance' column"
    assert results_empty.schema.get_field_index("distance") >= 0
    
    # 7. Test Projection with Vector Search
    # Only select 'id' but expect 'distance' to be added automatically
    results_proj = table.to_arrow(
        vector_filter={"column": "embedding", "query": query_vec, "k": 5},
        columns=["id"]
    )
    
    assert "id" in results_proj.column_names
    assert "distance" in results_proj.column_names, "Distance should be present even if not explicitly projected"
    assert "embedding" not in results_proj.column_names, "Embedding should NOT be present if not explicitly selected"

if __name__ == "__main__":
    # If run directly, run the test manually
    with tempfile.TemporaryDirectory() as tmp_dir:
        uri = "file://" + os.path.abspath(tmp_dir)
        test_pyarrow_schema_consistency(uri)
        print("Test passed!")
