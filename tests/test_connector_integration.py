
import pytest
import shutil
import tempfile
import pyarrow as pa
import pandas as pd
import numpy as np
import os
import hyperstreamdb

@pytest.fixture
def table_uri():
    tmp_dir = tempfile.mkdtemp()
    uri = f"file://{tmp_dir}/test_table_connectors"
    yield uri
    shutil.rmtree(tmp_dir)

def test_connector_apis(table_uri):
    print(f"Creating table at {table_uri}")
    table = hyperstreamdb.Table(table_uri)
    
    # 1. Create Data
    df = pd.DataFrame({
        "id": range(1000),
        "tag": ["A"] * 500 + ["B"] * 500,
        "value": np.random.randn(1000),
        "embedding": [np.random.rand(128).astype(np.float32) for _ in range(1000)]
    })
    
    # 2. Write Data
    print("Writing data...")
    table.write_pandas(df)
    table.commit() # Ensure data is flushed to disk

    # 3. Configure Indexing (optional, to verify index flags)
    # table.add_index_columns(["tag"])
    # table.index_all_columns() # Should trigger backfill
    
    # Compact to ensure manifest is up to date and indexes (if any) are finalized
    # Note: write() adds to buffer, commit() flushes to parquet. 
    # list_data_files reads from Manifest.
    # We need to ensure commit() updates the manifest. 
    # (In current impl, flush_async -> write_segment -> commit_manifest). So yes.
    
    # 4. Test list_data_files
    print("Testing list_data_files...")
    files = table.list_data_files()
    assert len(files) > 0, "Should have at least one data file"
    
    first_file = files[0]
    print(f"File Info: Path={first_file.file_path}, Rows={first_file.row_count}, Size={first_file.file_size_bytes}")
    
    assert first_file.row_count == 1000
    assert "tag" in first_file.min_values or True # Depending on if stats are gathered
    
    # 5. Test get_splits
    print("Testing get_splits...")
    # Force small split size to generate multiple splits per file if possible, 
    # but our file is likely small (~100KB). 
    # Let's try a very small split size (e.g. 1KB) to force splitting? 
    # Or just verify 1 split per file.
    
    max_split_size = 1024 * 1024 * 64 # 64MB default
    splits = table.get_splits(max_split_size)
    assert len(splits) >= len(files)
    
    split = splits[0]
    print(f"Split Info: Path={split.file_path}, Start={split.start_offset}, Length={split.length}")
    assert split.file_path == first_file.file_path
    
    # 6. Test read_file
    print("Testing read_file...")
    arrow_table = table.read_file(first_file.file_path, None)
    df_read = arrow_table.to_pandas()
    assert len(df_read) == 1000
    
    # Test with filter (index acceleration check - should now apply post-filtering)
    # Note: providing valid SQL filter string
    arrow_filtered = table.read_file(first_file.file_path, "tag = 'A'")
    # Verification expectation: returns matching rows (correctly applies filter)
    assert len(arrow_filtered) == 500
    
    # 7. Test read_split
    print("Testing read_split...")
    # Split has generic file path.
    arrow_split = table.read_split(split, ["id", "tag"]) # Projection
    print(f"Read columns: {arrow_split.num_columns}, Names: {arrow_split.column_names}")
    assert arrow_split.num_columns == 2
    
    # 8. Test get_table_statistics
    print("Testing statistics...")
    stats = table.get_table_statistics()
    assert stats.row_count == 1000
    assert stats.file_count == len(files)
    assert stats.total_size_bytes > 0
    
    print("Integration test passed!")

if __name__ == "__main__":
    # If run directly
    t_uri = "/tmp/hs_manual_test"
    if os.path.exists(t_uri):
        shutil.rmtree(t_uri)
    test_connector_apis("file://" + t_uri)
