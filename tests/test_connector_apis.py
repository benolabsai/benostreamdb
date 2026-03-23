import hyperstreamdb as hdb
import pyarrow as pa
import pandas as pd
import shutil
import os

def test_connector_apis():
    print("Testing Connector APIs...")
    
    # Setup
    uri = "file:///tmp/test_connector_api_table"
    if os.path.exists("/tmp/test_connector_api_table"):
        shutil.rmtree("/tmp/test_connector_api_table")
        
    table = hdb.Table(uri)
    
    # Create test data
    df = pd.DataFrame({
        "id": range(1000),
        "name": [f"user_{i}" for i in range(1000)],
        "age": [i % 50 for i in range(1000)],
        "category": ["A" if i % 2 == 0 else "B" for i in range(1000)]
    })
    
    # Index 'age' and 'category'
    table.add_index_columns(["age", "category"])
    
    print("- Writing data...")
    table.write_pandas(df)
    table.wait_for_indexes()
    
    print("- Compacting and Backfilling Indexes...")
    table.compact(min_file_size_bytes=1024)
    table.index_all_columns()
    table.wait_for_indexes()
    
    # 1. Test list_data_files
    print("\n1. Testing list_data_files()...")
    files = table.list_data_files()
    assert len(files) > 0
    print(f"  Found {len(files)} files")
    
    file_info = files[0]
    print(f"  File: {file_info.file_path}")
    print(f"  Rows: {file_info.row_count}")
    print(f"  Size: {file_info.file_size_bytes} bytes")
    print(f"  Scalar Indexes: {file_info.has_scalar_indexes}")
    print(f"  Indexed Columns: {file_info.indexed_columns}")
    
    assert file_info.row_count == 1000
    assert "age" in file_info.indexed_columns
    assert "category" in file_info.indexed_columns
    assert file_info.has_scalar_indexes
    
    # 2. Test read_file
    print("\n2. Testing read_file()...")
    # Read with filter (should use index)
    t = table.read_file(file_info.file_path, filter="age > 40")
    print(f"  Read {t.num_rows} rows where age > 40")
    
    # Verify correctness
    filtered_df = df[df["age"] > 40]
    # Note: Row count might differ slightly if compaction created multiple files, but for small data likely 1 file
    # Or read_file reads THAT file's content matching filter.
    # Since we likely have 1 file, it should match.
    assert t.num_rows == len(filtered_df)
    
    # 3. Test get_splits (Trino parallelism)
    print("\n3. Testing get_splits()...")
    # Force small split size to generate multiple splits
    splits = table.get_splits(max_split_size=1024) 
    print(f"  Generated {len(splits)} splits for {file_info.file_size_bytes} bytes with max_split=1024")
    
    assert len(splits) >= 1
    first_split = splits[0]
    print(f"  Split 0: offset={first_split.start_offset}, length={first_split.length}")
    print(f"  Can use indexes: {first_split.can_use_indexes}")
    
    assert first_split.can_use_indexes
    assert first_split.index_file_path is not None
    
    # 4. Test read_split (Column projection)
    print("\n4. Testing read_split()...")
    # Project only 'id' and 'name'
    t_split = table.read_split(first_split, columns=["id", "name"])
    print(f"  Read {t_split.num_rows} rows from split 0")
    print(f"  Columns: {t_split.column_names}")
    
    assert "id" in t_split.column_names
    assert "name" in t_split.column_names
    assert "age" not in t_split.column_names
    
    # 5. Test get_table_statistics
    print("\n5. Testing get_table_statistics()...")
    stats = table.get_table_statistics()
    print(f"  Total Rows: {stats.row_count}")
    print(f"  Total Size: {stats.total_size_bytes}")
    print(f"  Indexed Columns (Scalar): {stats.index_coverage.scalar_indexed_columns}")
    
    assert stats.row_count == 1000
    assert "age" in stats.index_coverage.scalar_indexed_columns
    
    print("\n✅ All Connector API tests passed!")

if __name__ == "__main__":
    test_connector_apis()
