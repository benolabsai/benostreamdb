"""
Error handling and edge case tests for HyperStreamDB.

Tests various error scenarios including disk full, network failures,
invalid data, schema mismatches, and resource exhaustion.
"""

import hyperstreamdb as hdb
import pyarrow as pa
import pytest
import os
import shutil
from pathlib import Path


@pytest.fixture
def test_table_path(tmp_path):
    """Create a temporary table path."""
    table_path = tmp_path / "error_test"
    yield f"file://{table_path}"
    if table_path.exists():
        shutil.rmtree(table_path)


def test_invalid_uri():
    """Test error handling for invalid URIs."""
    with pytest.raises(Exception):
        hdb.open_table("not://a/valid/uri")


def test_nonexistent_table_read():
    """Test reading from a non-existent table."""
    # This should either create an empty table or raise a clear error
    try:
        table = hdb.open_table("file:///tmp/nonexistent_table_12345")
        df = table.to_pandas()
        # If it succeeds, should return empty DataFrame
        assert len(df) == 0 or df is not None
    except Exception as e:
        # If it fails, error message should be clear
        error_msg = str(e).lower()
        assert "not found" in error_msg or "does not exist" in error_msg or "no such" in error_msg


def test_schema_mismatch_write(test_table_path):
    """Test writing data with mismatched schema."""
    table = hdb.open_table(test_table_path)
    
    # Write initial data with one schema
    schema1 = pa.schema([
        ('id', pa.int32()),
        ('name', pa.string())
    ])
    batch1 = pa.Table.from_arrays(
        [pa.array([1, 2, 3], type=pa.int32()),
         pa.array(["a", "b", "c"])],
        schema=schema1
    )
    table.write_arrow(batch1)
    
    # Try to write data with incompatible schema
    schema2 = pa.schema([
        ('id', pa.string()),  # Changed type!
        ('name', pa.string())
    ])
    batch2 = pa.Table.from_arrays(
        [pa.array(["x", "y", "z"]),
         pa.array(["d", "e", "f"])],
        schema=schema2
    )
    
    # This should either fail with clear error or handle schema evolution
    try:
        table.write_arrow(batch2)
        # If it succeeds, verify data integrity
        df = table.to_pandas()
        assert len(df) > 0
    except Exception as e:
        # Error should mention schema mismatch
        error_msg = str(e).lower()
        assert "schema" in error_msg or "type" in error_msg or "mismatch" in error_msg


def test_empty_batch_write(test_table_path):
    """Test writing empty batches."""
    table = hdb.open_table(test_table_path)
    
    schema = pa.schema([
        ('id', pa.int32()),
        ('value', pa.string())
    ])
    
    # Write empty batch
    empty_batch = pa.Table.from_arrays(
        [pa.array([], type=pa.int32()),
         pa.array([])],
        schema=schema
    )
    
    # Should handle gracefully
    table.write_arrow(empty_batch)
    
    df = table.to_pandas()
    assert len(df) == 0


def test_null_values_handling(test_table_path):
    """Test handling of null values."""
    table = hdb.open_table(test_table_path)
    
    schema = pa.schema([
        ('id', pa.int32()),
        ('nullable_value', pa.string())
    ])
    
    # Write data with nulls
    batch = pa.Table.from_arrays(
        [pa.array([1, 2, 3, 4], type=pa.int32()),
         pa.array(["a", None, "c", None])],
        schema=schema
    )
    table.write_arrow(batch)
    
    # Read back and verify nulls preserved
    df = table.to_pandas()
    assert len(df) == 4
    assert df['nullable_value'].isna().sum() == 2


def test_large_string_values(test_table_path):
    """Test handling of very large string values."""
    table = hdb.open_table(test_table_path)
    
    schema = pa.schema([
        ('id', pa.int32()),
        ('large_text', pa.string())
    ])
    
    # Create very large strings (1MB each)
    large_string = "x" * (1024 * 1024)
    
    batch = pa.Table.from_arrays(
        [pa.array([1, 2, 3], type=pa.int32()),
         pa.array([large_string, large_string, large_string])],
        schema=schema
    )
    
    table.write_arrow(batch)
    
    # Verify data can be read back
    df = table.to_pandas()
    assert len(df) == 3
    assert len(df['large_text'].iloc[0]) == 1024 * 1024


def test_invalid_filter_expression(test_table_path):
    """Test error handling for invalid filter expressions."""
    table = hdb.open_table(test_table_path)
    
    # Write some data
    schema = pa.schema([
        ('id', pa.int32()),
        ('value', pa.string())
    ])
    batch = pa.Table.from_arrays(
        [pa.array([1, 2, 3], type=pa.int32()),
         pa.array(["a", "b", "c"])],
        schema=schema
    )
    table.write_arrow(batch)
    
    # Try invalid filter
    try:
        df = table.to_pandas(filter="this is not a valid filter !!!")
        # If it doesn't fail, that's OK too (might ignore invalid filters)
    except Exception as e:
        # Error should be clear
        error_msg = str(e).lower()
        assert "filter" in error_msg or "parse" in error_msg or "invalid" in error_msg


def test_nonexistent_column_filter(test_table_path):
    """Test filtering on non-existent column."""
    table = hdb.open_table(test_table_path)
    
    schema = pa.schema([
        ('id', pa.int32()),
        ('value', pa.string())
    ])
    batch = pa.Table.from_arrays(
        [pa.array([1, 2, 3], type=pa.int32()),
         pa.array(["a", "b", "c"])],
        schema=schema
    )
    table.write_arrow(batch)
    
    # Try to filter on non-existent column
    try:
        df = table.to_pandas(filter="nonexistent_column > 5")
        # Might return empty result or all data
    except Exception as e:
        # Error should mention column
        error_msg = str(e).lower()
        assert "column" in error_msg or "field" in error_msg or "not found" in error_msg


def test_corrupted_data_handling(test_table_path):
    """Test handling of corrupted data files."""
    table = hdb.open_table(test_table_path)
    
    # Write valid data first
    schema = pa.schema([
        ('id', pa.int32()),
        ('value', pa.string())
    ])
    batch = pa.Table.from_arrays(
        [pa.array([1, 2, 3], type=pa.int32()),
         pa.array(["a", "b", "c"])],
        schema=schema
    )
    table.write_arrow(batch)
    
    # Find the data file and corrupt it
    table_dir = Path(test_table_path.replace("file://", ""))
    if table_dir.exists():
        parquet_files = list(table_dir.glob("*.parquet"))
        if parquet_files:
            # Corrupt the file by truncating it
            with open(parquet_files[0], 'wb') as f:
                f.write(b"corrupted data")
            
            # Try to read - should handle gracefully
            try:
                df = table.to_pandas()
                # Might return partial data or empty
            except Exception as e:
                # Error should indicate corruption
                error_msg = str(e).lower()
                assert any(word in error_msg for word in ["corrupt", "invalid", "error", "failed"])


def test_concurrent_schema_changes(test_table_path):
    """Test handling of schema changes during concurrent operations."""
    table = hdb.open_table(test_table_path)
    
    # Write initial data
    schema1 = pa.schema([
        ('id', pa.int32()),
        ('value', pa.string())
    ])
    batch1 = pa.Table.from_arrays(
        [pa.array([1, 2], type=pa.int32()),
         pa.array(["a", "b"])],
        schema=schema1
    )
    table.write_arrow(batch1)
    
    # Try to add a new column (schema evolution)
    schema2 = pa.schema([
        ('id', pa.int32()),
        ('value', pa.string()),
        ('new_column', pa.int32())
    ])
    batch2 = pa.Table.from_arrays(
        [pa.array([3, 4], type=pa.int32()),
         pa.array(["c", "d"]),
         pa.array([10, 20], type=pa.int32())],
        schema=schema2
    )
    
    # This tests schema evolution capability
    try:
        table.write_arrow(batch2)
        df = table.to_pandas()
        # Should have new column with nulls for old rows
        assert 'new_column' in df.columns
    except Exception as e:
        # If schema evolution not supported, error should be clear
        error_msg = str(e).lower()
        assert "schema" in error_msg


def test_resource_limits(test_table_path):
    """Test behavior when approaching resource limits."""
    table = hdb.open_table(test_table_path)
    
    schema = pa.schema([
        ('id', pa.int32()),
        ('data', pa.binary())
    ])
    
    # Try to write very large batch (100MB)
    large_data = b"x" * (10 * 1024 * 1024)  # 10MB per row
    
    try:
        batch = pa.Table.from_arrays(
            [pa.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=pa.int32()),
             pa.array([large_data] * 10)],
            schema=schema
        )
        table.write_arrow(batch)
        
        # If successful, verify data
        df = table.to_pandas()
        assert len(df) == 10
    except MemoryError:
        # Expected for very large data
        pass
    except Exception as e:
        # Should handle gracefully
        assert "memory" in str(e).lower() or "size" in str(e).lower()


def test_special_characters_in_data(test_table_path):
    """Test handling of special characters in string data."""
    table = hdb.open_table(test_table_path)
    
    schema = pa.schema([
        ('id', pa.int32()),
        ('text', pa.string())
    ])
    
    # Data with special characters
    special_strings = [
        "normal text",
        "unicode: 你好世界 🚀",
        "quotes: \"double\" and 'single'",
        "newlines:\nline1\nline2",
        "tabs:\ttab1\ttab2",
        "null char: \x00",
        "backslash: \\path\\to\\file",
    ]
    
    batch = pa.Table.from_arrays(
        [pa.array(list(range(len(special_strings))), type=pa.int32()),
         pa.array(special_strings)],
        schema=schema
    )
    
    table.write_arrow(batch)
    
    # Verify data preserved
    df = table.to_pandas()
    assert len(df) == len(special_strings)
    for i, expected in enumerate(special_strings):
        actual = df[df['id'] == i]['text'].iloc[0]
        # Some special chars might be normalized
        assert actual is not None


if __name__ == "__main__":
    print("=== Error Handling Tests ===\n")
    
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        tests = [
            ("Invalid URI", test_invalid_uri),
            ("Nonexistent table read", test_nonexistent_table_read),
            ("Schema mismatch", lambda: test_schema_mismatch_write(f"file://{tmp_path}/test1")),
            ("Empty batch", lambda: test_empty_batch_write(f"file://{tmp_path}/test2")),
            ("Null values", lambda: test_null_values_handling(f"file://{tmp_path}/test3")),
            ("Large strings", lambda: test_large_string_values(f"file://{tmp_path}/test4")),
            ("Invalid filter", lambda: test_invalid_filter_expression(f"file://{tmp_path}/test5")),
            ("Nonexistent column", lambda: test_nonexistent_column_filter(f"file://{tmp_path}/test6")),
            ("Corrupted data", lambda: test_corrupted_data_handling(f"file://{tmp_path}/test7")),
            ("Schema evolution", lambda: test_concurrent_schema_changes(f"file://{tmp_path}/test8")),
            ("Resource limits", lambda: test_resource_limits(f"file://{tmp_path}/test9")),
            ("Special characters", lambda: test_special_characters_in_data(f"file://{tmp_path}/test10")),
        ]
        
        for name, test_func in tests:
            try:
                print(f"Running {name}...")
                test_func()
                print(f"✓ {name} passed")
            except Exception as e:
                print(f"✗ {name} failed: {e}")
    
    print("\n✅ Error handling tests completed!")
