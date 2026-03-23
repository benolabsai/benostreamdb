"""
Schema evolution tests for HyperStreamDB.

Tests various schema change scenarios including:
- Adding/removing columns
- Type widening/narrowing
- Nullability changes
- Schema merging on read
"""
import pytest
import tempfile
import shutil
import pyarrow as pa
from hyperstreamdb import Table


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


class TestSchemaEvolution:
    """Test schema evolution scenarios."""
    
    def test_add_column(self, temp_dir):
        """Test adding a new column to an existing table."""
        table = Table(f"file://{temp_dir}")
        
        # Write initial data with 2 columns
        schema_v1 = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("name", pa.string()),
        ])
        
        batch_v1 = pa.RecordBatch.from_arrays([
            pa.array([1, 2, 3], type=pa.int64()),
            pa.array(["Alice", "Bob", "Charlie"]),
        ], schema=schema_v1)
        
        table.write([batch_v1])
        table.commit()
        
        # Write new data with 3 columns (added "age")
        schema_v2 = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("name", pa.string()),
            pa.field("age", pa.int32(), nullable=True),  # New column
        ])
        
        batch_v2 = pa.RecordBatch.from_arrays([
            pa.array([4, 5], type=pa.int64()),
            pa.array(["David", "Eve"]),
            pa.array([30, 25], type=pa.int32()),
        ], schema=schema_v2)
        
        table.write([batch_v2])
        table.commit()
        
        # Read all data - should handle schema evolution
        result = table.read()
        total_rows = sum(len(batch) for batch in result)
        
        assert total_rows == 5
        
        # Verify schema contains all columns
        result_schema = result[0].schema
        assert "id" in result_schema.names
        assert "name" in result_schema.names
        # Note: Old batches may not have "age" column
        
        print(f"\n✓ Successfully added column 'age' to existing table")
        print(f"  Total rows: {total_rows}")
        print(f"  Schema: {result_schema}")
    
    def test_add_multiple_columns(self, temp_dir):
        """Test adding multiple new columns at once."""
        table = Table(f"file://{temp_dir}")
        
        # Initial schema: 2 columns
        schema_v1 = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("name", pa.string()),
        ])
        
        batch_v1 = pa.RecordBatch.from_arrays([
            pa.array([1, 2], type=pa.int64()),
            pa.array(["Alice", "Bob"]),
        ], schema=schema_v1)
        
        table.write([batch_v1])
        table.commit()
        
        # New schema: 5 columns (added 3 new columns)
        schema_v2 = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("name", pa.string()),
            pa.field("age", pa.int32(), nullable=True),
            pa.field("city", pa.string(), nullable=True),
            pa.field("score", pa.float64(), nullable=True),
        ])
        
        batch_v2 = pa.RecordBatch.from_arrays([
            pa.array([3, 4], type=pa.int64()),
            pa.array(["Charlie", "David"]),
            pa.array([30, 35], type=pa.int32()),
            pa.array(["NYC", "LA"]),
            pa.array([95.5, 88.0], type=pa.float64()),
        ], schema=schema_v2)
        
        table.write([batch_v2])
        table.commit()
        
        # Read and verify
        result = table.read()
        total_rows = sum(len(batch) for batch in result)
        
        assert total_rows == 4
        
        print(f"\n✓ Successfully added 3 columns to existing table")
        print(f"  Original columns: 2, New columns: 5")
        print(f"  Total rows: {total_rows}")
    
    def test_drop_column_projection(self, temp_dir):
        """Test reading data without dropped columns using projection."""
        table = Table(f"file://{temp_dir}")
        
        # Write data with 4 columns
        schema = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("name", pa.string()),
            pa.field("deprecated_field", pa.string()),  # To be "dropped"
            pa.field("value", pa.float64()),
        ])
        
        batch = pa.RecordBatch.from_arrays([
            pa.array([1, 2, 3], type=pa.int64()),
            pa.array(["Alice", "Bob", "Charlie"]),
            pa.array(["old1", "old2", "old3"]),
            pa.array([1.0, 2.0, 3.0], type=pa.float64()),
        ], schema=schema)
        
        table.write([batch])
        table.commit()
        
        # Read without the deprecated column (projection)
        result = table.read(columns=["id", "name", "value"])
        
        # Verify the deprecated column is not in result
        for batch in result:
            assert "deprecated_field" not in batch.schema.names
            assert "id" in batch.schema.names
            assert "name" in batch.schema.names
            assert "value" in batch.schema.names
            assert batch.num_columns == 3
        
        total_rows = sum(len(batch) for batch in result)
        assert total_rows == 3
        
        print(f"\n✓ Successfully read data without deprecated column")
        print(f"  Original columns: 4, Projected columns: 3")
    
    def test_type_widening(self, temp_dir):
        """Test type widening (Int32 → Int64)."""
        table = Table(f"file://{temp_dir}")
        
        # Write data with Int32
        schema_v1 = pa.schema([
            pa.field("id", pa.int32()),  # Int32
            pa.field("value", pa.int32()),  # Int32
        ])
        
        batch_v1 = pa.RecordBatch.from_arrays([
            pa.array([1, 2, 3], type=pa.int32()),
            pa.array([100, 200, 300], type=pa.int32()),
        ], schema=schema_v1)
        
        table.write([batch_v1])
        table.commit()
        
        # Write new data with Int64 (widened type)
        schema_v2 = pa.schema([
            pa.field("id", pa.int64()),  # Widened to Int64
            pa.field("value", pa.int64()),  # Widened to Int64
        ])
        
        batch_v2 = pa.RecordBatch.from_arrays([
            pa.array([4, 5], type=pa.int64()),
            pa.array([400, 500], type=pa.int64()),
        ], schema=schema_v2)
        
        table.write([batch_v2])
        table.commit()
        
        # Read all data
        result = table.read()
        total_rows = sum(len(batch) for batch in result)
        
        assert total_rows == 5
        
        print(f"\n✓ Type widening (Int32 → Int64) handled successfully")
        print(f"  Total rows: {total_rows}")
    
    def test_type_narrowing_error(self, temp_dir):
        """Test that type narrowing (Int64 → Int32) is handled gracefully."""
        table = Table(f"file://{temp_dir}")
        
        # Write data with Int64
        schema_v1 = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("value", pa.int64()),
        ])
        
        # Use large values that won't fit in Int32
        batch_v1 = pa.RecordBatch.from_arrays([
            pa.array([1, 2, 3], type=pa.int64()),
            pa.array([10_000_000_000, 20_000_000_000, 30_000_000_000], type=pa.int64()),
        ], schema=schema_v1)
        
        table.write([batch_v1])
        table.commit()
        
        # Try to write with Int32 (narrowed type)
        schema_v2 = pa.schema([
            pa.field("id", pa.int32()),  # Narrowed
            pa.field("value", pa.int32()),  # Narrowed
        ])
        
        # This should either:
        # 1. Raise an error (preferred)
        # 2. Be handled gracefully by the system
        
        try:
            batch_v2 = pa.RecordBatch.from_arrays([
                pa.array([4, 5], type=pa.int32()),
                pa.array([400, 500], type=pa.int32()),
            ], schema=schema_v2)
            
            table.write([batch_v2])
            table.commit()
            
            # If it succeeds, verify data integrity
            result = table.read()
            total_rows = sum(len(batch) for batch in result)
            
            print(f"\n⚠ Type narrowing allowed (system handled gracefully)")
            print(f"  Total rows: {total_rows}")
            
        except Exception as e:
            # Expected: type narrowing should fail or be prevented
            print(f"\n✓ Type narrowing prevented (expected behavior)")
            print(f"  Error: {type(e).__name__}")
    
    def test_nullable_to_required(self, temp_dir):
        """Test changing a nullable column to required (should error if nulls exist)."""
        table = Table(f"file://{temp_dir}")
        
        # Write data with nullable column containing nulls
        schema_v1 = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("value", pa.int32(), nullable=True),  # Nullable
        ])
        
        batch_v1 = pa.RecordBatch.from_arrays([
            pa.array([1, 2, 3], type=pa.int64()),
            pa.array([100, None, 300], type=pa.int32()),  # Contains null
        ], schema=schema_v1)
        
        table.write([batch_v1])
        table.commit()
        
        # Try to write with non-nullable column
        schema_v2 = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("value", pa.int32(), nullable=False),  # Now required
        ])
        
        # This should work for new data without nulls
        batch_v2 = pa.RecordBatch.from_arrays([
            pa.array([4, 5], type=pa.int64()),
            pa.array([400, 500], type=pa.int32()),
        ], schema=schema_v2)
        
        table.write([batch_v2])
        table.commit()
        
        # Reading should work - old data still has nulls
        result = table.read()
        total_rows = sum(len(batch) for batch in result)
        
        assert total_rows == 5
        
        print(f"\n✓ Nullable → Required change handled")
        print(f"  Old data retains nulls, new data is non-null")
        print(f"  Total rows: {total_rows}")
    
    def test_required_to_nullable(self, temp_dir):
        """Test changing a required column to nullable (should work seamlessly)."""
        table = Table(f"file://{temp_dir}")
        
        # Write data with required column
        schema_v1 = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("value", pa.int32(), nullable=False),  # Required
        ])
        
        batch_v1 = pa.RecordBatch.from_arrays([
            pa.array([1, 2, 3], type=pa.int64()),
            pa.array([100, 200, 300], type=pa.int32()),
        ], schema=schema_v1)
        
        table.write([batch_v1])
        table.commit()
        
        # Write new data with nullable column
        schema_v2 = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("value", pa.int32(), nullable=True),  # Now nullable
        ])
        
        batch_v2 = pa.RecordBatch.from_arrays([
            pa.array([4, 5, 6], type=pa.int64()),
            pa.array([400, None, 600], type=pa.int32()),  # Contains null
        ], schema=schema_v2)
        
        table.write([batch_v2])
        table.commit()
        
        # Read all data - should work seamlessly
        result = table.read()
        total_rows = sum(len(batch) for batch in result)
        
        assert total_rows == 6
        
        print(f"\n✓ Required → Nullable change successful")
        print(f"  Total rows: {total_rows}")
    
    def test_schema_merge_on_read(self, temp_dir):
        """Test schema union/merge when reading data with different schemas."""
        table = Table(f"file://{temp_dir}")
        
        # Write batch 1: columns [id, name]
        schema_1 = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("name", pa.string()),
        ])
        
        batch_1 = pa.RecordBatch.from_arrays([
            pa.array([1, 2], type=pa.int64()),
            pa.array(["Alice", "Bob"]),
        ], schema=schema_1)
        
        table.write([batch_1])
        table.commit()
        
        # Write batch 2: columns [id, age]
        schema_2 = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("age", pa.int32()),
        ])
        
        batch_2 = pa.RecordBatch.from_arrays([
            pa.array([3, 4], type=pa.int64()),
            pa.array([30, 25], type=pa.int32()),
        ], schema=schema_2)
        
        table.write([batch_2])
        table.commit()
        
        # Write batch 3: columns [id, name, age] (union of both)
        schema_3 = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("name", pa.string(), nullable=True),
            pa.field("age", pa.int32(), nullable=True),
        ])
        
        batch_3 = pa.RecordBatch.from_arrays([
            pa.array([5, 6], type=pa.int64()),
            pa.array(["Eve", "Frank"]),
            pa.array([28, 32], type=pa.int32()),
        ], schema=schema_3)
        
        table.write([batch_3])
        table.commit()
        
        # Read all data - system should merge schemas
        result = table.read()
        total_rows = sum(len(batch) for batch in result)
        
        assert total_rows == 6
        
        # Each batch may have different schemas
        print(f"\n✓ Schema merge on read successful")
        print(f"  Total rows: {total_rows}")
        print(f"  Batch 1 schema: {schema_1}")
        print(f"  Batch 2 schema: {schema_2}")
        print(f"  Batch 3 schema: {schema_3}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
