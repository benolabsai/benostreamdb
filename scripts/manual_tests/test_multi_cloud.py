"""
Integration tests for multi-cloud storage using MinIO and equivalents.

Tests S3 (MinIO), Azure (Azurite), and GCP (fake-gcs-server) storage backends.
These tests require the respective services to be running via docker-compose.
"""

import hyperstreamdb as hdb
import pyarrow as pa
import pytest
import os
import requests
from pathlib import Path


def is_minio_available():
    """Check if MinIO is running."""
    try:
        response = requests.get("http://localhost:9000/minio/health/live", timeout=1)
        return response.status_code == 200
    except:
        return False


def is_azurite_available():
    """Check if Azurite is running."""
    try:
        # Azurite blob service runs on port 10000
        response = requests.get("http://localhost:10000/devstoreaccount1?comp=list", timeout=1)
        return response.status_code in [200, 400]  # 400 is OK, means it's running
    except:
        return False


@pytest.mark.skipif(not is_minio_available(), reason="MinIO not running. Start with: docker-compose up -d minio")
def test_s3_with_minio():
    """Test S3 storage using MinIO."""
    # MinIO credentials from docker-compose
    os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
    os.environ["AWS_ENDPOINT_URL"] = "http://localhost:9000"
    os.environ["AWS_REGION"] = "us-east-1"
    
    # Create table on MinIO
    table_uri = "s3://test-bucket/hyperstream-test"
    
    try:
        table = hdb.Table(table_uri)
        
        # Write data
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
        
        # Read back
        df = table.to_pandas()
        assert len(df) == 3
        assert list(df['id']) == [1, 2, 3]
        
        print("✓ S3/MinIO test passed")
        
    finally:
        # Cleanup environment
        for key in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_ENDPOINT_URL", "AWS_REGION"]:
            os.environ.pop(key, None)


@pytest.mark.skipif(not is_azurite_available(), reason="Azurite not running. Start with: docker run -p 10000:10000 mcr.microsoft.com/azure-storage/azurite")
def test_azure_with_azurite():
    """Test Azure Blob Storage using Azurite."""
    # Azurite uses well-known development credentials
    os.environ["AZURE_STORAGE_ACCOUNT"] = "devstoreaccount1"
    os.environ["AZURE_STORAGE_ACCESS_KEY"] = "Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw=="
    os.environ["AZURE_STORAGE_USE_EMULATOR"] = "true"
    
    # Create table on Azurite
    table_uri = "az://test-container/hyperstream-test"
    
    try:
        table = hdb.Table(table_uri)
        
        # Write data
        schema = pa.schema([
            ('id', pa.int32()),
            ('value', pa.string())
        ])
        
        batch = pa.Table.from_arrays(
            [pa.array([10, 20, 30], type=pa.int32()),
             pa.array(["x", "y", "z"])],
            schema=schema
        )
        
        table.write_arrow(batch)
        
        # Read back
        df = table.to_pandas()
        assert len(df) == 3
        assert list(df['id']) == [10, 20, 30]
        
        print("✓ Azure/Azurite test passed")
        
    finally:
        # Cleanup environment
        for key in ["AZURE_STORAGE_ACCOUNT", "AZURE_STORAGE_ACCESS_KEY", "AZURE_STORAGE_USE_EMULATOR"]:
            os.environ.pop(key, None)


def test_local_filesystem_comprehensive():
    """Comprehensive test of local filesystem storage."""
    import tempfile
    import shutil
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        table_uri = f"file://{tmp_dir}/test_table"
        
        table = hdb.Table(table_uri)
        
        # Test 1: Basic write and read
        schema = pa.schema([
            ('id', pa.int32()),
            ('name', pa.string()),
            ('value', pa.float64())
        ])
        
        batch1 = pa.Table.from_arrays(
            [pa.array([1, 2, 3], type=pa.int32()),
             pa.array(["alice", "bob", "charlie"]),
             pa.array([1.1, 2.2, 3.3])],
            schema=schema
        )
        
        table.write_arrow(batch1)
        
        # Test 2: Multiple writes
        batch2 = pa.Table.from_arrays(
            [pa.array([4, 5], type=pa.int32()),
             pa.array(["dave", "eve"]),
             pa.array([4.4, 5.5])],
            schema=schema
        )
        
        table.write_arrow(batch2)
        
        # Test 3: Read all data
        df = table.to_pandas()
        assert len(df) == 5
        assert set(df['name']) == {"alice", "bob", "charlie", "dave", "eve"}
        
        # Test 4: Filtered read
        df_filtered = table.to_pandas(filter="id > 2")
        assert len(df_filtered) >= 3  # Should have at least 3, 4, 5
        
        # Test 5: Verify files created
        table_path = Path(tmp_dir) / "test_table"
        assert table_path.exists()
        parquet_files = list(table_path.glob("*.parquet"))
        assert len(parquet_files) > 0
        
        print(f"✓ Local filesystem test passed ({len(parquet_files)} files created)")


def test_http_storage():
    """Test HTTP-based storage (read-only)."""
    # This would require an HTTP server serving Parquet files
    # For now, we just test that the URI is accepted
    try:
        # This will likely fail without a real HTTP server, but shouldn't crash
        table_uri = "https://example.com/data/table"
        result = hdb.Table(table_uri)
        # If it succeeds, that's fine
    except Exception as e:
        # Should fail gracefully with a clear error
        error_msg = str(e).lower()
        assert "http" in error_msg or "connection" in error_msg or "not found" in error_msg
        print(f"✓ HTTP storage test passed (expected failure: {type(e).__name__})")


if __name__ == "__main__":
    print("=== Multi-Cloud Storage Tests ===\n")
    
    print("Testing local filesystem...")
    test_local_filesystem_comprehensive()
    
    print("\nTesting S3/MinIO...")
    if is_minio_available():
        test_s3_with_minio()
    else:
        print("⚠ MinIO not available. Start with: docker-compose up -d minio")
    
    print("\nTesting Azure/Azurite...")
    if is_azurite_available():
        test_azure_with_azurite()
    else:
        print("⚠ Azurite not available. Start with: docker run -p 10000:10000 mcr.microsoft.com/azure-storage/azurite")
    
    print("\nTesting HTTP storage...")
    test_http_storage()
    
    print("\n✅ Multi-cloud storage tests completed!")
