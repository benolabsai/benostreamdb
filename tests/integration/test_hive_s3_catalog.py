import os
import time
import pandas as pd
import numpy as np
import pytest
import hyperstreamdb as hdb

# We will connect to the local Hive Metastore service exposed on port 9083
HIVE_METASTORE_URL = "thrift://localhost:9083"

def test_hive_s3_catalog_integration():
    """Verify that PyHiveCatalog can create a table on S3 (MinIO), write data, and query it back."""
    # Set S3 / MinIO environment variables for the Rust object store client
    os.environ["AWS_ENDPOINT_URL"] = "http://localhost:9000"
    os.environ["AWS_ACCESS_KEY_ID"] = "admin"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "admin123"
    
    # 1. Initialize Hive Metastore Catalog
    try:
        catalog = hdb.create_catalog("hive", {"url": HIVE_METASTORE_URL})
    except Exception as e:
        pytest.skip(f"Hive Metastore not available: {e}")

    # 2. Define schema
    schema = hdb.Schema([
        hdb.Field("id", hdb.DataType.int64()),
        hdb.Field("name", hdb.DataType.string()),
        hdb.Field("value", hdb.DataType.float64())
    ])

    # 3. Create a unique table in Hive Metastore on MinIO S3 storage
    table_name = f"hdb_s3_test_{int(time.time())}"
    location = f"s3a://warehouse/{table_name}"
    
    print(f"Creating table {table_name} at {location}...")
    catalog.create_table("default", table_name, schema, location)
    
    # Verify table existence in catalog
    assert catalog.table_exists("default", table_name) is True
    print("Table created and verified in catalog.")

    # 4. Load the table back from the catalog
    table = catalog.load_table("default", table_name)
    
    # 5. Write some sample data to S3
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "value": [10.5, 20.0, 30.5]
    })
    
    print("Writing data to S3...")
    table.write(df)
    table.commit()
    print("Data committed successfully.")

    # 6. Read the data back and verify correctness
    df_read = table.to_pandas()
    print("Read results:")
    print(df_read)
    
    assert len(df_read) == 3, f"Expected 3 rows, got {len(df_read)}"
    assert list(df_read["id"]) == [1, 2, 3]
    assert list(df_read["name"]) == ["Alice", "Bob", "Charlie"]
    assert list(df_read["value"]) == [10.5, 20.0, 30.5]
    
    print("All assertions passed!")
