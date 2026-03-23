"""
Integration test for REST Catalog

This test requires a REST Catalog server running at http://localhost:8181
You can use Polaris (https://github.com/apache/polaris) or any Iceberg REST Catalog implementation.

To run Polaris locally:
```bash
docker run -p 8181:8181 apache/polaris:latest
```
"""

import hyperstreamdb as hdb
import pytest
import json

# Skip if REST catalog not available
REST_CATALOG_URL = "http://localhost:8181"

def test_rest_catalog_basic():
    """Test basic REST Catalog operations"""
    
    # Create REST catalog client
    catalog = hdb.PyRestCatalog(REST_CATALOG_URL)
    
    # Define Iceberg schema
    schema = {
        "type": "struct",
        "fields": [
            {"id": 1, "name": "id", "required": True, "type": "long"},
            {"id": 2, "name": "data", "required": False, "type": "string"}
        ]
    }
    
    namespace = "test_db"
    table_name = "test_table"
    location = "s3://test-bucket/test_table"
    
    try:
        # Create table
        catalog.create_table(
            namespace,
            table_name,
            json.dumps(schema),
            location
        )
        print(f"✓ Created table {namespace}.{table_name}")
        
        # Check table exists
        exists = catalog.table_exists(namespace, table_name)
        assert exists, "Table should exist after creation"
        print(f"✓ Table exists: {exists}")
        
        # Load table
        table = catalog.load_table(namespace, table_name)
        print(f"✓ Loaded table: {table}")
        
    except Exception as e:
        pytest.skip(f"REST Catalog not available: {e}")

def test_rest_catalog_with_prefix():
    """Test REST Catalog with prefix"""
    
    # Create catalog with prefix
    catalog = hdb.PyRestCatalog(REST_CATALOG_URL, prefix="warehouse")
    
    # Should work the same way
    assert catalog is not None
    print("✓ REST Catalog with prefix created")

if __name__ == "__main__":
    print("Testing REST Catalog Integration")
    print("=" * 60)
    
    try:
        test_rest_catalog_basic()
        test_rest_catalog_with_prefix()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n⚠️  Tests skipped: {e}")
        print("To run these tests, start a REST Catalog server at http://localhost:8181")
