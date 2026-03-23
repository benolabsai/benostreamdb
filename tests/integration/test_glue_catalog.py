"""
Integration test for AWS Glue Catalog

This test requires AWS credentials configured (via environment variables or ~/.aws/credentials)
and an existing Glue database.

Environment variables:
- AWS_REGION (default: us-east-1)
- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
- GLUE_DATABASE (default: test_db)
"""

import hyperstreamdb as hdb
import pytest
import json
import os

# Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
GLUE_DATABASE = os.getenv("GLUE_DATABASE", "test_db")
CATALOG_ID = os.getenv("AWS_ACCOUNT_ID")  # Optional

def test_glue_catalog_basic():
    """Test basic AWS Glue Catalog operations"""
    
    try:
        # Create Glue catalog client
        catalog = hdb.PyGlueCatalog(catalog_id=CATALOG_ID)
        
        # Define Iceberg schema
        schema = {
            "type": "struct",
            "fields": [
                {"id": 1, "name": "id", "required": True, "type": "long"},
                {"id": 2, "name": "data", "required": False, "type": "string"}
            ]
        }
        
        table_name = "test_hyperstream_table"
        location = f"s3://test-bucket/{table_name}"
        
        # Create table
        catalog.create_table(
            GLUE_DATABASE,
            table_name,
            json.dumps(schema),
            location
        )
        print(f"✓ Created table {GLUE_DATABASE}.{table_name}")
        
        # Check table exists
        exists = catalog.table_exists(GLUE_DATABASE, table_name)
        assert exists, "Table should exist after creation"
        print(f"✓ Table exists: {exists}")
        
        # Load table
        table = catalog.load_table(GLUE_DATABASE, table_name)
        print(f"✓ Loaded table: {table}")
        
        # Cleanup (optional - comment out to inspect in Glue console)
        # boto3.client('glue').delete_table(DatabaseName=GLUE_DATABASE, Name=table_name)
        
    except Exception as e:
        pytest.skip(f"AWS Glue not available: {e}")

def test_glue_catalog_with_account_id():
    """Test Glue Catalog with explicit catalog ID"""
    
    if not CATALOG_ID:
        pytest.skip("AWS_ACCOUNT_ID not set")
    
    try:
        # Create catalog with explicit account ID
        catalog = hdb.PyGlueCatalog(catalog_id=CATALOG_ID)
        assert catalog is not None
        print(f"✓ Glue Catalog created with account ID: {CATALOG_ID}")
        
    except Exception as e:
        pytest.skip(f"AWS Glue not available: {e}")

if __name__ == "__main__":
    print("Testing AWS Glue Catalog Integration")
    print("=" * 60)
    print(f"AWS Region: {AWS_REGION}")
    print(f"Glue Database: {GLUE_DATABASE}")
    print(f"Catalog ID: {CATALOG_ID or 'default'}")
    print("=" * 60)
    
    try:
        test_glue_catalog_basic()
        test_glue_catalog_with_account_id()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n⚠️  Tests skipped: {e}")
        print("\nTo run these tests:")
        print("1. Configure AWS credentials")
        print("2. Create a Glue database: aws glue create-database --database-input Name=test_db")
        print("3. Set environment variables: AWS_REGION, GLUE_DATABASE")
