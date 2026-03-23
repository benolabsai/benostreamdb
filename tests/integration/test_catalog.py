import sys
import os
import hyperstreamdb as hdb
import pytest
import time
import requests
import json

# Setup
NESSIE_URL = "http://localhost:19120"

def test_catalog_flow():
    print(f"Connecting to Nessie at {NESSIE_URL}...")
    
    # 1. Initialize Catalog
    try:
        catalog = hdb.PyNessieCatalog(NESSIE_URL)
    except AttributeError:
        # Fallback if class name differs or not exposed
        # The rust generic name is PyNessieCatalog, exposed as?
        # In lib.rs: m.add_class::<python_binding::PyNessieCatalog>()?;
        # So it should be hdb.PyNessieCatalog or hdb.NessieCatalog if renamed.
        # It's hdb.PyNessieCatalog.
        print("AttributeError: PyNessieCatalog not found in module.")
        sys.exit(1)

    # 2. Creating a table
    branch = "main"
    table = "db.test_table"
    location = "file:///tmp/warehouse/db/test_table"
    import time
    table_name = f"test_table_{int(time.time())}"
    location = f"file:///tmp/{table_name}"
    # Minimal Iceberg schema
    schema = {
        "type": "struct",
        "fields": [
            {"id": 1, "name": "id", "required": True, "type": "int"},
            {"id": 2, "name": "data", "required": False, "type": "string"}
        ]
    }
    schema_json = json.dumps(schema)
    
    print(f"Creating table {table_name} on branch {branch}...")
    try:
        catalog.create_table(branch, table_name, location, schema_json)
        print("Table created successfully.")
    except Exception as e:
        print(f"Failed to create table: {e}")
        # Proceed if it's already exists or other error to see more info
    
    # 3. Create a dev branch
    print("Creating dev branch from main...")
    try:
        catalog.create_branch("dev", "main")
        print("Branch 'dev' created.")
    except Exception as e:
        print(f"Failed to create branch: {e}")

    # 4. Verification via requests
    print("Verifying via REST API...")
    try:
        resp = requests.get(f"{NESSIE_URL}/api/v2/trees/dev")
        if resp.status_code == 200:
            print("Successfully verified 'dev' branch exists.")
        else:
            print(f"Failed to get 'dev' branch: {resp.status_code} {resp.text}")
    except Exception as e:
        print(f"Verification request failed: {e}")

if __name__ == "__main__":
    test_catalog_flow()
