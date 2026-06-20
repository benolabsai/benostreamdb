import os
import shutil
import pyarrow as pa
from pyiceberg.catalog.sql import SqlCatalog
import hyperstreamdb as hdb

def test_compatibility():
    print("=== STARTING PYICEBERG COMPATIBILITY TEST ===")
    
    # 1. Setup paths
    warehouse_path = "/tmp/pyiceberg_warehouse"
    catalog_db_path = "/tmp/pyiceberg_catalog.db"
    hdb_cache_path = "/tmp/pyiceberg_hdb_cache"

    # Clean up directories from any previous runs
    shutil.rmtree(warehouse_path, ignore_errors=True)
    shutil.rmtree(hdb_cache_path, ignore_errors=True)
    if os.path.exists(catalog_db_path):
        os.remove(catalog_db_path)

    # Ensure warehouse exists
    os.makedirs(warehouse_path, exist_ok=True)

    print(f"SqlCatalog Warehouse: {warehouse_path}")
    print(f"SqlCatalog DB: {catalog_db_path}")
    print(f"HyperStreamDB cache: {hdb_cache_path}")

    # 2. Initialize SqlCatalog via PyIceberg
    catalog = SqlCatalog(
        "pyiceberg_test",
        **{
            "uri": f"sqlite:///{catalog_db_path}",
            "warehouse": f"file://{warehouse_path}",
        }
    )

    # Create Namespace
    catalog.create_namespace("default")

    # Define Schema
    schema = pa.schema([
        ("id", pa.int64()),
        ("category", pa.string()),
        ("value", pa.float64())
    ])

    table_name = "default.test_pyiceberg"

    # 3. Create table and write data using PyIceberg
    print("Writing table using PyIceberg...")
    iceberg_table = catalog.create_table(
        table_name,
        schema=schema
    )

    data = pa.Table.from_pydict({
        "id": [1, 2, 3, 4, 5],
        "category": ["A", "B", "A", "C", "B"],
        "value": [10.5, 20.0, 15.2, 5.0, 30.5]
    })
    
    iceberg_table.append(data)
    print("Successfully wrote data using PyIceberg.")

    # 4. Find the metadata JSON file created by PyIceberg
    metadata_dir = os.path.join(warehouse_path, "default", "test_pyiceberg", "metadata")
    if not os.path.exists(metadata_dir):
        raise FileNotFoundError(f"PyIceberg metadata directory not found at: {metadata_dir}")

    metadata_files = [f for f in os.listdir(metadata_dir) if f.endswith(".metadata.json")]
    if not metadata_files:
        raise FileNotFoundError(f"No .metadata.json file found in {metadata_dir}")

    # Pick the latest metadata file
    metadata_files.sort()
    latest_metadata = metadata_files[-1]
    metadata_path = f"file://{os.path.join(metadata_dir, latest_metadata)}"
    print(f"Located latest Iceberg metadata: {metadata_path}")

    # 5. Register table with HyperStreamDB
    print("Registering external Iceberg table with HyperStreamDB...")
    hdb_table = hdb.Table.register_external(f"file://{hdb_cache_path}", metadata_path)
    print("Successfully registered external table.")

    # 6. Read table using HyperStreamDB
    print("Reading table data using HyperStreamDB...")
    hdb_data = hdb_table.to_arrow()
    print("HyperStreamDB Read Result:")
    print(hdb_data)

    # 7. Assertions to verify correctness
    assert hdb_data.num_rows == 5, f"Expected 5 rows, got {hdb_data.num_rows}"
    assert set(hdb_data.column("category").to_pylist()) == {"A", "B", "C"}, "Categories mismatch"
    assert hdb_data.column("id").to_pylist() == [1, 2, 3, 4, 5], "IDs mismatch"
    assert hdb_data.column("value").to_pylist() == [10.5, 20.0, 15.2, 5.0, 30.5], "Values mismatch"

    print("=== PYICEBERG COMPATIBILITY TEST PASSED SUCCESSFULLY ===")

if __name__ == "__main__":
    test_compatibility()
