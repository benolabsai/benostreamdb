import hyperstreamdb as hdb
import os
import pytest
from hyperstreamdb import PyNessieCatalog, PyHiveCatalog

def test_create_catalog_direct():
    print("Testing create_catalog (direct)...")
    # Nessie
    catalog = hdb.create_catalog("nessie", {"url": "http://localhost:19120"})
    assert isinstance(catalog, PyNessieCatalog)
    print("Direct Nessie creation passed.")

    # Hive (simulated)
    catalog = hdb.create_catalog("hive", {"url": "thrift://localhost:9083"})
    assert isinstance(catalog, PyHiveCatalog)
    print("Direct Hive creation passed.")
    
    # Error case
    with pytest.raises(ValueError):
        hdb.create_catalog("unknown", {})
    print("Error handling passed.")

def test_create_catalog_from_config():
    print("Testing create_catalog_from_config (TOML)...")
    config_path = os.path.abspath("test_catalog.toml")
    
    # Ensure file exists
    assert os.path.exists(config_path)
    
    catalog = hdb.create_catalog_from_config(config_path)
    assert isinstance(catalog, PyNessieCatalog)
    print("TOML Nessie creation passed.")

if __name__ == "__main__":
    test_create_catalog_direct()
    test_create_catalog_from_config()
