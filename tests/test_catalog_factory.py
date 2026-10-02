import benostreamdb as bsdb
import os
import pytest
from benostreamdb import PyNessieCatalog, PyHiveCatalog

def test_create_catalog_direct():
    print("Testing create_catalog (direct)...")
    # Nessie
    catalog = bsdb.create_catalog("nessie", {"url": "http://localhost:19120"})
    assert isinstance(catalog, PyNessieCatalog)
    print("Direct Nessie creation passed.")

    # Hive (simulated)
    catalog = bsdb.create_catalog("hive", {"url": "thrift://localhost:9083"})
    assert isinstance(catalog, PyHiveCatalog)
    print("Direct Hive creation passed.")
    
    # Error case
    with pytest.raises(ValueError):
        bsdb.create_catalog("unknown", {})
    print("Error handling passed.")

def test_create_catalog_from_config(tmp_path):
    print("Testing create_catalog_from_config (TOML)...")
    # Write the config into a temp file rather than depending on a repo-root
    # `test_catalog.toml` that is not checked in.
    config_path = tmp_path / "test_catalog.toml"
    config_path.write_text(
        'catalog_type = "nessie"\n\n[config]\nurl = "http://localhost:19120"\n'
    )

    catalog = bsdb.create_catalog_from_config(str(config_path))
    assert isinstance(catalog, PyNessieCatalog)
    print("TOML Nessie creation passed.")

if __name__ == "__main__":
    test_create_catalog_direct()
    test_create_catalog_from_config()
