import hyperstreamdb as hdb
import os
import pytest
from hyperstreamdb import PyNessieCatalog

def test_load_default_catalog():
    print("Testing load_default_catalog...")
    
    # create a dummy hyperstream.toml in current directory
    config_content = """
    catalog_type = "nessie"
    [config]
    url = "http://localhost:19120"
    """
    
    with open("hyperstream.toml", "w") as f:
        f.write(config_content)
        
    try:
        catalog = hdb.load_default_catalog()
        assert isinstance(catalog, PyNessieCatalog)
        print("Default catalog loading passed.")
    finally:
        if os.path.exists("hyperstream.toml"):
            os.remove("hyperstream.toml")

if __name__ == "__main__":
    test_load_default_catalog()
