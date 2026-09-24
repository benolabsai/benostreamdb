import benostreamdb as bsdb
import os
import pytest
from benostreamdb import PyNessieCatalog

def test_load_default_catalog():
    print("Testing load_default_catalog...")
    
    # create a dummy benostream.toml in current directory
    config_content = """
    catalog_type = "nessie"
    [config]
    url = "http://localhost:19120"
    """
    
    with open("benostream.toml", "w") as f:
        f.write(config_content)
        
    try:
        catalog = bsdb.load_default_catalog()
        assert isinstance(catalog, PyNessieCatalog)
        print("Default catalog loading passed.")
    finally:
        if os.path.exists("benostream.toml"):
            os.remove("benostream.toml")

if __name__ == "__main__":
    test_load_default_catalog()
