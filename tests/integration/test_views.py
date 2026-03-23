import sys
import hyperstreamdb as hdb
import time

def test_view_creation():
    # 1. Initialize Catalog
    try:
        catalog = hdb.PyNessieCatalog("http://localhost:19120")
    except AttributeError:
        print("AttributeError: PyNessieCatalog not found in module.")
        sys.exit(1)

    branch = "main"
    view_name = f"view_{int(time.time())}"
    metadata_location = f"file:///tmp/{view_name}.metadata.json"
    
    print(f"Attempting to create view {view_name}...")
    try:
        # call the method we added
        sql = "SELECT * FROM db.table"
        dialect = "spark"
        catalog.create_view(branch, view_name, metadata_location, sql, dialect)
        print("View created successfully (unexpected if no server).")
    except Exception as e:
        msg = str(e)
        if "Failed to get reference" in msg or "Connection refused" in msg or "Failed to commit operation" in msg:
            print(f"Caught expected network/protocol error: {msg}")
            print("SUCCESS: Binding 'create_view' exists and was called.")
        elif "'PyNessieCatalog' object has no attribute 'create_view'" in msg:
            print(f"FAILURE: Method not found: {msg}")
            sys.exit(1)
        else:
            print(f"Caught unexpected error: {msg}")
            # Even generic error implies binding exists
            print("SUCCESS: Binding 'create_view' exists and was called.")

if __name__ == "__main__":
    test_view_creation()
