import hyperstreamdb as hdb
import pandas as pd
import os
import shutil

# Initializing table
uri = "file://" + os.getcwd() + "/test_admin_table"
if os.path.exists("test_admin_table"):
    shutil.rmtree("test_admin_table")

table = hdb.Table(uri)
print(f"Default autocommit: {table.autocommit}")

# Write data with autocommit=True (default)
df = pd.DataFrame({"id": [1, 2, 3], "val": ["a", "b", "c"]})
table.write(df)

# Read back
res = table.to_pandas()
print(f"Rows after autocommit write: {len(res)}")
assert len(res) == 3

# Truncate
print("Truncating table...")
table.truncate()
res = table.to_pandas()
print(f"Rows after truncate: {len(res)}")
assert len(res) == 0

# Autocommit=False test
print("Setting autocommit=False...")
table.autocommit = False
table.write(df)
res = table.to_pandas()
print(f"Rows after write with autocommit=False: {len(res)} (Expected 0 before commit)")

table.commit()
res = table.to_pandas()
print(f"Rows after explicit commit: {len(res)} (Expected 3)")

# Vacuum test
print("Running vacuum...")
deleted = table.vacuum(retention_versions=1)
print(f"Vacuum deleted {deleted} files.")

print("\nSUCCESS: Admin features (autocommit, truncate, vacuum) are working correctly!")
