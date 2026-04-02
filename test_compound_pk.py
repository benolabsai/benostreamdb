import hyperstreamdb as hdb
import pandas as pd
import numpy as np
import os
import shutil

# Setup
db_path = "/tmp/hdb_compound_pk_test"
if os.path.exists(db_path):
    shutil.rmtree(db_path)

# 1. Create a table with Compound Primary Key
# We'll use 'user_id' and 'timestamp' as the identity
table = hdb.Table(f"file://{db_path}", primary_key=["user_id", "timestamp"])

# 2. Ingest some data
data = pd.DataFrame({
    "user_id": [1, 2, 1, 3],
    "timestamp": [1000, 1000, 2000, 1000],
    "value": [10.5, 20.0, 30.5, 40.0],
    "vector": [np.random.rand(128).tolist() for _ in range(4)]
})

print(f"Ingesting {len(data)} rows with Compound PK...")
table.write(data)
table.commit()
table.wait_for_background_tasks()

# 3. Verify PK 
print(f"Current PK: {table.primary_key}")

# 4. Ingest updated data for existing PK (Deduplication check)
# (1, 1000) is already there. We want to 'upsert' it by rewriting.
new_data = pd.DataFrame({
    "user_id": [1],
    "timestamp": [1000],
    "value": [999.9], # New value
    "vector": [np.random.rand(128).tolist() for _ in range(1)]
})

print("Performing upsert on (user_id=1, timestamp=1000)...")
table.upsert(new_data, key_column=["user_id", "timestamp"])

# MUST commit to see changes and trigger deduplication
print("Committing changes...")
table.commit()

# Ensure background indexing finishes
table.wait_for_background_tasks()

# 5. Check if it worked
# (In current implementation, write() appends. We need to implement the 'upsert' logic if not already handled by default write() for PK tables)
# Actually, the user's objective was to *enable* this. 

# 6. Test Hybrid Query (RRF placeholder check)
# res = table.query(vector=np.random.rand(128).tolist(), filter="user_id = 1")
# print(f"Query results: {len(res)}")

summary = table.to_pandas()
summary = table.to_pandas()
print("\nFinal Table Data:")
print(summary)

print("\nDebug Info:")
print(f"Schema columns: {table.to_arrow().schema.names if table.to_arrow().num_rows > 0 else 'No columns found'}")
print(f"Table segments on disk: {os.listdir(db_path)}")
if os.path.exists(os.path.join(db_path, "metadata")):
    print(f"Metadata files: {os.listdir(os.path.join(db_path, 'metadata'))}")

# Cleanup
# shutil.rmtree(db_path)
