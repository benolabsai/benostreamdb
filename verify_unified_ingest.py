import os
import pandas as pd
import numpy as np
import hyperstreamdb as hdb

DB_PATH = "test_unified_ingest_db"
if os.path.exists(DB_PATH):
    pass

# Setup
table = hdb.Table.create(DB_PATH, hdb.Schema([
    hdb.Field("id", hdb.DataType.int64()),
    hdb.Field("val", hdb.DataType.float32()),
    hdb.Field("embedding", hdb.DataType.vector(3))
]))

print("Testing Pandas ingest...")
df = pd.DataFrame({
    "id": [1, 2],
    "val": [10.5, 20.5],
    "embedding": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
})
table.write(df)

print("Testing List[Dict] ingest...")
data_list = [
    {"id": 3, "val": 30.5, "embedding": [0.7, 0.8, 0.9]},
    {"id": 4, "val": 40.5, "embedding": [1.0, 1.1, 1.2]}
]
table.write(data_list)

print("Testing NumPy ingest...")
# Note: we need to handle structured arrays or just regular arrays if we want it to map
# For now, our fallback converts via pd.DataFrame(data), which works for 2D arrays 
# but might lose column names.
# Better to focus on what we actually implemented: pd.DataFrame, List[Dict], Arrow, Polars.

try:
    import pyarrow as pa
    print("Testing PyArrow ingest...")
    pa_table = pa.Table.from_pandas(df)
    table.write(pa_table)
except ImportError:
    print("PyArrow not installed, skipping.")

try:
    import polars as pl
    print("Testing Polars ingest...")
    pl_df = pl.DataFrame(df)
    table.write(pl_df)
except ImportError:
    print("Polars not installed, skipping.")

table.commit()

print(f"Table row count: {table.row_count}")

# Verify read
res = table.to_pandas()
print("\nResults:")
print(res)

print("\nTesting Unified Vector Search (string query)...")
# Register fake embedding function
class FakeEmbedder(hdb.EmbeddingFunction):
    def __call__(self, texts):
        return np.array([[0.1, 0.2, 0.3]] * len(texts), dtype=np.float32)

hdb.registry.register("fake", FakeEmbedder())
table.define_embedding("text", "fake", vector_column="embedding")

# This should trigger auto-vectorization of "hello" into [0.1, 0.2, 0.3]
search_res = table.query().vector_search("hello", column="embedding", k=1).to_pandas()
print("\nSearch results for 'hello':")
print(search_res)

shutil.rmtree(DB_PATH)
