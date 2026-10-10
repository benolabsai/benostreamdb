import benostreamdb as bsdb
import h5py
import numpy as np
import pandas as pd
import os
import shutil

if os.path.exists("test_db"): shutil.rmtree("test_db")

f = h5py.File('benchmarks/competitors/data/lastfm-64-dot.hdf5', 'r')
train = f['train'][:]
test = f['test'][:]
neighbors = f['neighbors'][:]

table = bsdb.Table("test_db")
table.add_index("embedding", {"type": "hnsw_tq8", "complexity": 16, "quality": 200, "metric": "inner_product"})

df = pd.DataFrame({
    "id": np.arange(len(train), dtype=np.int64),
    "embedding": [row.tolist() for row in train]
})
table.write(df)
table.commit()
table.wait_for_background_tasks()

print("Searching...")
res = table.search("embedding", test[0].tolist(), k=10, ef_search=200)
print(res)
