import pyarrow as pa
import benostreamdb as bsdb
import time
import os
import psutil

t = bsdb.Table.create("file:///tmp/hdb_test_oom", pa.schema([("id", pa.int64()), ("embedding", pa.list_(pa.float32(), 768))]))
t.add_index("embedding", {"type": "hnsw", "device": "cpu"})

print("Starting writes...")
for i in range(100):
    # Create batch of 250,000 vectors
    ids = pa.array(range(i*250000, (i+1)*250000), type=pa.int64())
    import numpy as np
    emb_data = np.random.rand(250000 * 768).astype(np.float32)
    emb = pa.FixedSizeListArray.from_arrays(pa.array(emb_data), 768)
    batch = pa.RecordBatch.from_arrays([ids, emb], names=["id", "embedding"])
    
    t0 = time.time()
    t.write(pa.Table.from_batches([batch]))
    rss = psutil.Process(os.getpid()).memory_info().rss / 1e9
    print(f"Batch {i}: {time.time()-t0:.2f}s, RSS: {rss:.2f} GB")

t.commit()
t.wait_for_background_tasks()
print("Done!")
