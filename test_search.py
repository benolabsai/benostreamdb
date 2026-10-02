import benostreamdb as bs
import numpy as np
import time

t = bs.open_table("/home/ralbright/projects/benolabs/benchmark_db")
q = np.random.rand(128).astype(np.float32)
res = t.vector_search_scored("embedding", q.tolist(), 10)
print(f"Results: {len(res)}")
