import benostreamdb as bs
import numpy as np

t = bs.open_table("/home/ralbright/projects/benolabs/benchmark_db")
res = t.vector_search_scored("embedding", [0.5]*128, 10)
print(f"Results length: {len(res)}")
