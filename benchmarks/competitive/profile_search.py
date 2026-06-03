import cProfile
import pstats
import time
import numpy as np
import hyperstreamdb as hdb
import tempfile
import sys

def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        table = hdb.Table(f"file://{tmpdir}/test_table")
        
        n_rows = 1000
        dim = 768
        np.random.seed(42)
        vectors = np.random.rand(n_rows, dim).astype(np.float32)
        
        table.add_pq_index("embedding", compression=8)
        import pandas as pd
        df = pd.DataFrame({'embedding': list(vectors)})
        table.write_pandas(df)
        table.commit()
        table.wait_for_background_tasks()
        
        query_vector = np.random.rand(dim).astype(np.float32)
        
        # Warmup
        table.search('embedding', query_vector.tolist(), k=10)
        
        def run_search():
            for _ in range(50):
                table.search('embedding', query_vector.tolist(), k=10)
        
        print("Starting profiling...")
        start = time.time()
        cProfile.runctx('run_search()', globals(), locals(), 'search.prof')
        elapsed = time.time() - start
        
        print(f"Elapsed time for 50 searches: {elapsed:.3f} s ({(elapsed*1000)/50:.3f} ms/search)")
        
        p = pstats.Stats('search.prof')
        p.strip_dirs().sort_stats('cumtime').print_stats(20)

if __name__ == "__main__":
    main()
