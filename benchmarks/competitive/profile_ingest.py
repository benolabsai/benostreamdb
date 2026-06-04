import time
import os
os.environ['HYPERSTREAM_WAL_SYNC_INTERVAL_MS'] = '10'
os.environ['HYPERSTREAM_WAL_SYNC_BATCH_SIZE'] = '100000'
os.environ['RUST_LOG'] = 'info'
import numpy as np
import pyarrow as pa
import hyperstreamdb as hdb
import tempfile
import sys

def profile():
    n_rows = 100_000
    dim = 768
    
    print(f"Generating {n_rows} rows of {dim}-dim vectors...")
    vectors = np.random.rand(n_rows, dim).astype(np.float32)
    metadata = {
        'id': np.arange(n_rows),
        'embedding': list(vectors)
    }
    import pandas as pd
    df = pd.DataFrame(metadata)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        table = hdb.Table(f"file://{tmpdir}/test_table")
        
        start = time.time()
        
        t0 = time.time()
        print("Building PyArrow Table with FixedSizeList...")
        t0 = time.time()
        
        schema = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("embedding", pa.list_(pa.float32(), 768))
        ])
        
        # Flatten the vectors into a 1D pyarrow array, then build a FixedSizeListArray
        flat_vectors = vectors.flatten()
        flat_pa = pa.array(flat_vectors)
        fsl_pa = pa.FixedSizeListArray.from_arrays(flat_pa, 768)
        
        pa_table = pa.Table.from_arrays([pa.array(np.arange(n_rows)), fsl_pa], schema=schema)
        t1 = time.time()
        print(f"PyArrow FixedSizeList Table building took: {t1 - t0:.3f}s")
        
        print("Schema from custom pyarrow table:")
        print(pa_table.schema)
        
        t0 = time.time()
        table.write(pa_table)
        t1 = time.time()
        print(f"table.write(pa_table) took: {t1 - t0:.3f}s")
        
        t2 = time.time()
        table.commit()
        t3 = time.time()
        print(f"table.commit took: {t3 - t2:.3f}s")
        
        t4 = time.time()
        table.wait_for_background_tasks()
        t5 = time.time()
        print(f"wait_for_background_tasks took: {t5 - t4:.3f}s")

if __name__ == '__main__':
    profile()
