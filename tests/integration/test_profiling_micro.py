import sys
import time
import hyperstreamdb as hdb
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path
import shutil

def test_profiling():
    # 1. Setup Data
    num_docs = 5000
    data_dir = Path("tests/data/profiling_micro")
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    schema = pa.schema([
        ('id', pa.int64()),
        ('category', pa.string()),
        ('val', pa.int64())
    ])
    
    table_data = {
        'id': range(num_docs),
        'category': ['science' if i % 2 == 0 else 'art' for i in range(num_docs)],
        'val': range(num_docs)
    }
    arrow_table = pa.Table.from_pydict(table_data, schema=schema)
    pq.write_table(arrow_table, data_dir / "file1.parquet")
    
    # 2. Ingest
    db_path = "/tmp/hyperstream_test/profiling_micro"
    if Path(db_path).exists():
        shutil.rmtree(db_path)
        
    table = hdb.PyTable(f"file://{db_path}")
    # Scalar index only
    table.add_index_columns(["category"])
    
    print("Ingesting...")
    table.write_arrow(arrow_table)
    print("Ingest complete.")
    
    # 3. Warm up
    print("Warming up...")
    _ = table.to_pandas(filter="category = 'science'")
    
    # 4. Profile
    print("Running 100 queries...")
    start = time.time()
    for _ in range(100):
        _ = table.to_pandas(filter="category = 'science'")
    duration = (time.time() - start) * 1000
    print(f"Total time for 100 queries: {duration:.2f}ms")
    print(f"Avg time per query: {duration/100:.2f}ms")

if __name__ == "__main__":
    test_profiling()
