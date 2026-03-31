
import hyperstreamdb as hdb
import pandas as pd
import numpy as np
import os

def test_explain():
    db_path = "test_explain_db"
    if os.path.exists(db_path):
        import shutil
        shutil.rmtree(db_path)
    
    # Create table
    table = hdb.Table(db_path, explain=True)
    
    # Write some data
    df = pd.DataFrame({
        "id": range(100),
        "val": [f"val_{i}" for i in range(100)],
        "emb": [list(np.random.randn(128).astype(np.float32)) for _ in range(100)]
    })
    table.write(df)
    table.commit()
    
    print("\n--- EXPLAIN: Simple Filter ---")
    results = table.to_pandas(filter="id > 50")
    print(f"Results: {len(results)} rows")
    
    print("\n--- EXPLAIN: Vector Search ---")
    q_emb = list(np.random.randn(128).astype(np.float32))
    results = table.to_pandas(vector_filter={"column": "emb", "query": q_emb, "k": 5})
    print(f"Results: {len(results)} rows")

    print("\n--- EXPLAIN: Hybrid ---")
    results = table.to_pandas(
        filter="id < 20",
        vector_filter={"column": "emb", "query": q_emb, "k": 5}
    )
    print(f"Results: {len(results)} rows")

if __name__ == "__main__":
    test_explain()
