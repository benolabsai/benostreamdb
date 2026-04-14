import hyperstreamdb as hs
import numpy as np
import pandas as pd
import shutil
import os

def test_hybrid_search():
    print("\n--- Testing Hybrid RRF Search (Vector + BM25) ---")
    
    db_dir = os.path.abspath("./test_hybrid_db")
    db_path = f"file://{db_dir}"
    if os.path.exists(db_dir):
        shutil.rmtree(db_dir)
        
    # 1. Setup Table and Indexes
    table = hs.Table(db_path)
    # Add indexes for both vector and keyword search
    table.add_index_columns(["embedding"])
    
    # 2. Prepare Data
    # Row 1: Strong Keyword ("fox"), Weak Vector (all zeros)
    # Row 2: Weak Keyword ("dog"), Strong Vector (all ones)
    # Row 3: Strong Keyword ("fox"), Strong Vector (all ones)
    
    data = {
        "id": [1, 2, 3],
        "content": [
            "The quick brown fox jumps over the lazy dog",
            "A very energetic canine",
            "A fast fox that also has a strong vector"
        ],
        "embedding": [
            [0.0] * 128,  # Weak vs query [1,1,1...]
            [1.0] * 128,  # Strong vs query [1,1,1...]
            [0.9] * 128   # Strong vs query [1,1,1...]
        ]
    }
    df = pd.DataFrame(data)
    
    # 3. Write Data
    table.write(df)
    table.commit() # Ensure data is committed to manifest before adding index
    
    # Explicitly add BM25 index on 'content'
    table.add_index("content", "bm25")
    
    table.flush()
    table.wait_for_background_tasks() # Wait for backfill to finish
    
    # 4. Perform Hybrid Search
    # Query for "fox" (Keyword) + Vector matching [1,1,1...]
    print("\nExecuting Hybrid Search...")
    query_vector = [1.0] * 128
    
    # In HyperStreamDB Python API, hybrid search is triggered by providing
    # a SQL-style filter string to the 'search' method.
    # The Smart Trigger detects the column name and dispatches to BM25.
    results_df = table.search(
        column="embedding",
        query=query_vector,
        k=10,
        filter="content = 'fox'"
    )
    
    # results_df is a Pandas DataFrame (returned by search -> to_pandas)
    
    print(f"Retrieved {len(results_df)} rows.")
    if len(results_df) > 0:
        print("Results (Ranked by RRF):")
        print(results_df[["id", "content"]])
        
        # Expected behavior:
        # Row 3 should be #1 (Strong Keyword + Strong Vector)
        # Row 1 and 2 will compete for #2/#3
        
        ids = results_df["id"].tolist()
        assert 3 in ids, "Row 3 must be in results"
        if ids[0] == 3:
            print("SUCCESS: Row 3 is correctly ranked #1!")
        else:
            print(f"DEBUG: Top result was {ids[0]}")
            
    # 5. Cleanup
    shutil.rmtree(db_dir)

if __name__ == "__main__":
    try:
        test_hybrid_search()
    except Exception as e:
        print(f"Test Failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
