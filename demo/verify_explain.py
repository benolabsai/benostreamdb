import hyperstreamdb as hdb
import pandas as pd
import numpy as np
import os
import shutil

def verify_explain():
    print("=== Verifying HyperStreamDB EXPLAIN ===")
    
    db_path = "test_explain_db"
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
        
    # 1. Setup Table
    table = hdb.Table(db_path)
    table.add_index_columns(["embedding", "category"])
    
    # 2. Insert Data
    data = []
    for i in range(100):
        data.append({
            "id": i,
            "category": "A" if i < 50 else "B",
            "embedding": [float(i)] * 128
        })
    df = pd.DataFrame(data)
    table.write(df)
    table.commit()
    
    # 3. Test EXPLAIN on Scalar Filter
    print("\n--- EXPLAIN: Scalar Filter ---")
    plan = table.filter("category = 'A'").explain()
    print(plan)
    
    # Check for expected keywords
    if "Execution Scope" in plan and "Inverted Index" in plan:
        print("SUCCESS: Scalar explain looks reasonable.")
    else:
        print("WARNING: Scalar explain might be missing details.")
        
    # 4. Test EXPLAIN on Vector Search
    print("\n--- EXPLAIN: Vector Search ---")
    query_vec = [1.0] * 128
    plan_vector = table.vector_search(query_vec, k=5).explain()
    print(plan_vector)
    
    if "Vector Search" in plan_vector or "KNN" in plan_vector or "HNSW" in plan_vector or "IVF" in plan_vector:
        print("SUCCESS: Vector explain looks reasonable.")
    else:
        # Check if it mentions Sequential Scan if index not built yet (wait, we did commit)
        print("NOTICE: Vector explain summary shown.")

    # 5. Test EXPLAIN on Hybrid Search
    print("\n--- EXPLAIN: Hybrid Search ---")
    plan_hybrid = table.filter("category = 'B'").vector_search(query_vec, k=10).explain()
    print(plan_hybrid)

if __name__ == "__main__":
    verify_explain()
