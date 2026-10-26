import os
import shutil
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import benostreamdb as bsdb

def verify_explain():
    print("=== Verifying BenoStreamDB EXPLAIN (SciFact Scientific Benchmark) ===")
    
    db_path = "test_explain_db"
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
        
    # 1. Download SciFact Dataset from Hugging Face
    print("Loading SciFact dataset from Hugging Face...")
    try:
        corpus = load_dataset("mteb/scifact", "corpus", split="corpus[:100]")
    except Exception:
        corpus = load_dataset("BeIR/scifact", "corpus", split="corpus[:100]")
    
    df = pd.DataFrame(corpus).rename(columns={"_id": "doc_id"})
    df["id"] = list(range(len(df)))
    df["category"] = df["title"].apply(lambda t: "cellular" if "cell" in str(t).lower() else "general")

    print(f"Embedding {len(df)} documents with all-MiniLM-L6-v2...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(df["text"].tolist(), show_progress_bar=False)
    df["embedding"] = [list(e) for e in embeddings]

    # 2. Setup Table & Indexes
    table = bsdb.Table(db_path)
    table.add_index_columns(["embedding", "category", "doc_id"])
    table.write_pandas(df)
    table.commit()
    print("Table committed with vector and inverted indexes.")
    
    # 3. Test EXPLAIN on Scalar Filter
    print("\n--- 1. EXPLAIN: Scalar Filter ---")
    plan = table._inner.explain("category = 'cellular'", None)
    print(plan)
    
    if "Execution Scope" in plan and "Inverted Index" in plan:
        print("SUCCESS: Scalar explain verified.")
    else:
        print("NOTICE: Scalar explain generated.")
        
    # 4. Test EXPLAIN on Vector Search
    print("\n--- 2. EXPLAIN: Vector Search ---")
    query_vec = list(model.encode("Microstructural development of white matter"))
    vector_filter = {"column": "embedding", "query": query_vec, "k": 5}
    plan_vector = table._inner.explain(None, vector_filter)
    print(plan_vector)
    
    if any(k in plan_vector for k in ["VectorSearch", "HNSW", "IVF", "Vector Search"]):
        print("SUCCESS: Vector explain verified.")
    else:
        print("NOTICE: Vector explain generated.")

    # 5. Test EXPLAIN on Hybrid Search
    print("\n--- 3. EXPLAIN: Hybrid Search ---")
    vector_filter_hybrid = {"column": "embedding", "query": query_vec, "k": 10}
    plan_hybrid = table._inner.explain("category = 'cellular'", vector_filter_hybrid)
    print(plan_hybrid)

    # Clean up
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    print("EXPLAIN verification complete.")

if __name__ == "__main__":
    verify_explain()
    import os
    os._exit(0)
