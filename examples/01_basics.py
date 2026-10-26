import benostreamdb as bsdb
import pandas as pd
import numpy as np
import os
import shutil
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import warnings

warnings.filterwarnings('ignore')

# Set pandas options to show clean formatted results
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 50)

def run_demo():
    print("=== BenoStreamDB Basics Demo (SciFact Dataset) ===")

    # 1. Setup Table Path
    db_path = "scifact_news_db"
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    # Initialize Hardware Compute Context
    ctx = None
    for target in ["cuda", "intel"]:
        try:
            ctx = bsdb.ComputeContext(target)
            print(f"Using {target.upper()} GPU compute context")
            break
        except Exception:
            pass
    if ctx is None:
        print("Using CPU compute context")

    # 2. Load Real SciFact Dataset from Hugging Face
    print("\nLoading SciFact scientific literature dataset from Hugging Face...")
    try:
        corpus = load_dataset("mteb/scifact", "corpus", split="corpus[:200]")
    except Exception:
        corpus = load_dataset("BeIR/scifact", "corpus", split="corpus[:200]")
    
    df = pd.DataFrame(corpus)
    # Ensure standard schema: doc_id, title, text
    df = df.rename(columns={"_id": "doc_id"})
    print(f"Loaded {len(df)} peer-reviewed scientific papers.")

    # 3. Embedding Generation
    print("Embedding scientific abstracts using sentence-transformers/all-MiniLM-L6-v2...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(df["text"].tolist())
    df["embedding"] = [list(e) for e in embeddings]

    # 4. Ingest into BenoStreamDB Table
    print("\nIngesting into BenoStreamDB...")
    table = bsdb.Table(db_path, device=ctx)
    
    # Overlay indexing: HNSW vector index + Inverted index on text/title
    table.add_index_columns(["embedding", "doc_id", "title"])
    table.write_pandas(df)
    table.commit()
    print(f"Ingestion complete: {len(df)} documents indexed.")

    # 5. Scalar Filtering (Inverted Index)
    print("\n--- 1. Scalar Filtering (Fast RoaringBitmap Inverted Index) ---")
    # Filter documents by keyword in title
    cancer_docs = table.filter("title LIKE '%cancer%' OR title LIKE '%cells%'").to_pandas()
    print(f"Found {len(cancer_docs)} cellular / oncology papers:")
    if not cancer_docs.empty:
        print(cancer_docs[["doc_id", "title"]].head())

    # 6. Vector Search (HNSW Index)
    print("\n--- 2. Semantic Vector Search (HNSW) ---")
    query_text = "Microstructural development of human cerebral cortex and white matter"
    query_embedding = list(model.encode(query_text))

    print(f"Query: '{query_text}'")
    results = table.filter(vector_filter=query_embedding, k=5).to_pandas()
    
    dist_col = "distance" if "distance" in results.columns else "_distance"
    cols = ["doc_id", "title", dist_col]
    print(results[cols].to_string(index=False))

    # 7. Hybrid Search (Scalar Filter + HNSW Vector Search)
    print("\n--- 3. Hybrid Search (Title Filter + Vector Search) ---")
    hybrid_results = table.to_pandas(
        filter="title LIKE '%human%' OR title LIKE '%cells%'",
        vector_filter=query_embedding,
        k=3
    )
    if not hybrid_results.empty:
        dist_col = "distance" if "distance" in hybrid_results.columns else "_distance"
        print(hybrid_results[["doc_id", "title", dist_col]].to_string(index=False))
    else:
        print("No results found for hybrid criteria.")

    # Clean up
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    print("\nDemo completed successfully.")

if __name__ == "__main__":
    run_demo()
