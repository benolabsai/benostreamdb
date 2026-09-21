import hyperstreamdb as hdb
import pandas as pd
import numpy as np

# Set pandas options to show full text in results
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 50)
import os
import shutil
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def run_demo():
    print("=== HyperStreamDB Basics Demo ===")

    # 1. Setup
    db_path = "news_db"
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    # Initialize the Intel GPU context (or CPU if not available)
    # Note: If "intel" fails, fallback to CPU
    try:
        ctx = hdb.ComputeContext("intel")
        print("Using Intel GPU context")
    except Exception as e:
        print(f"Intel GPU not available ({e}), using CPU")
        ctx = None

    # 2. Load Data
    print("\nLoading AG News dataset...")
    dataset = load_dataset("ag_news", split="test[:500]")
    df = pd.DataFrame(dataset)
    label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
    df["category"] = df["label"].map(label_map)

    # 3. Embedding
    print(f"Embedding {len(df)} articles using all-MiniLM-L6-v2...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(df["text"].tolist())
    df["embedding"] = [list(e) for e in embeddings]

    # 4. Ingest into HyperStreamDB
    print("\nIngesting into HyperStreamDB...")
    table = hdb.Table(db_path, device=ctx)
    
    # Enable Vector Indexing for the embedding column and Inverted Index for category
    # Note: Use add_index_columns for multiple columns
    table.add_index_columns(["embedding", "category", "text"])
    
    table.write(df)
    table.commit()
    print("Ingestion complete.")

    # 5. Scalar Filtering
    print("\n--- Scalar Filtering ---")
    # Using the fluent API: table.filter(expr).to_pandas()
    # Or table.to_pandas(filter=expr)
    sports_query = table.filter("category = 'Sports'")
    sports_df = sports_query.to_pandas()
    print(f"Found {len(sports_df)} sports articles.")
    if len(sports_df) > 0:
        print(sports_df[["category", "text"]].head())

    # 6. Vector Search
    print("\n--- Vector Search ---")
    query_text = "Winning medals in international sports competitions"
    query_embedding = list(model.encode(query_text))

    # Using the fluent API: table.filter(vector_filter=...).to_pandas()
    # Or table.to_pandas(vector_filter=...)
    # The current API seems to prefer vector_filter as a dict or list in to_pandas
    
    print(f"Searching for: '{query_text}'")
    results = table.to_pandas(vector_filter=query_embedding, k=5)
    
    if "_distance" in results.columns:
        print(results[["category", "text", "_distance"]])
    elif "distance" in results.columns:
        print(results[["category", "text", "distance"]])
    else:
        print(results[["category", "text"]])

    # 7. Hybrid Search
    print("\n--- Hybrid Search (Sci/Tech + Vector) ---")
    query_text = "What is the latest news in Artificial Intelligence?"
    query_embedding = list(model.encode(query_text))
    # Use Fluent API (as requested by user)
    hybrid_results = (table.filter("category = 'Sci/Tech'")
                           .vector_search(query_embedding, k=5)
                           .to_pandas())
    if not hybrid_results.empty:
        cols = ["category", "text"]
        if "_distance" in hybrid_results.columns: cols.append("_distance")
        elif "distance" in hybrid_results.columns: cols.append("distance")
        print(hybrid_results[cols].to_string())
    else:
        print("No results found for hybrid search.")

if __name__ == "__main__":
    run_demo()
