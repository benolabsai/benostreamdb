import hyperstreamdb as hdb
import pandas as pd
import numpy as np
import os
import shutil
import requests
import time

# Set pandas options to show full text in results
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 50)
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import warnings

# Suppress warnings and chatty loggers
warnings.filterwarnings('ignore')
try:
    from transformers import logging as tf_logging
    tf_logging.set_verbosity_error()
except ImportError:
    pass

def run_rag_demo():
    print("=== HyperStreamDB RAG Pipeline Demo ===")
    
    db_path = "rag_db"
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    # Initialize context
    # Standard CPU context provides the most stable IO path for the HNSW-IVF reader
    ctx = None
    print("Using stable CPU compute context")

    # 1. Setup Models
    # Using all-MiniLM-L6-v2 for 100% stability.
    print("Loading embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    model.max_seq_length = 512

    # 2. Ingest Knowledge Base (SQuAD)
    print("\nLoading and embedding SQuAD contexts...")
    dataset_full = load_dataset("squad", split="validation")
    full_df = pd.DataFrame(dataset_full)
    unique_contexts_df = full_df.drop_duplicates(subset=["context"]).copy()
    
    print(f"Generating embeddings for {len(unique_contexts_df)} unique contexts...")
    # This model generates 384D vectors
    embeddings = model.encode(unique_contexts_df["context"].tolist(), show_progress_bar=True)
    unique_contexts_df["embedding"] = [list(e) for e in embeddings]

    print(f"Ingesting {len(unique_contexts_df)} entries into HyperStreamDB...")
    table = hdb.Table(db_path, explain=True, context=ctx)
    table.add_index_columns(["embedding", "context"])
    table.write(unique_contexts_df[["context", "title", "embedding"]])
    print("Committing table (flushes data and builds vector indexes)...")
    table.commit()
    print("Ingestion complete.")

    # 3. Retrieval
    def retrieve(question, k=3):
        print(f"Question: '{question}'")
        q_emb = model.encode(question)
        # Using a title filter to show the full "Postgres EXPLAIN" style query plan
        results = table.to_pandas(
            vector_filter={"column": "embedding", "query": q_emb, "k": k},
            filter="title != 'Unknown'"
        )
        if results.empty: return []
        print(f"   Found: {len(results)} contexts")
        return results["context"].tolist()

    # 4. Generation
    def generate_answer(question, contexts):
        if not contexts: return "No results."
        
        prompt = f"Using context below, answer the question precisely: {question}\n\nContexts:\n" + "\n".join([f"- {c}" for c in contexts])
        
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key and os.path.exists("groq_api_key.txt"):
            with open("groq_api_key.txt", "r") as f:
                api_key = f.read().strip()
                
        if not api_key: return "[GROQ_API_KEY missing]"

        max_retries = 3
        for attempt in range(max_retries):
            try:
                url = "https://api.groq.com/openai/v1/chat/completions"
                headers = {"Authorization": f"Bearer {api_key}"}
                payload = {
                    "model": "qwen/qwen3-32b",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                }
                r = requests.post(url, headers=headers, json=payload, timeout=20)
                r.raise_for_status() # Raise error for bad status codes
                return r.json()["choices"][0]["message"]["content"]
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"   Groq API connection error: {e}. Retrying ({attempt+1}/{max_retries})...")
                    time.sleep(2 * (attempt + 1))
                    continue
                return f"[Error: {e}]"

    # 5. Live Runs
    test_questions = full_df["question"].unique()[:3].tolist()
    for question in test_questions:
        print(f"\n--- RAG RUN: {question} ---")
        contexts = retrieve(question, k=3)
        answer = generate_answer(question, contexts)
        print(f"Final Answer:\n{answer}")

if __name__ == "__main__":
    run_rag_demo()
