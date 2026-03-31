import hyperstreamdb as hdb
import pandas as pd
import numpy as np
import os
import shutil
import requests

# Set pandas options to show full text in results
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 50)
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def run_rag_demo():
    print("=== HyperStreamDB RAG Pipeline Demo ===")

    # 1. Setup
    db_path = "rag_db"
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    # Initialize context
    try:
        ctx = hdb.ComputeContext("intel")
        print("Using Intel GPU context")
    except Exception as e:
        print(f"Intel GPU not available ({e}), using CPU")
        ctx = None

    # Load embedding model
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # 2. Ingest Knowledge Base (SQuAD)
    print("\nLoading and embedding SQuAD contexts...")
    dataset_full = load_dataset("squad", split="validation")
    full_df = pd.DataFrame(dataset_full)
    
    # We take unique contexts to avoid redundant entries in the vector DB
    unique_contexts_df = full_df.drop_duplicates(subset=["context"]).copy()
    
    # Roughly map some titles for metadata
    # unique_contexts_df already has "title", "context" from full_df
    
    # Generate embeddings for the unique contexts
    print(f"Generating embeddings for {len(unique_contexts_df)} unique contexts...")
    embeddings = model.encode(unique_contexts_df["context"].tolist())
    unique_contexts_df["embedding"] = [list(e) for e in embeddings]

    print(f"Ingesting {len(unique_contexts_df)} knowledge entries into HyperStreamDB...")
    table = hdb.Table(db_path, context=ctx)
    table.add_index_columns(["embedding", "context"])
    
    table.write(unique_contexts_df[["context", "title", "embedding"]])
    table.commit()
    print("Ingestion complete.")

    # 3. Retrieval
    def retrieve(question, k=3, max_distance=None):
        """Retrieve relevant contexts with optional distance filtering"""
        print(f"Question: '{question}'")
        q_emb = model.encode(question)
        
        # Searching HyperStreamDB
        # Note: Vector search can be done by providing the query vector directly to to_pandas
        results = table.to_pandas(vector_filter={"column": "embedding", "query": q_emb, "k": k})
        
        if results.empty:
            print("   Found: 0 results")
            return []
            
        # Optional distance filtering (post-retrieval for now)
        # Check column name (some parts of the system might return 'distance' or '_distance')
        dist_col = "distance" if "distance" in results.columns else "_distance" if "_distance" in results.columns else None
        
        if max_distance is not None and dist_col:
            results = results[results[dist_col] <= max_distance]
            
        print(f"   Found: {len(results)} contexts (after filtering)")
        return results["context"].tolist()

    # 4. Ollama Generation (Llama 3 / qwen3.5)
    def generate_answer(question, contexts):
        """Interact with groq if available"""
        if not contexts:
            return "No relevant context found to answer the question."
            
        prompt = f"""Answer the question accurately using only the provided context.

Context:
"""
        for i, ctx in enumerate(contexts):
            prompt += f"{i+1}. {ctx}\n"
        
        prompt += f"\nQuestion: {question}\nAnswer:"
        
        # Try Groq (if key available) or fallback to local Ollama/Mock
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key and os.path.exists("groq_api_key.txt"):
            with open("groq_api_key.txt", "r") as f:
                api_key = f.read().strip()
                
        if api_key:
            try:
                groq_url = "https://api.groq.com/openai/v1/chat/completions"
                headers = {"Authorization": f"Bearer {api_key}"}
                payload = {
                    "model": "qwen/qwen3-32b",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 1.0,
                }
                r = requests.post(groq_url, headers=headers, json=payload, timeout=10)
                data = r.json()
                if "choices" in data:
                    return data["choices"][0]["message"]["content"]
                else:
                    return f"[Groq API Error: {data.get('error', {}).get('message', 'Unknown Error')}]"
            except Exception as e:
                return f"[Groq logic failed ({e}). Returning retrieved contexts...]\n\n" + "\n\n".join(contexts)
        else:
            return "[GROQ_API_KEY not found. Please set it to enable live generation.]\n\nContexts:\n" + "\n\n".join(contexts)

    # 5. Live Demo Runs
    # Use actual questions from the SQuAD dataset slice we loaded
    test_questions = full_df["question"].unique()[:3].tolist()

    for question in test_questions:
        print(f"\n--- RAG RUN: {question} ---")
        contexts = retrieve(question, k=3)
        answer = generate_answer(question, contexts)
        print(f"Final Answer:\n{answer}")

if __name__ == "__main__":
    run_rag_demo()
