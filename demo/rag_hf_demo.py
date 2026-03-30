import os
import getpass
import pandas as pd
import numpy as np
import hyperstreamdb as hdb
from datasets import load_dataset
from huggingface_hub import InferenceClient

# RAG Demo: HyperStreamDB + Hugging Face
# ------------------------------------

# 1. Setup API Token
if "HF_TOKEN" not in os.environ:
    os.environ["HF_TOKEN"] = getpass.getpass("Enter your Hugging Face API Token: ")

# 2. Initialize Hugging Face Inference Client
# We use this for both generating embeddings and asking questions
client = InferenceClient(token=os.environ["HF_TOKEN"])

print("\n--- HYPERSTREAMDB RAG DEMO ---")

# 3. Define Inference Functions
def get_embeddings(texts):
    """Generate embeddings using a hosted model"""
    # Using 'all-MiniLM-L6-v2' (fast, versatile)
    return client.feature_extraction(
        texts,
        model="sentence-transformers/all-MiniLM-L6-v2"
    )

def get_answer(question, contexts):
    """Generate an answer using a hosted LLM (Mistral)"""
    context_str = "\n".join([f"- {c}" for c in contexts])
    prompt = f"Answer the question accurately using ONLY the provided context.\n\nContext:\n{context_str}\n\nQuestion: {question}\nAnswer:"
    
    return client.text_generation(
        prompt,
        model="mistralai/Mistral-7B-Instruct-v0.2",
        max_new_tokens=250,
        temperature=0.7
    )

# 4. Ingest Data (SQuAD Knowledge Base)
dataset = load_dataset("squad", split="train")
unique_contexts = pd.DataFrame(dataset)["context"].unique()[:300]
df = pd.DataFrame({
    "id": range(len(unique_contexts)),
    "context": unique_contexts,
    "title": [dataset[i]["title"] for i in range(len(unique_contexts))]
})

print(f"\nIngesting {len(df)} knowledge base articles...")
embeddings = get_embeddings(df["context"].tolist())
df["embedding"] = [np.array(e).astype(np.float32) for e in embeddings]

# Setup HyperStreamDB Table
table_uri = "./rag_hf_db"
table = hdb.Table(table_uri)
table.add_index_columns(["embedding"])  # Enable semantic graph indexing
table.write_pandas(df)
table.commit()  # Finalize the initial manifest

print(f"Index built! Knowledge base ready at {table_uri}")

# 5. Retrieval + Generation
def run_query(question):
    print(f"\nUser Question: {question}")
    
    # Retrieval
    q_emb = np.array(get_embeddings([question])[0]).astype(np.float32)
    # HyperStreamDB search handles the vector indexing + scalar filtering automatically
    results = table.to_pandas(vector_filter={"column": "embedding", "query": q_emb, "k": 3})
    
    contexts = results["context"].tolist()
    print(f"Retrieved {len(contexts)} contexts from HyperStreamDB.")
    
    # Generation
    print("Generating answer via Mistral...")
    answer = get_answer(question, contexts)
    print(f"\nFINAL ANSWER:\n{answer}")
    print("-" * 30)

# Final Demo Query
run_query("Why did Notre Dame's main building burn down in 1879?")
