import os
import shutil
import pandas as pd
import numpy as np
import requests
import benostreamdb as bsdb
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# RAG Demo: BenoStreamDB + Hugging Face SciFact Benchmark
# -------------------------------------------------------

print("=== BENOSTREAMDB RAG DEMO (SciFact Benchmark) ===")

# 1. Setup Models & Embeddings
local_embedder = SentenceTransformer("all-MiniLM-L6-v2")

def get_embeddings(texts):
    return local_embedder.encode(texts, show_progress_bar=False)

def get_answer(claim, contexts):
    context_str = "\n".join([f"- {c}" for c in contexts])
    base_url = os.environ.get("OPENCODE_BASE_URL", "https://opencode.ai/zen/go/v1")
    api_key = os.environ.get("OPENCODE_API_KEY", "")
    
    if api_key:
        try:
            client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                default_headers={"x-opencode-session": "benolabs-howto"}
            )
            prompt = f"Evaluate this scientific claim based ONLY on the provided context.\n\nContext:\n{context_str}\n\nClaim: {claim}"
            res = client.chat.completions.create(
                model="deepseek-v4.1-flash",
                messages=[
                    {"role": "system", "content": "You are a concise scientific assistant. Answer in 2 sentences with factual evidence."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                extra_body={"thinking": {"type": "disabled"}}
            )
            return res.choices[0].message.content.strip()
        except Exception as e:
            print(f"[Notice] OpenCode Go call failed ({e}); falling back to local citation.")
            
    return contexts[0][:250] + "..."

# 2. Ingest Data (SciFact Knowledge Base from Hugging Face)
print("Loading SciFact dataset from Hugging Face...")
try:
    corpus = load_dataset("mteb/scifact", "corpus", split="corpus[:150]")
except Exception:
    corpus = load_dataset("BeIR/scifact", "corpus", split="corpus[:150]")

df = pd.DataFrame(corpus).rename(columns={"_id": "doc_id"})
df["id"] = list(range(len(df)))

print(f"Generating embeddings for {len(df)} peer-reviewed abstracts...")
embeddings = get_embeddings(df["text"].tolist())
df["embedding"] = [list(e) for e in embeddings]

# 3. Setup BenoStreamDB Table
table_uri = "./scifact_hf_db"
shutil.rmtree(table_uri, ignore_errors=True)
table = bsdb.Table(table_uri)
table.add_index_columns(["embedding", "doc_id", "title"])
table.write_pandas(df)
table.commit()

print(f"Index built! Knowledge base ready at {table_uri}")

# 4. Retrieval + Generation
def run_query(claim):
    print(f"\nScientific Claim: {claim}")
    
    # Retrieval
    q_emb = list(get_embeddings([claim])[0])
    results = table.filter(vector_filter=q_emb, k=3).to_pandas()
    
    contexts = results["text"].tolist()
    print(f"Retrieved {len(contexts)} contexts from BenoStreamDB.")
    
    # Generation
    print("Generating answer via OpenCode Go (deepseek-v4.1-flash)...")
    answer = get_answer(claim, contexts)
    print(f"\nEVALUATION:\n{answer}")
    print("-" * 50)

# 5. Run Demo Queries
run_query("Microstructural development of human cerebral cortex and white matter can be measured in vivo.")
run_query("Cerebral organoids model human brain development and microcephaly.")

# 6. Clean Up
shutil.rmtree(table_uri, ignore_errors=True)
print("\nDemo completed cleanly.")
