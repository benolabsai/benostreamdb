import benostreamdb as bsdb
import pandas as pd
import numpy as np
import os
import shutil
import warnings
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from openai import OpenAI

warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', 90)
pd.set_option('display.width', 1000)

def run_rag_demo():
    print("=== BenoStreamDB RAG Pipeline Demo (SciFact Dataset) ===")
    
    db_path = "scifact_rag_db"
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    # 1. Hardware compute context
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

    # 2. Embedding Model
    print("Loading embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # 3. Load SciFact Knowledge Base from Hugging Face
    print("\nLoading SciFact corpus from Hugging Face...")
    try:
        corpus = load_dataset("mteb/scifact", "corpus", split="corpus[:250]")
    except Exception:
        corpus = load_dataset("BeIR/scifact", "corpus", split="corpus[:250]")
        
    df = pd.DataFrame(corpus).rename(columns={"_id": "doc_id"})
    print(f"Generating embeddings for {len(df)} scientific papers...")
    embeddings = model.encode(df["text"].tolist(), show_progress_bar=False)
    df["embedding"] = [list(e) for e in embeddings]

    # 4. Ingest into BenoStreamDB
    print(f"Ingesting {len(df)} entries into BenoStreamDB...")
    table = bsdb.Table(db_path, explain=True, device=ctx)
    table.add_index_columns(["embedding", "doc_id", "title"])
    table.write_pandas(df)
    table.commit()
    print("Ingestion complete.")

    # 5. OpenCode Go Client Setup
    api_key = os.environ.get("OPENCODE_API_KEY")
    client = OpenAI(
        base_url="https://opencode.ai/zen/go/v1",
        api_key=api_key or "dummy",
        default_headers={"x-opencode-session": "benolabs-howto"}
    )

    def retrieve(query: str, k: int = 3, max_distance: float = 1.3):
        """Retrieve most relevant scientific papers from BenoStreamDB."""
        query_vec = list(model.encode(query))
        res = table.filter(vector_filter=query_vec, k=k).to_pandas()
        dist_col = "distance" if "distance" in res.columns else "_distance"
        if max_distance is not None and not res.empty:
            res = res[res[dist_col] <= max_distance]
        return res

    def generate_answer(claim: str, retrieved_df: pd.DataFrame) -> str:
        """Synthesize answer citing retrieved papers using OpenCode Go (deepseek-v4.1-flash)."""
        if retrieved_df.empty:
            return "No sufficiently relevant literature found in the database."

        dist_col = "distance" if "distance" in retrieved_df.columns else "_distance"
        context_blocks = []
        for _, row in retrieved_df.iterrows():
            context_blocks.append(
                f"[Doc ID: {row['doc_id']}] Title: {row['title']}\n"
                f"Abstract: {row['text'][:400]}... (Distance: {row[dist_col]:.4f})"
            )
        combined_context = "\n\n".join(context_blocks)

        prompt = f"""You are an expert scientific research assistant. Evaluate the following scientific claim based ONLY on the provided peer-reviewed literature context.
Cite specific document IDs in your answer.

Claim: {claim}

Retrieved Literature:
{combined_context}

Provide a concise, factual evaluation in 2-3 sentences."""

        if api_key:
            try:
                response = client.chat.completions.create(
                    model="deepseek-v4.1-flash",
                    messages=[
                        {"role": "system", "content": "You are an objective scientific evaluation assistant. Answer concisely and cite document IDs."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=250,
                    temperature=0.2,
                    extra_body={"thinking": {"type": "disabled"}}
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"[Warning: OpenCode Go call failed ({e}); falling back to local citation]")

        # Fallback if API key missing or network failure
        top = retrieved_df.iloc[0]
        return f"[Synthesized from Doc {top['doc_id']}: {top['title']}] Evidence indicates: {top['text'][:200]}..."

    # 6. Execute Scientific Claims Evaluation
    test_claims = [
        "Microstructural development of human cerebral cortex and white matter can be measured in vivo.",
        "Cerebral organoids model human brain development and microcephaly pathogenesis."
    ]

    for claim in test_claims:
        print("\n" + "=" * 70)
        print(f"Scientific Claim: \"{claim}\"")
        matches = retrieve(claim, k=3)
        dist_col = "distance" if "distance" in matches.columns else "_distance"
        print(f"Retrieved {len(matches)} papers (top match dist: {matches[dist_col].iloc[0]:.4f} - Doc {matches['doc_id'].iloc[0]})")
        
        answer = generate_answer(claim, matches)
        print(f"\nSynthesized Evaluation:\n{answer}")

    # 7. Cleanup
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    run_rag_demo()
