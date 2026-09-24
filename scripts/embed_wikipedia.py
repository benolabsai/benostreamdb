import argparse
import pandas as pd
from sentence_transformers import SentenceTransformer
import pyarrow as pa
import pyarrow.parquet as pq

def main():
    parser = argparse.ArgumentParser(description="Embed Wikipedia summaries")
    parser.add_argument("--input", type=str, required=True, help="Input nodes.parquet")
    parser.add_argument("--output", type=str, required=True, help="Output nodes.parquet with embeddings")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model")
    
    args = parser.parse_args()
    
    print(f"Loading {args.input}...")
    df = pd.read_parquet(args.input)
    
    print(f"Loaded {len(df)} nodes. Initializing model {args.model}...")
    model = SentenceTransformer(args.model)
    
    print("Computing embeddings...")
    # Compute embeddings (returns a numpy array)
    embeddings = model.encode(df['summary'].tolist(), show_progress_bar=True)
    
    # We want to store embeddings as a list of floats so parquet handles it as list<float>
    print("Formatting embeddings for parquet...")
    df['embedding'] = embeddings.tolist()
    
    print(f"Saving to {args.output}...")
    table = pa.Table.from_pandas(df)
    pq.write_table(table, args.output)
    
    print("Done!")

if __name__ == "__main__":
    main()
