import sys
import os
import polars as pl
import pyarrow.parquet as pq

# Ensure benostreamdb is in path
sys.path.insert(0, os.path.abspath("python"))
import benostreamdb as bsdb

def main():
    candidates = [
        os.path.expanduser("~/data/benostreamdb/three_tier/section_embeddings_parts/part_0000.parquet"),
        "data/three_tier/section_embeddings_parts/part_0000.parquet",
    ]
    part_path = next((p for p in candidates if os.path.exists(p)), None)
    if not part_path:
        print("Error: part_0000.parquet does not exist.")
        return

    table_uri = "file://" + os.path.abspath("data/sample_db")
    print(f"Creating sample database at {table_uri}")
    
    # Check if we've already created it, so we can re-run quickly
    if os.path.exists("data/sample_db"):
        embt = bsdb.Table(table_uri)
    else:
        import pyarrow as pa
        schema = pa.schema([
            pa.field("section_id", pa.int64()),
            pa.field("embedding", pa.list_(pa.float32()))
        ])
        embt = bsdb.Table.create(table_uri, schema)
        print(f"Loading {part_path}...")
        pq_f = pq.ParquetFile(part_path)
        done = 0
        # Load up to 50,000 rows to keep it fast
        for batch in pq_f.iter_batches(batch_size=10000):
            df = pl.from_arrow(batch)
            embt.write(df)
            done += len(df)
            print(f"Loaded {done} rows...")
            if done >= 50000:
                break
                
        print("Adding HNSW index...")
        embt.add_index("embedding", algorithm="hnsw")
        print("Waiting for background tasks to finish...")
        embt.wait_for_background_tasks()
        print("Table ready.")

    print("Extracting a query vector from the data...")
    # Grab the first vector to use as a query
    first_row = pl.read_parquet(part_path, n_rows=1)
    section_id = first_row["section_id"][0]
    first_vec = first_row["embedding"][0].to_list()
    
    print(f"Querying for section_id {section_id}...")
    res = embt.read(vector_query=("embedding", first_vec, 5))
    
    print("Top 5 results:")
    print(res)

if __name__ == "__main__":
    main()
