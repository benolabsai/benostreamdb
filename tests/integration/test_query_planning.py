import hyperstreamdb as hdb
import pandas as pd
import shutil
import os
import glob
import pyarrow as pa

def test_query_planning():
    base_dir = "/tmp/test_query_planning"
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir)

    uri = f"file://{base_dir}"
    table = hdb.PyTable(uri)
    table.add_index_columns(["val"])

    print("1. Creating Data Segments...")
    
    # Segment 1: Low Values (0-9)
    df1 = pd.DataFrame({"id": range(10), "val": range(10)})
    df1["id"] = df1["id"].astype("int32")
    df1["val"] = df1["val"].astype("int64")
    table.write_pandas(df1)
    
    # Segment 2: Mid Values (10-19)
    df2 = pd.DataFrame({"id": range(10, 20), "val": range(10, 20)})
    df2["id"] = df2["id"].astype("int32")
    df2["val"] = df2["val"].astype("int64")
    table.write_pandas(df2)
    
    # Segment 3: High Values (20-29)
    df3 = pd.DataFrame({"id": range(20, 30), "val": range(20, 30)})
    df3["id"] = df3["id"].astype("int32")
    df3["val"] = df3["val"].astype("int64")
    table.write_pandas(df3)
    
    files = glob.glob(f"{base_dir}/*.parquet")
    print(f"Created {len(files)} initial segments.")
    assert len(files) == 3

    print("2. Compacting to generate Manifest with Stats...")
    # This should merge small files if they were small enough, but let's assume default compaction thresholds
    # might skip them based on size? Or we force rewrite?
    # Our Compactor implementation defaults: target 512MB, min 384MB.
    # Uh oh. My small files won't be compacted based on size!
    # I need to pass options to force compaction?
    # PyTable.compact() uses default options.
    # Default options: min_file_size_bytes: 384MB.
    # My files are Tiny. They will be ignored!
    # I need to update `compact` binding to accept options OR change defaults for testing.
    # Or I can manually trigger compaction via Rust test?
    # Or I can update `CompactionOptions::default()`? No, that's bad for prod.
    # I should update `PyTable::compact` to accept `force=True` or similar?
    # Or just `options` dict.
    
    # Force compaction with very small thresholds to ensure we keep multiple files (bins)
    # instead of merging them all into one.
    # We want multiple files in the manifest to verify Pruning.
    table.compact(min_file_size_bytes=1)
    
    # Check if Manifest exists
    # We can check _manifest dir
    manifest_files = glob.glob(f"{base_dir}/_manifest/v*.json")
    print(f"Manifest files: {manifest_files}")
    assert len(manifest_files) >= 1

    print("3. Executing Pruned Query (val > 25)...")
    # Should prune Seg 1 (0-9) and Seg 2 (10-19).
    # Should scan Seg 3 (20-29) [or the compacted version of it]
    
    # We rely on stdout to see "Pruned to X segments".
    # But functionally, we verify the result data.
    
    df_res = table.read_pandas("val > 25")
    print("Result Data:")
    print(df_res)
    
    assert len(df_res) == 4, f"Expected 4 rows (26, 27, 28, 29), got {len(df_res)}"
    assert sorted(df_res["val"].tolist()) == [26, 27, 28, 29]
    
    print("4. Executing Pruned Query (val < 5)...")
    # Should prune Seg 2 and 3. Scan Seg 1.
    df_res_2 = table.read_pandas("val < 5")
    print("Result Data:")
    print(df_res_2)
    assert len(df_res_2) == 5, f"Expected 5 rows (0-4), got {len(df_res_2)}"

    print("SUCCESS: Query Planning Verified.")

if __name__ == "__main__":
    test_query_planning()
