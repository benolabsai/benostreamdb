
import hyperstreamdb as hdb
import pandas as pd
import shutil
import os
import glob

def test_merge_pruning():
    base_dir = "/tmp/test_merge"
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir)
    
    uri = f"file://{base_dir}"
    table = hdb.Table(uri)
    table.add_index_columns(["id", "val"])
    
    # 1. Write Segment A (IDs 0-9)
    df_a = pd.DataFrame({
        "id": range(0, 10),
        "val": range(100, 110)
    })
    # We must explicitly cast to int32 because our Rust writer expects Int32 for the index implementation
    df_a["id"] = df_a["id"].astype("int32")
    table.write_pandas(df_a)
    print("Written Segment A")

    # 2. Write Segment B (IDs 10-19)
    df_b = pd.DataFrame({
        "id": range(10, 20),
        "val": range(200, 210)
    })
    df_b["id"] = df_b["id"].astype("int32")
    table.write_pandas(df_b)
    print("Written Segment B")

    # 3. Write Segment C (IDs 20-29)
    df_c = pd.DataFrame({
        "id": range(20, 30),
        "val": range(300, 310)
    })
    df_c["id"] = df_c["id"].astype("int32")
    table.write_pandas(df_c)
    print("Written Segment C")
    
    # Verify files exist
    # Verify files exist
    files = glob.glob(f"{base_dir}/*.inv.parquet")
    print(f"Index files found: {len(files)}")
    assert len(files) >= 3, "Should have at least 3 inverted index files"

    # 4. Prune is implicit in merge_pandas now. Let's run full merge.
    # Updates:
    # ID 5 (A) -> Val 999
    # ID 25 (C) -> Val 888
    # ID 105 (New) -> Val 777
    updates_df = pd.DataFrame({
        "id": [5, 25, 105],
        "val": [999, 888, 777]
    })
    updates_df["id"] = updates_df["id"].astype("int32")
    
    print("Running Merge...")
    table.merge_pandas(updates_df, "id")
    print("Merge Complete.")
    
    # 5. Verification
    
    # A. Check Files
    # Should have new segments replacing A and C
    # Segment B should be untouched
    # New segment for 105
    files = glob.glob(f"{base_dir}/*.parquet")
    print(f"Total Parquet files: {len(files)}")
    
    # B. Read Data
    # Used pandas to verify content
    # hdb.Table.to_pandas()
    result_df = table.to_pandas(None)
    print("Result Data:")
    print(result_df.sort_values("id"))
    
    # Check Row Count: 30 original + 1 new = 31?
    # No, 2 updates (replace) + 1 insert = 30 - 2 + 2 + 1 = 31. Correct.
    assert len(result_df) == 31, f"Expected 31 rows, got {len(result_df)}"
    
    # Check Updates
    row_5 = result_df[result_df["id"] == 5].iloc[0]
    assert row_5["val"] == 999, f"Row 5 should have val 999, got {row_5['val']}"
    
    row_25 = result_df[result_df["id"] == 25].iloc[0]
    assert row_25["val"] == 888, f"Row 25 should have val 888, got {row_25['val']}"
    
    row_105 = result_df[result_df["id"] == 105].iloc[0]
    assert row_105["val"] == 777, f"Row 105 should have val 777, got {row_105['val']}"
    
    print("SUCCESS: Merge Execution Verified.")

if __name__ == "__main__":
    test_merge_pruning()
