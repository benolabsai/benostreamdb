import hyperstreamdb as pyhyperstream
import pandas as pd
import shutil
import os
import glob
import time

TABLE_URI = "file:///tmp/hyperstream_test_selective_indexing"

def setup_module():
    if os.path.exists("/tmp/hyperstream_test_selective_indexing"):
        shutil.rmtree("/tmp/hyperstream_test_selective_indexing")

def count_index_files(table_uri, extension):
    """Count files with specific extension in table directory."""
    path = table_uri.replace("file://", "")
    return len(glob.glob(f"{path}/*.{extension}"))

def get_files(table_uri):
    path = table_uri.replace("file://", "")
    return glob.glob(f"{path}/*")

def test_selective_indexing():
    print("1. Creating Table and Ingesting Data (Default: NO INDEX)")
    table = pyhyperstream.PyTable(TABLE_URI)
    
    df = pd.DataFrame({
        "id": range(100),
        "tag": ["A"] * 50 + ["B"] * 50,
        "value": range(100)
    })
    
    table.write_pandas(df)
    
    # Verify NO indexes exist
    idx_count = count_index_files(TABLE_URI, "idx")
    inv_count = count_index_files(TABLE_URI, "inv.parquet")
    print(f"Default Ingest: Found {idx_count} .idx files, {inv_count} .inv.parquet files")
    
    assert idx_count == 0, "Default should not create .idx files"
    assert inv_count == 0, "Default should not create .inv.parquet files"
    
    print("\n2. Enabling Index for 'tag' column (Testing Add + Backfill)")
    table.add_index_columns(["tag"])
    
    # Should have backfilled the existing segment for 'tag'
    # 'tag' is String -> Inverted Index (.inv.parquet)
    # Note: My implementation currently doesn't produce .idx for String, only .inv.parquet
    
    inv_count = count_index_files(TABLE_URI, "inv.parquet")
    print(f"After Add Index('tag'): Found {inv_count} .inv.parquet files")
    assert inv_count >= 1, "Should have backfilled index for 'tag'"
    
    # Write NEW data
    df2 = pd.DataFrame({
        "id": range(100, 200),
        "tag": ["C"] * 100,
        "value": range(100, 200)
    })
    table.write_pandas(df2)
    
    inv_count_2 = count_index_files(TABLE_URI, "inv.parquet")
    print(f"After Second Write: Found {inv_count_2} .inv.parquet files")
    assert inv_count_2 == inv_count + 1, "Should have indexed new segment for 'tag'"

    print("\n3. Removing Index for 'tag'")
    table.remove_index_columns(["tag"])
    
    # Write THIRD batch
    df3 = pd.DataFrame({
         "id": range(200, 300),
         "tag": ["D"] * 100,
         "value": range(200, 300)
    })
    table.write_pandas(df3)
    
    inv_count_3 = count_index_files(TABLE_URI, "inv.parquet")
    print(f"After Remove Index: Found {inv_count_3} .inv.parquet files")
    assert inv_count_3 == inv_count_2, "Should NOT have indexed new segment"
    
    print("\n4. Index ALL Columns (Backfill Everything)")
    # Columns: id (Int64), tag (String), value (Int64)
    # Int64 creates .idx AND .inv.parquet (in my impl)
    # String creates .inv.parquet
    
    table.index_all_columns()
    
    # We have 3 segments. All should be indexed.
    # Expected:
    # 3 segments * (id_inv + id_idx + value_inv + value_idx + tag_inv)
    # Wait, 'id' is Int64? yes (range in py produces int64 usually)
    
    idx_count_final = count_index_files(TABLE_URI, "idx")
    inv_count_final = count_index_files(TABLE_URI, "inv.parquet")
    
    print(f"After Index All: Found {idx_count_final} .idx and {inv_count_final} .inv files")
    
    assert idx_count_final > 0, "Should have backfilled scalar indexes"
    assert inv_count_final > inv_count_3, "Should have backfilled more inverted indexes"
    
    print("\n5. Checking Configuration")
    cols = table.get_index_columns()
    print(f"Configured columns: {cols}")
    # Note: index_all_columns() sets the 'index_all' flag. get_index_columns() returns the manual list.
    # Manual list might still contain 'tag' depending on implementation details of remove_index_columns 
    # (my impl used retain, so 'tag' matches and is removed).
    
    table.remove_all_index_columns()
    cols = table.get_index_columns()
    assert len(cols) == 0, "Should be empty after remove_all"
    
    print("\nSUCCESS: All selective indexing tests passed!")

if __name__ == "__main__":
    setup_module()
    test_selective_indexing()
