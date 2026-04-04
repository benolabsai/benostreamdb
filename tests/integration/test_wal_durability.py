
import hyperstreamdb as hdb
import pandas as pd
import tempfile
import shutil
import os
import time
import glob

def test_wal_durability():
    print("="*60)
    print("Testing Write-Ahead Log (WAL) Durability")
    print("="*60)

    # Setup
    ts = int(time.time())
    uri = f"file:///tmp/test_wal_{ts}"
    base_path = f"/tmp/test_wal_{ts}"
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    
    # 1. Write Data (Unflushed)
    print("\nPhase 1: Writing Data (Buffered)...")
    table = hdb.Table(uri)
    table.autocommit = False
    df = pd.DataFrame({'id': range(10), 'val': range(10)})
    table.write_pandas(df)
    time.sleep(0.2) # Allow WAL worker to finish writing
    
    # Verify it's in buffer (via read)
    res = table.to_pandas()
    assert len(res) == 10
    print("✓ Data written and visible in buffer.")

    # Check WAL file exists
    # URI is file:///tmp/test_wal_...
    # Path logic in Rust: uri.strip_prefix("file://")... join("_wal").join("log.arrow")
    base_path = uri.replace("file://", "")
    wal_files = glob.glob(os.path.join(base_path, "_wal", "log_*.arrow"))
    assert len(wal_files) > 0, f"WAL file missing in {os.path.join(base_path, '_wal')}"
    wal_path = wal_files[0]
    print(f"✓ WAL file verified at {wal_path}")
    wal_size = os.path.getsize(wal_path)
    print(f"✓ WAL size: {wal_size} bytes")

    # 2. Simulate Crash / Restart
    print("\nPhase 2: Simulating Restart (Re-opening Table)...")
    del table # Drop instance (release lock if any, though we use simple file append)
    
    # Re-open
    table2 = hdb.Table(uri)
    
    # 3. Verify Recovery
    print("Phase 3: Verifying Recovery...")
    res2 = table2.to_pandas()
    assert len(res2) == 10, f"Expected 10 rows after recovery, found {len(res2)}"
    print("✓ Data successfully recovered from WAL.")
    
    # Verify Indexing (Optional but good)
    # If we had vector index, search should work too.

    # 4. Flush and Verify WAL Cleanup
    print("\nPhase 4: Flushing to Disk...")
    table2.commit() # Flush
    
    # Check WAL is truncated/deleted
    # Note: Our implementation calls remove_file via truncate
    
    if os.path.exists(wal_path):
         assert not os.path.exists(wal_path), "WAL file should be deleted after flush"
    
    print("✓ WAL file cleaned up.")
    
    # Verify data still readable from Parquet
    res3 = table2.to_pandas()
    assert len(res3) == 10
    print("✓ Data readable from persistent storage.")

    print("\nPASSED: WAL Durability Test")

if __name__ == "__main__":
    test_wal_durability()
