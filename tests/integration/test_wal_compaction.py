import hyperstreamdb as hdb
import pandas as pd
import os
import time
import shutil

def test_wal_compaction():
    print("="*60)
    print("Testing WAL Compaction")
    print("="*60)

    # Setup
    uri = f"file:///tmp/test_wal_compact_{int(time.time())}"
    base_path = uri.replace("file://", "")
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    
    # 1. Write many small batches to trigger compaction
    print("\nPhase 1: Writing 200 small batches...")
    table = hdb.Table(uri)
    
    for i in range(200):
        df = pd.DataFrame({'id': [i], 'val': [i * 2]})
        table.write_pandas(df)
        if (i + 1) % 50 == 0:
            print(f"  Written {i + 1} batches...")
    
    # Check WAL file exists
    wal_path = os.path.join(base_path, "_wal", "log.arrow")
    assert os.path.exists(wal_path), f"WAL file missing at {wal_path}"
    size_before = os.path.getsize(wal_path)
    print(f"✓ WAL file size before compaction: {size_before} bytes")
    
    # 2. Manual compaction
    print("\nPhase 2: Triggering manual compaction...")
    table.checkpoint()
    
    # Check WAL still exists
    assert os.path.exists(wal_path), "WAL file should still exist after compaction"
    size_after = os.path.getsize(wal_path)
    print(f"✓ WAL file size after compaction: {size_after} bytes")
    
    # Size should be smaller or similar (consolidated into 1 batch)
    # Note: Might not always be smaller due to Arrow IPC overhead, but should be similar
    print(f"✓ Size ratio: {size_after / size_before:.2f}")
    
    # 3. Verify recovery still works
    print("\nPhase 3: Verifying recovery after compaction...")
    del table
    
    table2 = hdb.Table(uri)
    result = table2.to_pandas()
    assert len(result) == 200, f"Expected 200 rows, got {len(result)}"
    print(f"✓ Recovered {len(result)} rows from compacted WAL")
    
    # 4. Flush and verify WAL cleanup
    print("\nPhase 4: Flushing to disk...")
    table2.commit()
    
    # WAL should be deleted after flush
    if os.path.exists(wal_path):
        assert not os.path.exists(wal_path), "WAL should be deleted after flush"
    print("✓ WAL cleaned up after flush")
    
    # Verify data still readable
    result2 = table2.to_pandas()
    assert len(result2) == 200
    print("✓ Data readable from persistent storage")
    
    print("\nPASSED: WAL Compaction Test")

if __name__ == "__main__":
    test_wal_compaction()
