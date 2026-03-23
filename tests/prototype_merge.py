
import time
import random
import uuid
from typing import Dict, List, Set

# Simulation Parameters
NUM_ROWS = 1_000_000
NUM_SEGMENTS = 1000
ROWS_PER_SEGMENT = NUM_ROWS // NUM_SEGMENTS
NUM_UPDATES = 10  # Number of rows to Upsert

print(f"--- HyperStreamDB Merge Prototype ---")
print(f"Dataset: {NUM_ROWS:,} rows across {NUM_SEGMENTS} segments")
print(f"Updates: {NUM_UPDATES} keys")

# 1. Generate Data
print("\n[1] Generating Data...")
data = []
# Simulate an Inverted Index (Value -> SegmentID -> [RowIDs])
# Using a simplified Dict[Value, Set[SegmentID]] for pruning simulation
index: Dict[str, Set[int]] = {} 

all_keys = []

for seg_id in range(NUM_SEGMENTS):
    segment_data = []
    for row_id in range(ROWS_PER_SEGMENT):
        key = f"user_{uuid.uuid4()}"[:12]
        val = random.randint(1, 1000)
        segment_data.append((key, val))
        all_keys.append(key)
        
        # Build Index
        if key not in index:
            index[key] = set()
        index[key].add(seg_id)
        
    data.append(segment_data)

# Pick random keys to update
update_keys = random.sample(all_keys, NUM_UPDATES)
# Add some new keys (Inserts)
new_keys = [f"new_user_{i}" for i in range(10)]
merge_keys = update_keys + new_keys

print(f"Data generated. Index size: {len(index):,} entries.")

# 2. Strategy A: Standard Scan (Iceberg/Delta without specific partition pruning)
# Worst case: Scan all segments to find matches
print("\n[2] Benchmarking Standard Scan...")
start_time = time.time()
scanned_count = 0
found_matches = 0

for seg_id, segment in enumerate(data):
    # Simulate reading the segment
    # In a real system, this is deserialization cost + processing
    for row in segment:
        scanned_count += 1
        if row[0] in merge_keys:
            found_matches += 1

scan_time = time.time() - start_time
print(f"Scan Time: {scan_time:.4f} sec")
print(f"Rows Scanned: {scanned_count:,}")
print(f"Matches Found: {found_matches}")


# 3. Strategy B: Index-Accelerated (HyperStream)
print("\n[3] Benchmarking Index Pruning...")
start_time = time.time()
pruned_segments = set()
segments_checked = 0

# Step 1: Index Pruning (Identify Candidate Segments)
# In reality, this is looking up RoaringBitmaps from S3 (fast random reads)
for key in merge_keys:
    if key in index:
        candidates = index[key]
        pruned_segments.update(candidates)

# Step 2: Read only candidate segments
# (We only 'scan' the segments that were flagged by the index)
index_scanned_count = 0
index_found_matches = 0

for seg_id in pruned_segments:
    segments_checked += 1
    segment = data[seg_id]
    for row in segment:
        index_scanned_count += 1
        if row[0] in merge_keys:
            index_found_matches += 1

# Note: In a real implementation using RoaringBitmaps, we wouldn't even 'scan' the candidate segment rows.
# We would jump directly to the row ID. But for this simulation, "Segment Pruning" is the main metric.

index_time = time.time() - start_time
print(f"Index Time: {index_time:.4f} sec")
print(f"Segments Touched: {len(pruned_segments)} / {NUM_SEGMENTS}")
print(f"Rows Scanned (Effective): {index_scanned_count:,}")
print(f"Matches Found: {index_found_matches}")

# 4. Results
speedup = scan_time / index_time if index_time > 0 else 0
print(f"\n--- Results ---")
print(f"Speedup: {speedup:.2f}x")
if index_found_matches != found_matches:
    print("WARNING: Mismatch in matches found!")
else:
    print("Verification: Correctness confirmed.")

