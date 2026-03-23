"""
Example: Choosing Vector Index Type (HNSW vs IVF-PQ)

HyperStreamDB supports two vector index types:
1. HNSW (default) - Best for <10M vectors, high recall
2. IVF-PQ - Best for >100M vectors, memory-efficient (not yet fully implemented)
"""

import hyperstreamdb as hdb
import pandas as pd
import numpy as np

# Example 1: Using HNSW (default)
print("Example 1: HNSW Index (default)")
table_hnsw = hdb.Table("file:///tmp/test_hnsw")

df = pd.DataFrame({
    'id': [1, 2, 3],
    'embedding': [
        np.random.rand(128).tolist(),
        np.random.rand(128).tolist(),
        np.random.rand(128).tolist()
    ]
})

table_hnsw.write_pandas(df)
print("✓ Written with HNSW index (default)")

# Example 2: Explicitly choosing HNSW
print("\nExample 2: Explicitly choosing HNSW")
table_hnsw_explicit = hdb.Table(
    "file:///tmp/test_hnsw_explicit",
    vector_index_type=hdb.VectorIndexType.Hnsw
)
table_hnsw_explicit.write_pandas(df)
print("✓ Written with HNSW index (explicit)")

# Example 3: Choosing IVF-PQ (falls back to HNSW for now)
print("\nExample 3: IVF-PQ (not yet implemented, falls back to HNSW)")
table_ivf = hdb.Table(
    "file:///tmp/test_ivf",
    vector_index_type=hdb.VectorIndexType.IvfPq
)
table_ivf.write_pandas(df)
print("✓ Written with IVF-PQ (currently falls back to HNSW)")

print("\n" + "="*60)
print("Summary:")
print("- HNSW: Best for <10M vectors, high recall")
print("- IVF-PQ: Best for >100M vectors, memory-efficient")
print("- Default: HNSW")
print("="*60)
