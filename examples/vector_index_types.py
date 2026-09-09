"""
Example: Choosing Vector Index Types and Quantization in HyperStreamDB

HyperStreamDB provides native vector index strategies in the free community core:
1. HNSW (Uncompressed) - Exact float32 vectors, highest precision.
2. HNSW + TurboQuant 8-bit (hnsw_tq8) - 4x compression via Fast Walsh-Hadamard Transform (FWHT), >99% recall retention.
3. HNSW + TurboQuant 4-bit (hnsw_tq4) - 8x compression for massive datasets, maximum memory efficiency.
4. HNSW + Product Quantization (hnsw_pq) - Traditional sub-space vector quantization.
"""

import numpy as np
import pandas as pd
import hyperstreamdb as hdb

# Prepare sample 128-dimensional embedding data
np.random.seed(42)
df = pd.DataFrame({
    'id': [1, 2, 3, 4, 5],
    'text': [f"Document {i}" for i in range(1, 6)],
    'embedding': [np.random.rand(128).astype(np.float32).tolist() for _ in range(5)]
})

# -------------------------------------------------------------
# Example 1: Standard Uncompressed HNSW
# -------------------------------------------------------------
print("=" * 60)
print("Example 1: Uncompressed HNSW Index")
table_hnsw = hdb.Table("file:///tmp/test_hnsw")
table_hnsw.write_pandas(df)
table_hnsw.add_index("embedding", "hnsw")
table_hnsw.commit()
print("✓ Written with standard uncompressed HNSW index")

# -------------------------------------------------------------
# Example 2: TurboQuant 8-bit (TQ8) — Recommended Production Default
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("Example 2: TurboQuant 8-bit (4x Compression, >99% Recall)")
table_tq8 = hdb.Table("file:///tmp/test_tq8")
table_tq8.write_pandas(df)
# Add HNSW index with TQ8 quantization
table_tq8.add_index("embedding", "hnsw_tq8")
table_tq8.commit()

query_vec = np.random.rand(128).astype(np.float32).tolist()
results = table_tq8.vector_search("embedding", query_vec, k=3)
print(f"✓ Written and searched with HNSW-TQ8 (returned {len(results)} results)")

# -------------------------------------------------------------
# Example 3: TurboQuant 4-bit (TQ4) — Maximum Compression
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("Example 3: TurboQuant 4-bit (8x Compression)")
table_tq4 = hdb.Table("file:///tmp/test_tq4")
table_tq4.write_pandas(df)
table_tq4.add_index("embedding", "hnsw_tq4")
table_tq4.commit()

results_tq4 = table_tq4.vector_search("embedding", query_vec, k=3)
print(f"✓ Written and searched with HNSW-TQ4 (returned {len(results_tq4)} results)")

# -------------------------------------------------------------
# Example 4: Explicit quantize() API
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("Example 4: Explicit table.quantize() Configuration")
table_custom = hdb.Table("file:///tmp/test_custom_tq")
table_custom.write_pandas(df)
table_custom.quantize(
    column="embedding",
    type_="TQ8",
    metric="l2",
    complexity=32,  # M connections
    quality=200     # ef_construction
)
table_custom.commit()
print("✓ Configured custom quantization parameters successfully")

print("\n" + "=" * 60)
print("Summary of Quantization Options:")
print("- 'hnsw': Float32 baseline (1x compression, 100% recall)")
print("- 'hnsw_tq8': TurboQuant 8-bit (4x compression, >99% recall)")
print("- 'hnsw_tq4': TurboQuant 4-bit (8x compression, ultra-compact)")
print("- 'hnsw_pq': Product Quantization (configurable subspaces)")
print("=" * 60)
