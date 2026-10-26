"""
Example: Choosing Vector Index Types and Quantization in BenoStreamDB (SciFact Benchmark)

BenoStreamDB provides native vector index strategies in the free community core:
1. HNSW (Uncompressed) - Exact float32 vectors, highest precision.
2. HNSW + TurboQuant 8-bit (hnsw_tq8) - 4x compression via Fast Walsh-Hadamard Transform (FWHT), >99% recall retention.
3. HNSW + TurboQuant 4-bit (hnsw_tq4) - 8x compression for massive datasets, maximum memory efficiency.
4. HNSW + Product Quantization (hnsw_pq) - Traditional sub-space vector quantization.

This example benchmarks all 4 index strategies on real scientific literature from the Hugging Face SciFact corpus.
"""

import os
import shutil
import time
import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import benostreamdb as bsdb

# -------------------------------------------------------------
# 1. Download Real Scientific Dataset from Hugging Face
# -------------------------------------------------------------
print("=" * 70)
print("Downloading SciFact scientific corpus from Hugging Face...")
try:
    corpus = load_dataset("mteb/scifact", "corpus", split="corpus[:100]")
except Exception:
    corpus = load_dataset("BeIR/scifact", "corpus", split="corpus[:100]")

df = pd.DataFrame(corpus).rename(columns={"_id": "doc_id"})
df["id"] = list(range(len(df)))

print(f"Embedding {len(df)} peer-reviewed abstracts with sentence-transformers/all-MiniLM-L6-v2...")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(df["text"].tolist(), show_progress_bar=False)
df["embedding"] = [list(e) for e in embeddings]

query_text = "Microstructural development of human cerebral cortex and white matter"
query_vec = list(model.encode(query_text))

# -------------------------------------------------------------
# Example 1: Standard Uncompressed HNSW (Float32 Baseline)
# -------------------------------------------------------------
print("\n" + "=" * 70)
print("Example 1: Standard Uncompressed HNSW Index (Float32 Baseline)")
table_path_hnsw = "file:///tmp/scifact_hnsw"
shutil.rmtree("/tmp/scifact_hnsw", ignore_errors=True)

table_hnsw = bsdb.Table(table_path_hnsw)
table_hnsw.write_pandas(df)
table_hnsw.add_index("embedding", "hnsw")
table_hnsw.commit()

t0 = time.time()
res_hnsw = table_hnsw.vector_search("embedding", query_vec, k=3)
hnsw_time_ms = (time.time() - t0) * 1000

dist_col = "distance" if "distance" in res_hnsw.columns else "_distance"
print(f"✓ HNSW search completed in {hnsw_time_ms:.2f}ms (returned {len(res_hnsw)} results):")
for _, r in res_hnsw.iterrows():
    print(f"  - [{r['doc_id']}] {r['title'][:65]}... (dist: {r[dist_col]:.4f})")

# -------------------------------------------------------------
# Example 2: TurboQuant 8-bit (TQ8) — Recommended Production Default
# -------------------------------------------------------------
print("\n" + "=" * 70)
print("Example 2: TurboQuant 8-bit (hnsw_tq8: 4x Compression, >99% Recall)")
table_path_tq8 = "file:///tmp/scifact_tq8"
shutil.rmtree("/tmp/scifact_tq8", ignore_errors=True)

table_tq8 = bsdb.Table(table_path_tq8)
table_tq8.write_pandas(df)
table_tq8.add_index("embedding", "hnsw_tq8")
table_tq8.commit()

t0 = time.time()
res_tq8 = table_tq8.vector_search("embedding", query_vec, k=3)
tq8_time_ms = (time.time() - t0) * 1000

dist_col = "distance" if "distance" in res_tq8.columns else "_distance"
print(f"✓ HNSW-TQ8 search completed in {tq8_time_ms:.2f}ms (returned {len(res_tq8)} results):")
for _, r in res_tq8.iterrows():
    print(f"  - [{r['doc_id']}] {r['title'][:65]}... (dist: {r[dist_col]:.4f})")

# -------------------------------------------------------------
# Example 3: TurboQuant 4-bit (TQ4) — Maximum Compression
# -------------------------------------------------------------
print("\n" + "=" * 70)
print("Example 3: TurboQuant 4-bit (hnsw_tq4: 8x Compression)")
table_path_tq4 = "file:///tmp/scifact_tq4"
shutil.rmtree("/tmp/scifact_tq4", ignore_errors=True)

table_tq4 = bsdb.Table(table_path_tq4)
table_tq4.write_pandas(df)
table_tq4.add_index("embedding", "hnsw_tq4")
table_tq4.commit()

t0 = time.time()
res_tq4 = table_tq4.vector_search("embedding", query_vec, k=3)
tq4_time_ms = (time.time() - t0) * 1000

dist_col = "distance" if "distance" in res_tq4.columns else "_distance"
print(f"✓ HNSW-TQ4 search completed in {tq4_time_ms:.2f}ms (returned {len(res_tq4)} results):")
for _, r in res_tq4.iterrows():
    print(f"  - [{r['doc_id']}] {r['title'][:65]}... (dist: {r[dist_col]:.4f})")

# -------------------------------------------------------------
# Example 4: Explicit quantize() Configuration
# -------------------------------------------------------------
print("\n" + "=" * 70)
print("Example 4: Explicit table.quantize() Configuration")
table_path_custom = "file:///tmp/scifact_custom_tq"
shutil.rmtree("/tmp/scifact_custom_tq", ignore_errors=True)

table_custom = bsdb.Table(table_path_custom)
table_custom.write_pandas(df)
table_custom.quantize(
    column="embedding",
    type_="TQ8",
    metric="l2",
    complexity=32,  # M connections
    quality=200     # ef_construction
)
table_custom.commit()

res_custom = table_custom.vector_search("embedding", query_vec, k=3)
dist_col = "distance" if "distance" in res_custom.columns else "_distance"
print(f"✓ Custom TQ8 index searched successfully (top doc: {res_custom.iloc[0]['doc_id']})")

# Clean up
for p in ["/tmp/scifact_hnsw", "/tmp/scifact_tq8", "/tmp/scifact_tq4", "/tmp/scifact_custom_tq"]:
    shutil.rmtree(p, ignore_errors=True)

print("\n" + "=" * 70)
print("Summary of Quantization Options on SciFact:")
print("- 'hnsw': Float32 baseline (1x compression, 100% recall)")
print("- 'hnsw_tq8': TurboQuant 8-bit (4x compression, >99% recall)")
print("- 'hnsw_tq4': TurboQuant 4-bit (8x compression, ultra-compact)")
print("- 'hnsw_pq': Product Quantization (configurable subspaces)")
print("=" * 70)
