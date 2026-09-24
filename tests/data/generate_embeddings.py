#!/usr/bin/env python3
"""
Generate synthetic vector embeddings for testing
Creates 10M 768-dimensional vectors (simulating BERT embeddings)
"""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

# Default embedding dimension (384 = 384-dimensional vectors, 768 = BERT-base, 1536 = OpenAI ada-002, 3072 = OpenAI text-embedding-3-large)
DEFAULT_EMBEDDING_DIM = 768

def generate_embeddings(num_vectors=10_000_000, dim=DEFAULT_EMBEDDING_DIM, batch_size=100_000):
    """Generate synthetic embeddings in batches"""
    
    output_dir = Path("tests/data/embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_vectors:,} vectors of dimension {dim}")
    print(f"Batch size: {batch_size:,}")
    
    schema = pa.schema([
        ('id', pa.int64()),
        ('embedding', pa.list_(pa.float32(), dim)),
        ('category', pa.string()),
    ])
    
    categories = ['science', 'technology', 'sports', 'politics', 'entertainment']
    
    for batch_idx in range(0, num_vectors, batch_size):
        batch_end = min(batch_idx + batch_size, num_vectors)
        batch_count = batch_end - batch_idx
        
        # Generate random embeddings (normalized)
        embeddings = np.random.randn(batch_count, dim).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Create batch
        batch = pa.RecordBatch.from_arrays([
            pa.array(range(batch_idx, batch_end), type=pa.int64()),
            pa.array(embeddings.tolist(), type=pa.list_(pa.float32(), dim)),
            pa.array(np.random.choice(categories, batch_count)),
        ], schema=schema)
        
        # Write to Parquet
        output_file = output_dir / f"embeddings_{batch_idx:09d}.parquet"
        
        with pq.ParquetWriter(output_file, schema) as writer:
            writer.write_batch(batch)
        
        print(f"Written batch {batch_idx:,} - {batch_end:,} to {output_file.name}")
    
    print(f"\nGeneration complete!")
    print(f"Total files: {len(list(output_dir.glob('*.parquet')))}")
    print(f"Total size: {sum(f.stat().st_size for f in output_dir.glob('*.parquet')) / 1e9:.2f} GB")

if __name__ == "__main__":
    generate_embeddings()
