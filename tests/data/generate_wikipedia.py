#!/usr/bin/env python3
"""
Generate synthetic Wikipedia-like dataset with embeddings for testing hybrid queries.

This simulates a document corpus with:
- Title, category, text metadata (for scalar filtering)
- Embeddings (for vector search)
- Combined hybrid queries: WHERE category='science' AND embedding ~ query_vector
"""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import random
import string

# Embedding dimensions (same as vector search test)
DEFAULT_EMBEDDING_DIM = 768

# Wikipedia-like categories
CATEGORIES = [
    'science', 'technology', 'history', 'geography', 'arts',
    'sports', 'politics', 'business', 'health', 'entertainment'
]

# Simulated word pools for generating realistic-ish titles
ADJECTIVES = ['Modern', 'Ancient', 'Global', 'Digital', 'Natural', 'Urban', 'Classical', 'New']
NOUNS = ['History', 'Systems', 'Theory', 'Analysis', 'Development', 'Research', 'Studies', 'Methods']
TOPICS = ['Economics', 'Physics', 'Biology', 'Computing', 'Politics', 'Culture', 'Medicine', 'Engineering']

def generate_title():
    """Generate a Wikipedia-like article title"""
    pattern = random.choice([
        lambda: f"{random.choice(ADJECTIVES)} {random.choice(TOPICS)}",
        lambda: f"{random.choice(TOPICS)} {random.choice(NOUNS)}",
        lambda: f"The {random.choice(ADJECTIVES)} {random.choice(NOUNS)} of {random.choice(TOPICS)}",
        lambda: f"{random.choice(TOPICS)}: A {random.choice(NOUNS)}",
    ])
    return pattern()

def generate_wikipedia(num_docs=100_000, dim=DEFAULT_EMBEDDING_DIM, batch_size=10_000):
    """Generate synthetic Wikipedia dataset with embeddings"""
    
    output_dir = Path("tests/data/wikipedia")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_docs:,} Wikipedia-like documents with {dim}D embeddings")
    print(f"Batch size: {batch_size:,}")
    
    schema = pa.schema([
        ('doc_id', pa.int64()),
        ('title', pa.string()),
        ('category', pa.string()),
        ('word_count', pa.int32()),
        ('view_count', pa.int64()),
        ('is_featured', pa.bool_()),
        ('embedding', pa.list_(pa.float32(), dim)),
    ])
    
    for batch_idx in range(0, num_docs, batch_size):
        batch_end = min(batch_idx + batch_size, num_docs)
        batch_count = batch_end - batch_idx
        
        # Generate metadata
        doc_ids = list(range(batch_idx, batch_end))
        titles = [generate_title() for _ in range(batch_count)]
        categories = np.random.choice(CATEGORIES, batch_count).tolist()
        word_counts = np.random.randint(100, 50000, batch_count).astype(np.int32)
        view_counts = np.random.exponential(10000, batch_count).astype(np.int64)
        is_featured = (np.random.random(batch_count) < 0.05).tolist()  # 5% featured
        
        # Generate embeddings (normalized)
        # Cluster embeddings by category for realistic search behavior
        embeddings = np.random.randn(batch_count, dim).astype(np.float32)
        
        # Add category-specific bias to make similar categories cluster
        for i, cat in enumerate(categories):
            cat_idx = CATEGORIES.index(cat)
            embeddings[i, cat_idx * 50:(cat_idx + 1) * 50] += 2.0  # Boost category dimensions
        
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Create batch
        batch = pa.RecordBatch.from_arrays([
            pa.array(doc_ids, type=pa.int64()),
            pa.array(titles, type=pa.string()),
            pa.array(categories, type=pa.string()),
            pa.array(word_counts.tolist(), type=pa.int32()),
            pa.array(view_counts.tolist(), type=pa.int64()),
            pa.array(is_featured, type=pa.bool_()),
            pa.array(embeddings.tolist(), type=pa.list_(pa.float32(), dim)),
        ], schema=schema)
        
        # Write to Parquet
        output_file = output_dir / f"wikipedia_{batch_idx:09d}.parquet"
        
        with pq.ParquetWriter(output_file, schema) as writer:
            writer.write_batch(batch)
        
        print(f"Written batch {batch_idx:,} - {batch_end:,} to {output_file.name}")
    
    print(f"\nGeneration complete!")
    print(f"Total files: {len(list(output_dir.glob('*.parquet')))}")
    print(f"Total size: {sum(f.stat().st_size for f in output_dir.glob('*.parquet')) / 1e6:.2f} MB")

if __name__ == "__main__":
    generate_wikipedia()
