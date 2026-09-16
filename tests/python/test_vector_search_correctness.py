import os
import shutil
import pytest
import pyarrow as pa
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import hyperstreamdb as hdb

@pytest.fixture
def test_data():
    """Generates synthetic embedding data for exact nearest neighbor testing."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 128
    
    # Generate random L2-normalized embeddings
    embeddings = np.random.randn(n_samples, n_features).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    
    # Query vector
    query = np.random.randn(n_features).astype(np.float32)
    query = query / np.linalg.norm(query)
    
    df = pd.DataFrame({
        "id": range(n_samples),
        "embedding": embeddings.tolist()
    })
    
    return df, query

def test_recall_vs_sklearn_l2(test_data, tmpdir):
    df, query = test_data
    uri = str(tmpdir.join("test_l2"))
    
    schema = pa.schema([
        ("id", pa.int64()),
        ("embedding", pa.list_(pa.float32(), 128))
    ])
    
    table = hdb.Table.create(uri, schema)
    table.write(df)
    table.commit()
    
    # 1. Exact k-NN via scikit-learn
    k = 10
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="brute")
    X = np.vstack(df["embedding"].values)
    nn.fit(X)
    
    # sklearn returns distances (euclidean), HyperStreamDB returns distances (L2 squared for Euclidean sometimes, let's check)
    sk_distances, sk_indices = nn.kneighbors([query])
    sk_indices = sk_indices[0].tolist()
    # sklearn euclidean is sqrt(sum((x-y)^2))
    # Let's check what hyperstreamdb returns. Usually vector search returns exact L2 squared distance.
    # L2 distance squared
    sk_sq_distances = (sk_distances[0] ** 2).tolist()
    
    # 2. HyperStreamDB vector search
    # Assuming Table.search() or Table.vector_search() exists in python API.
    # From __init__.py we know Table.search() exists: `table.search(query).top_k(10).to_pandas()`
    # Let's assume the syntax is `table.search(query, vector_column_name="embedding").top_k(k)`? Or maybe it auto-detects.
    # Let's use `table.search(query).top_k(k).to_pandas()` based on ROADMAP
    res_df = table.search("embedding", query.tolist(), k=k)
    
    assert "id" in res_df.columns
    assert "distance" in res_df.columns or "score" in res_df.columns
    
    dist_col = "distance" if "distance" in res_df.columns else "score"
    hdb_indices = res_df["id"].tolist()
    hdb_distances = res_df[dist_col].tolist()
    
    # 3. Calculate Recall
    intersection = set(sk_indices).intersection(set(hdb_indices))
    recall = len(intersection) / k
    
    # For a dataset of 1000, exact search (brute force) in HDB should yield 1.0 recall
    assert recall == 1.0, f"Recall was {recall}, expected 1.0"
    
    # 4. Check distance metric correctness
    # HyperStreamDB might return L2 squared distances. Let's compare to sk_sq_distances
    # We will use np.isclose to compare
    for sk_dist, hdb_dist in zip(sk_sq_distances, hdb_distances):
        # Allow small floating point differences
        assert np.isclose(sk_dist, hdb_dist, rtol=1e-5, atol=1e-5) or np.isclose(np.sqrt(sk_dist), hdb_dist, rtol=1e-5, atol=1e-5), \
            f"Distance mismatch: sklearn L2_sq={sk_dist}, L2={np.sqrt(sk_dist)}, hdb={hdb_dist}"

def test_recall_vs_sklearn_cosine(test_data, tmpdir):
    df, query = test_data
    uri = str(tmpdir.join("test_cosine"))
    
    schema = pa.schema([
        ("id", pa.int64()),
        ("embedding", pa.list_(pa.float32(), 128))
    ])
    
    table = hdb.Table.create(uri, schema)
    # In hyperstreamdb, we can pass metric directly to search
    table.write(df)
    table.commit()
    
    # 1. Exact k-NN via scikit-learn
    k = 10
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
    X = np.vstack(df["embedding"].values)
    nn.fit(X)
    
    sk_distances, sk_indices = nn.kneighbors([query])
    sk_indices = sk_indices[0].tolist()
    
    # 2. HyperStreamDB vector search
    res_df = table.search("embedding", query.tolist(), k=k, metric="cosine")
    
    assert "id" in res_df.columns
    dist_col = "distance" if "distance" in res_df.columns else "score"
    
    hdb_indices = res_df["id"].tolist()
    hdb_distances = res_df[dist_col].tolist()
    
    # 3. Calculate Recall
    intersection = set(sk_indices).intersection(set(hdb_indices))
    recall = len(intersection) / k
    
    assert recall == 1.0, f"Recall was {recall}, expected 1.0"
    
    # 4. Distance metric correctness
    # sklearn cosine distance is 1 - cosine_similarity. HyperStreamDB might return the same, or just cosine similarity.
    # We check if it's 1 - cos_sim or cos_sim.
    for sk_dist, hdb_dist in zip(sk_distances[0], hdb_distances):
        sk_sim = 1.0 - sk_dist
        assert np.isclose(sk_dist, hdb_dist, rtol=1e-4, atol=1e-4) or np.isclose(sk_sim, hdb_dist, rtol=1e-4, atol=1e-4), \
            f"Distance mismatch: sklearn dist={sk_dist}, sim={sk_sim}, hdb={hdb_dist}"
