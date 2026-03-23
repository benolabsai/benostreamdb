"""
Property-based tests for Python distance API

Feature: python-vector-api-gpu-acceleration
Tests Properties 1, 2, 3, 5, and 6 from the design document
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings
import hyperstreamdb as hdb


# Feature: python-vector-api-gpu-acceleration, Property 1: Distance Computation Correctness
# **Validates: Requirements 1.2**
@given(
    dim=st.integers(min_value=2, max_value=128),
    seed=st.integers(min_value=0, max_value=2**31-1)
)
@settings(max_examples=100, deadline=None)
def test_l2_distance_correctness(dim, seed):
    """For any two vectors of equal dimension, L2 distance should match mathematical definition"""
    np.random.seed(seed)
    a = np.random.randn(dim).astype(np.float32)
    b = np.random.randn(dim).astype(np.float32)
    
    result = hdb.l2(a, b)
    expected = np.sqrt(np.sum((a - b) ** 2))
    
    assert isinstance(result, float)
    assert np.isclose(result, expected, rtol=1e-5), f"L2 distance mismatch: {result} vs {expected}"


# Feature: python-vector-api-gpu-acceleration, Property 1: Distance Computation Correctness
# **Validates: Requirements 1.2**
@given(
    dim=st.integers(min_value=2, max_value=128),
    seed=st.integers(min_value=0, max_value=2**31-1)
)
@settings(max_examples=100, deadline=None)
def test_cosine_distance_correctness(dim, seed):
    """For any two vectors of equal dimension, cosine distance should match mathematical definition"""
    np.random.seed(seed)
    a = np.random.randn(dim).astype(np.float32)
    b = np.random.randn(dim).astype(np.float32)
    
    result = hdb.cosine(a, b)
    
    # Compute expected cosine distance
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        expected = 1.0  # Distance is 1 when one vector is zero
    else:
        similarity = dot / (norm_a * norm_b)
        expected = 1.0 - similarity
    
    assert isinstance(result, float)
    assert np.isclose(result, expected, rtol=1e-5, atol=1e-6), \
        f"Cosine distance mismatch: {result} vs {expected}"



# Feature: python-vector-api-gpu-acceleration, Property 1: Distance Computation Correctness
# **Validates: Requirements 1.2**
@given(
    dim=st.integers(min_value=2, max_value=128),
    seed=st.integers(min_value=0, max_value=2**31-1)
)
@settings(max_examples=100, deadline=None)
def test_inner_product_correctness(dim, seed):
    """For any two vectors of equal dimension, inner product should match mathematical definition"""
    np.random.seed(seed)
    a = np.random.randn(dim).astype(np.float32)
    b = np.random.randn(dim).astype(np.float32)
    
    result = hdb.inner_product(a, b)
    expected = np.dot(a, b)
    
    assert isinstance(result, float)
    assert np.isclose(result, expected, rtol=1e-4, atol=1e-6), \
        f"Inner product mismatch: {result} vs {expected}"


# Feature: python-vector-api-gpu-acceleration, Property 1: Distance Computation Correctness
# **Validates: Requirements 1.2**
@given(
    dim=st.integers(min_value=2, max_value=128),
    seed=st.integers(min_value=0, max_value=2**31-1)
)
@settings(max_examples=100, deadline=None)
def test_l1_distance_correctness(dim, seed):
    """For any two vectors of equal dimension, L1 distance should match mathematical definition"""
    np.random.seed(seed)
    a = np.random.randn(dim).astype(np.float32)
    b = np.random.randn(dim).astype(np.float32)
    
    result = hdb.l1(a, b)
    expected = np.sum(np.abs(a - b))
    
    assert isinstance(result, float)
    assert np.isclose(result, expected, rtol=1e-5), \
        f"L1 distance mismatch: {result} vs {expected}"


# Feature: python-vector-api-gpu-acceleration, Property 1: Distance Computation Correctness
# **Validates: Requirements 1.2**
@given(
    dim=st.integers(min_value=2, max_value=128),
    seed=st.integers(min_value=0, max_value=2**31-1)
)
@settings(max_examples=100, deadline=None)
def test_hamming_distance_correctness(dim, seed):
    """For any two vectors of equal dimension, Hamming distance should count differing elements"""
    np.random.seed(seed)
    a = np.random.randint(0, 10, size=dim).astype(np.float32)
    b = np.random.randint(0, 10, size=dim).astype(np.float32)
    
    result = hdb.hamming(a, b)
    expected = float(np.sum(a != b))
    
    assert isinstance(result, float)
    assert result == expected, f"Hamming distance mismatch: {result} vs {expected}"


# Feature: python-vector-api-gpu-acceleration, Property 1: Distance Computation Correctness
# **Validates: Requirements 1.2**
@given(
    dim=st.integers(min_value=2, max_value=128),
    seed=st.integers(min_value=0, max_value=2**31-1)
)
@settings(max_examples=100, deadline=None)
def test_jaccard_distance_correctness(dim, seed):
    """For any two vectors of equal dimension, Jaccard distance should match mathematical definition"""
    np.random.seed(seed)
    # Use binary vectors for Jaccard
    a = (np.random.rand(dim) > 0.5).astype(np.float32)
    b = (np.random.rand(dim) > 0.5).astype(np.float32)
    
    result = hdb.jaccard(a, b)
    
    # Compute expected Jaccard distance
    intersection = np.sum((a > 0) & (b > 0) & (a == b))
    union = np.sum((a > 0) | (b > 0))
    
    if union == 0:
        expected = 0.0
    else:
        expected = 1.0 - (intersection / union)
    
    assert isinstance(result, float)
    assert np.isclose(result, expected, rtol=1e-5, atol=1e-6), \
        f"Jaccard distance mismatch: {result} vs {expected}"


# Feature: python-vector-api-gpu-acceleration, Property 2: Input Validation
# **Validates: Requirements 1.3, 1.4, 8.4, 8.5**
@given(
    dim_a=st.integers(min_value=2, max_value=50),
    dim_b=st.integers(min_value=2, max_value=50),
    seed=st.integers(min_value=0, max_value=2**31-1)
)
@settings(max_examples=100, deadline=None)
def test_dimension_mismatch_raises_error(dim_a, dim_b, seed):
    """For any vectors with mismatched dimensions, the system should raise ValueError"""
    if dim_a == dim_b:
        return  # Skip when dimensions match
    
    np.random.seed(seed)
    a = np.random.randn(dim_a).astype(np.float32)
    b = np.random.randn(dim_b).astype(np.float32)
    
    with pytest.raises(ValueError, match="dimension mismatch"):
        hdb.l2(a, b)


# Feature: python-vector-api-gpu-acceleration, Property 2: Input Validation
# **Validates: Requirements 1.3, 1.4, 8.4, 8.5**
@given(
    dim=st.integers(min_value=2, max_value=50),
    nan_idx=st.integers(min_value=0, max_value=49),
    seed=st.integers(min_value=0, max_value=2**31-1)
)
@settings(max_examples=100, deadline=None)
def test_nan_values_raise_error(dim, nan_idx, seed):
    """For any vector containing NaN, the system should raise ValueError"""
    if nan_idx >= dim:
        nan_idx = dim - 1
    
    np.random.seed(seed)
    a = np.random.randn(dim).astype(np.float32)
    b = np.random.randn(dim).astype(np.float32)
    a[nan_idx] = np.nan
    
    with pytest.raises(ValueError, match="NaN"):
        hdb.l2(a, b)


# Feature: python-vector-api-gpu-acceleration, Property 2: Input Validation
# **Validates: Requirements 1.3, 1.4, 8.4, 8.5**
@given(
    dim=st.integers(min_value=2, max_value=50),
    inf_idx=st.integers(min_value=0, max_value=49),
    seed=st.integers(min_value=0, max_value=2**31-1)
)
@settings(max_examples=100, deadline=None)
def test_inf_values_raise_error(dim, inf_idx, seed):
    """For any vector containing infinite values, the system should raise ValueError"""
    if inf_idx >= dim:
        inf_idx = dim - 1
    
    np.random.seed(seed)
    a = np.random.randn(dim).astype(np.float32)
    b = np.random.randn(dim).astype(np.float32)
    a[inf_idx] = np.inf
    
    with pytest.raises(ValueError, match="infinite"):
        hdb.l2(a, b)


# Feature: python-vector-api-gpu-acceleration, Property 3: Input Type Flexibility
# **Validates: Requirements 1.5**
@given(
    dim=st.integers(min_value=2, max_value=50),
    seed=st.integers(min_value=0, max_value=2**31-1)
)
@settings(max_examples=100, deadline=None)
def test_accepts_numpy_arrays(dim, seed):
    """For any valid vector data as NumPy array, the distance function should accept it"""
    np.random.seed(seed)
    a = np.random.randn(dim).astype(np.float32)
    b = np.random.randn(dim).astype(np.float32)
    
    result = hdb.l2(a, b)
    assert isinstance(result, float)
    assert result >= 0.0


# Feature: python-vector-api-gpu-acceleration, Property 5: Batch Operation Shape Correctness
# **Validates: Requirements 3.1**
@given(
    dim=st.integers(min_value=2, max_value=50),
    n_vectors=st.integers(min_value=1, max_value=20),
    seed=st.integers(min_value=0, max_value=2**31-1)
)
@settings(max_examples=100, deadline=None)
def test_batch_operation_shape_correctness(dim, n_vectors, seed):
    """For any query vector and database matrix with N vectors, batch function should return N distances"""
    np.random.seed(seed)
    query = np.random.randn(dim).astype(np.float32)
    vectors = np.random.randn(n_vectors, dim).astype(np.float32)
    
    distances = hdb.l2_batch(query, vectors)
    
    assert isinstance(distances, np.ndarray)
    assert distances.shape == (n_vectors,), \
        f"Expected shape ({n_vectors},), got {distances.shape}"
    assert distances.dtype == np.float32


# Feature: python-vector-api-gpu-acceleration, Property 6: Batch Operation Metric Support
# **Validates: Requirements 3.5**
@given(
    dim=st.integers(min_value=2, max_value=50),
    n_vectors=st.integers(min_value=1, max_value=10),
    seed=st.integers(min_value=0, max_value=2**31-1)
)
@settings(max_examples=100, deadline=None)
def test_batch_operation_all_metrics(dim, n_vectors, seed):
    """For any of the 6 distance metrics, a corresponding batch function should exist and work correctly"""
    np.random.seed(seed)
    query = np.random.randn(dim).astype(np.float32)
    vectors = np.random.randn(n_vectors, dim).astype(np.float32)
    
    # Test all 6 metrics
    batch_functions = [
        hdb.l2_batch,
        hdb.cosine_batch,
        hdb.inner_product_batch,
        hdb.l1_batch,
        hdb.hamming_batch,
        hdb.jaccard_batch,
    ]
    
    for batch_fn in batch_functions:
        distances = batch_fn(query, vectors)
        assert isinstance(distances, np.ndarray)
        assert distances.shape == (n_vectors,), \
            f"Function {batch_fn.__name__} returned wrong shape: {distances.shape}"


# Feature: python-vector-api-gpu-acceleration, Property 6: Batch Operation Metric Support
# **Validates: Requirements 3.5**
@given(
    dim=st.integers(min_value=2, max_value=50),
    n_vectors=st.integers(min_value=1, max_value=10),
    seed=st.integers(min_value=0, max_value=2**31-1)
)
@settings(max_examples=100, deadline=None)
def test_batch_matches_single_pair(dim, n_vectors, seed):
    """Batch operations should produce same results as computing distances individually"""
    np.random.seed(seed)
    query = np.random.randn(dim).astype(np.float32)
    vectors = np.random.randn(n_vectors, dim).astype(np.float32)
    
    # Test L2 metric
    batch_distances = hdb.l2_batch(query, vectors)
    
    for i in range(n_vectors):
        single_distance = hdb.l2(query, vectors[i])
        assert np.isclose(batch_distances[i], single_distance, rtol=1e-5), \
            f"Batch distance mismatch at index {i}: {batch_distances[i]} vs {single_distance}"


# ============================================================================
# Sparse Vector Property Tests
# ============================================================================

# Feature: python-vector-api-gpu-acceleration, Property 9: Sparse Distance Equivalence
# **Validates: Requirements 5.2, 7.2**
@given(
    dim=st.integers(min_value=10, max_value=100),
    sparsity=st.floats(min_value=0.7, max_value=0.95),  # 70-95% sparse
    seed=st.integers(min_value=0, max_value=2**31-1)
)
@settings(max_examples=100, deadline=None)
def test_sparse_l2_equivalence(dim, sparsity, seed):
    """For any two sparse vectors, sparse L2 distance should match dense L2 distance"""
    np.random.seed(seed)
    
    # Create sparse vectors
    n_nonzero_a = max(1, int(dim * (1 - sparsity)))
    n_nonzero_b = max(1, int(dim * (1 - sparsity)))
    
    indices_a = np.sort(np.random.choice(dim, size=n_nonzero_a, replace=False)).astype(np.uint32)
    values_a = np.random.randn(n_nonzero_a).astype(np.float32)
    
    indices_b = np.sort(np.random.choice(dim, size=n_nonzero_b, replace=False)).astype(np.uint32)
    values_b = np.random.randn(n_nonzero_b).astype(np.float32)
    
    # Create sparse vectors
    sparse_a = hdb.SparseVector(indices_a, values_a, dim)
    sparse_b = hdb.SparseVector(indices_b, values_b, dim)
    
    # Compute sparse distance
    sparse_dist = hdb.l2_sparse(sparse_a, sparse_b)
    
    # Convert to dense and compute dense distance
    dense_a = sparse_a.to_dense()
    dense_b = sparse_b.to_dense()
    dense_dist = hdb.l2(dense_a, dense_b)
    
    assert isinstance(sparse_dist, float)
    assert np.isclose(sparse_dist, dense_dist, rtol=1e-5, atol=1e-6), \
        f"Sparse L2 distance mismatch: {sparse_dist} vs {dense_dist}"


# Feature: python-vector-api-gpu-acceleration, Property 9: Sparse Distance Equivalence
# **Validates: Requirements 5.2, 7.2**
@given(
    dim=st.integers(min_value=10, max_value=100),
    sparsity=st.floats(min_value=0.7, max_value=0.95),
    seed=st.integers(min_value=0, max_value=2**31-1)
)
@settings(max_examples=100, deadline=None)
def test_sparse_cosine_equivalence(dim, sparsity, seed):
    """For any two sparse vectors, sparse cosine distance should match dense cosine distance"""
    np.random.seed(seed)
    
    # Create sparse vectors
    n_nonzero_a = max(1, int(dim * (1 - sparsity)))
    n_nonzero_b = max(1, int(dim * (1 - sparsity)))
    
    indices_a = np.sort(np.random.choice(dim, size=n_nonzero_a, replace=False)).astype(np.uint32)
    values_a = np.random.randn(n_nonzero_a).astype(np.float32)
    
    indices_b = np.sort(np.random.choice(dim, size=n_nonzero_b, replace=False)).astype(np.uint32)
    values_b = np.random.randn(n_nonzero_b).astype(np.float32)
    
    # Create sparse vectors
    sparse_a = hdb.SparseVector(indices_a, values_a, dim)
    sparse_b = hdb.SparseVector(indices_b, values_b, dim)
    
    # Compute sparse distance
    sparse_dist = hdb.cosine_sparse(sparse_a, sparse_b)
    
    # Convert to dense and compute dense distance
    dense_a = sparse_a.to_dense()
    dense_b = sparse_b.to_dense()
    dense_dist = hdb.cosine(dense_a, dense_b)
    
    assert isinstance(sparse_dist, float)
    assert np.isclose(sparse_dist, dense_dist, rtol=1e-5, atol=1e-6), \
        f"Sparse cosine distance mismatch: {sparse_dist} vs {dense_dist}"


# Feature: python-vector-api-gpu-acceleration, Property 9: Sparse Distance Equivalence
# **Validates: Requirements 5.2, 7.2**
@given(
    dim=st.integers(min_value=10, max_value=100),
    sparsity=st.floats(min_value=0.7, max_value=0.95),
    seed=st.integers(min_value=0, max_value=2**31-1)
)
@settings(max_examples=100, deadline=None)
def test_sparse_inner_product_equivalence(dim, sparsity, seed):
    """For any two sparse vectors, sparse inner product should match dense inner product"""
    np.random.seed(seed)
    
    # Create sparse vectors
    n_nonzero_a = max(1, int(dim * (1 - sparsity)))
    n_nonzero_b = max(1, int(dim * (1 - sparsity)))
    
    indices_a = np.sort(np.random.choice(dim, size=n_nonzero_a, replace=False)).astype(np.uint32)
    values_a = np.random.randn(n_nonzero_a).astype(np.float32)
    
    indices_b = np.sort(np.random.choice(dim, size=n_nonzero_b, replace=False)).astype(np.uint32)
    values_b = np.random.randn(n_nonzero_b).astype(np.float32)
    
    # Create sparse vectors
    sparse_a = hdb.SparseVector(indices_a, values_a, dim)
    sparse_b = hdb.SparseVector(indices_b, values_b, dim)
    
    # Compute sparse inner product
    sparse_prod = hdb.inner_product_sparse(sparse_a, sparse_b)
    
    # Convert to dense and compute dense inner product
    dense_a = sparse_a.to_dense()
    dense_b = sparse_b.to_dense()
    dense_prod = hdb.inner_product(dense_a, dense_b)
    
    assert isinstance(sparse_prod, float)
    assert np.isclose(sparse_prod, dense_prod, rtol=1e-5, atol=1e-6), \
        f"Sparse inner product mismatch: {sparse_prod} vs {dense_prod}"


# Feature: python-vector-api-gpu-acceleration, Property 10: Sparse Vector Validation
# **Validates: Requirements 5.5**
@given(
    dim=st.integers(min_value=10, max_value=100),
    n_nonzero=st.integers(min_value=2, max_value=20),
    seed=st.integers(min_value=0, max_value=2**31-1)
)
@settings(max_examples=100, deadline=None)
def test_sparse_vector_unsorted_indices_error(dim, n_nonzero, seed):
    """For any SparseVector with unsorted indices, the system should raise ValueError"""
    if n_nonzero >= dim:
        n_nonzero = dim - 1
    
    np.random.seed(seed)
    
    # Create unsorted indices
    indices = np.random.choice(dim, size=n_nonzero, replace=False).astype(np.uint32)
    # Ensure they're NOT sorted by reversing and making contiguous
    indices = np.ascontiguousarray(np.sort(indices)[::-1])
    values = np.random.randn(n_nonzero).astype(np.float32)
    
    with pytest.raises(ValueError, match="must be sorted"):
        hdb.SparseVector(indices, values, dim)


# Feature: python-vector-api-gpu-acceleration, Property 10: Sparse Vector Validation
# **Validates: Requirements 5.5**
@given(
    dim=st.integers(min_value=10, max_value=100),
    n_nonzero=st.integers(min_value=1, max_value=20),
    seed=st.integers(min_value=0, max_value=2**31-1)
)
@settings(max_examples=100, deadline=None)
def test_sparse_vector_out_of_bounds_error(dim, n_nonzero, seed):
    """For any SparseVector with out-of-bounds indices, the system should raise ValueError"""
    if n_nonzero >= dim:
        n_nonzero = dim - 1
    
    np.random.seed(seed)
    
    # Create valid indices but add one out-of-bounds index
    indices = np.sort(np.random.choice(dim, size=n_nonzero, replace=False)).astype(np.uint32)
    # Replace last index with out-of-bounds value
    indices[-1] = np.uint32(dim + 10)
    values = np.random.randn(n_nonzero).astype(np.float32)
    
    with pytest.raises(ValueError, match="out of bounds"):
        hdb.SparseVector(indices, values, dim)


# Feature: python-vector-api-gpu-acceleration, Property 10: Sparse Vector Validation
# **Validates: Requirements 5.5**
@given(
    dim=st.integers(min_value=10, max_value=100),
    n_indices=st.integers(min_value=1, max_value=20),
    n_values=st.integers(min_value=1, max_value=20),
    seed=st.integers(min_value=0, max_value=2**31-1)
)
@settings(max_examples=100, deadline=None)
def test_sparse_vector_length_mismatch_error(dim, n_indices, n_values, seed):
    """For any SparseVector with mismatched indices/values lengths, the system should raise ValueError"""
    # Adjust n_indices if it's too large
    if n_indices >= dim:
        n_indices = dim - 1
    
    # Skip when lengths match after adjustment
    if n_indices == n_values:
        return
    
    np.random.seed(seed)
    
    indices = np.sort(np.random.choice(dim, size=n_indices, replace=False)).astype(np.uint32)
    values = np.random.randn(n_values).astype(np.float32)
    
    with pytest.raises(ValueError, match="Length mismatch"):
        hdb.SparseVector(indices, values, dim)


# Feature: python-vector-api-gpu-acceleration, Property 10: Sparse Vector Validation
# **Validates: Requirements 5.5**
@given(
    dim=st.integers(min_value=10, max_value=100),
    n_nonzero=st.integers(min_value=1, max_value=20),
    nan_idx=st.integers(min_value=0, max_value=19),
    seed=st.integers(min_value=0, max_value=2**31-1)
)
@settings(max_examples=100, deadline=None)
def test_sparse_vector_nan_values_error(dim, n_nonzero, nan_idx, seed):
    """For any SparseVector with NaN values, the system should raise ValueError"""
    if n_nonzero >= dim:
        n_nonzero = dim - 1
    if nan_idx >= n_nonzero:
        nan_idx = n_nonzero - 1
    
    np.random.seed(seed)
    
    indices = np.sort(np.random.choice(dim, size=n_nonzero, replace=False)).astype(np.uint32)
    values = np.random.randn(n_nonzero).astype(np.float32)
    values[nan_idx] = np.nan
    
    with pytest.raises(ValueError, match="NaN"):
        hdb.SparseVector(indices, values, dim)


# Feature: python-vector-api-gpu-acceleration, Property 10: Sparse Vector Validation
# **Validates: Requirements 5.5**
@given(
    dim=st.integers(min_value=10, max_value=100),
    n_nonzero=st.integers(min_value=1, max_value=20),
    seed=st.integers(min_value=0, max_value=2**31-1)
)
@settings(max_examples=100, deadline=None)
def test_sparse_vector_dimension_mismatch_error(dim, n_nonzero, seed):
    """For any two sparse vectors with different dimensions, distance functions should raise ValueError"""
    if n_nonzero >= dim:
        n_nonzero = dim - 1
    
    np.random.seed(seed)
    
    # Create two sparse vectors with different dimensions
    indices_a = np.sort(np.random.choice(dim, size=n_nonzero, replace=False)).astype(np.uint32)
    values_a = np.random.randn(n_nonzero).astype(np.float32)
    sparse_a = hdb.SparseVector(indices_a, values_a, dim)
    
    dim_b = dim + 10
    indices_b = np.sort(np.random.choice(dim_b, size=n_nonzero, replace=False)).astype(np.uint32)
    values_b = np.random.randn(n_nonzero).astype(np.float32)
    sparse_b = hdb.SparseVector(indices_b, values_b, dim_b)
    
    with pytest.raises(ValueError, match="dimension mismatch"):
        hdb.l2_sparse(sparse_a, sparse_b)
