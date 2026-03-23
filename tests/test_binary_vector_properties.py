"""
Property-based tests for binary vector operations

Feature: python-vector-api-gpu-acceleration
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings
import hyperstreamdb as hdb


# ============================================================================
# Property 11: Binary Hamming Distance Correctness
# **Validates: Requirements 6.2**
# ============================================================================

@given(
    num_bits=st.integers(min_value=8, max_value=256),
    seed=st.integers(min_value=0, max_value=2**31-1)
)
@settings(max_examples=100)
def test_binary_hamming_distance_correctness(num_bits, seed):
    """
    For any two bit-packed binary vectors, computing Hamming distance using 
    hamming_packed should return the correct count of differing bits.
    """
    np.random.seed(seed)
    
    # Generate random binary vectors
    num_bytes = (num_bits + 7) // 8
    a_packed = np.random.randint(0, 256, size=num_bytes, dtype=np.uint8)
    b_packed = np.random.randint(0, 256, size=num_bytes, dtype=np.uint8)
    
    # Compute distance using our function
    result = hdb.hamming_packed(a_packed, b_packed)
    
    # Compute expected distance by unpacking and comparing bits
    expected = 0
    for byte_a, byte_b in zip(a_packed, b_packed):
        xor = byte_a ^ byte_b
        expected += bin(xor).count('1')
    
    assert result == expected, f"Expected {expected} differing bits, got {result}"
    assert isinstance(result, (int, np.integer)), f"Result should be integer, got {type(result)}"


@given(
    num_bits=st.integers(min_value=8, max_value=256),
    seed=st.integers(min_value=0, max_value=2**31-1)
)
@settings(max_examples=100)
def test_binary_hamming_distance_identical_vectors(num_bits, seed):
    """
    For any binary vector, the Hamming distance to itself should be 0.
    """
    np.random.seed(seed)
    
    num_bytes = (num_bits + 7) // 8
    a_packed = np.random.randint(0, 256, size=num_bytes, dtype=np.uint8)
    
    result = hdb.hamming_packed(a_packed, a_packed)
    
    assert result == 0, f"Hamming distance to self should be 0, got {result}"


@given(
    num_bits=st.integers(min_value=8, max_value=256),
    seed=st.integers(min_value=0, max_value=2**31-1)
)
@settings(max_examples=100)
def test_binary_hamming_distance_symmetry(num_bits, seed):
    """
    For any two binary vectors, hamming(a, b) should equal hamming(b, a).
    """
    np.random.seed(seed)
    
    num_bytes = (num_bits + 7) // 8
    a_packed = np.random.randint(0, 256, size=num_bytes, dtype=np.uint8)
    b_packed = np.random.randint(0, 256, size=num_bytes, dtype=np.uint8)
    
    result_ab = hdb.hamming_packed(a_packed, b_packed)
    result_ba = hdb.hamming_packed(b_packed, a_packed)
    
    assert result_ab == result_ba, f"Hamming distance should be symmetric: {result_ab} != {result_ba}"


# ============================================================================
# Property 12: Binary Vector Auto-Packing
# **Validates: Requirements 6.4**
# ============================================================================

@given(
    num_bits=st.integers(min_value=8, max_value=128).filter(lambda x: x % 8 == 0),
    seed=st.integers(min_value=0, max_value=2**31-1)
)
@settings(max_examples=100)
def test_binary_vector_auto_packing_hamming(num_bits, seed):
    """
    For any unpacked binary vector (array of 0s and 1s), the system should 
    automatically pack it and compute the same Hamming distance as manually 
    packed vectors.
    """
    np.random.seed(seed)
    
    # Generate unpacked binary vectors (0.0 and 1.0)
    a_unpacked = np.random.randint(0, 2, size=num_bits).astype(np.float32)
    b_unpacked = np.random.randint(0, 2, size=num_bits).astype(np.float32)
    
    # Manually pack the vectors
    def pack_vector(unpacked):
        num_bytes = (len(unpacked) + 7) // 8
        packed = np.zeros(num_bytes, dtype=np.uint8)
        for i, val in enumerate(unpacked):
            if val != 0.0:
                byte_idx = i // 8
                bit_idx = i % 8
                packed[byte_idx] |= 1 << (7 - bit_idx)  # MSB first
        return packed
    
    a_packed = pack_vector(a_unpacked)
    b_packed = pack_vector(b_unpacked)
    
    # Compute distance using auto-packing
    result_auto = hdb.hamming_auto(a_unpacked, b_unpacked)
    
    # Compute distance using manually packed vectors
    result_manual = hdb.hamming_packed(a_packed, b_packed)
    
    assert result_auto == result_manual, \
        f"Auto-packed distance {result_auto} should equal manually packed distance {result_manual}"


@given(
    num_bits=st.integers(min_value=8, max_value=128).filter(lambda x: x % 8 == 0),
    seed=st.integers(min_value=0, max_value=2**31-1)
)
@settings(max_examples=100)
def test_binary_vector_auto_packing_jaccard(num_bits, seed):
    """
    For any unpacked binary vector (array of 0s and 1s), the system should 
    automatically pack it and compute the same Jaccard distance as manually 
    packed vectors.
    """
    np.random.seed(seed)
    
    # Generate unpacked binary vectors (0.0 and 1.0)
    a_unpacked = np.random.randint(0, 2, size=num_bits).astype(np.float32)
    b_unpacked = np.random.randint(0, 2, size=num_bits).astype(np.float32)
    
    # Manually pack the vectors
    def pack_vector(unpacked):
        num_bytes = (len(unpacked) + 7) // 8
        packed = np.zeros(num_bytes, dtype=np.uint8)
        for i, val in enumerate(unpacked):
            if val != 0.0:
                byte_idx = i // 8
                bit_idx = i % 8
                packed[byte_idx] |= 1 << (7 - bit_idx)  # MSB first
        return packed
    
    a_packed = pack_vector(a_unpacked)
    b_packed = pack_vector(b_unpacked)
    
    # Compute distance using auto-packing
    result_auto = hdb.jaccard_auto(a_unpacked, b_unpacked)
    
    # Compute distance using manually packed vectors
    result_manual = hdb.jaccard_packed(a_packed, b_packed)
    
    assert np.isclose(result_auto, result_manual, rtol=1e-5), \
        f"Auto-packed distance {result_auto} should equal manually packed distance {result_manual}"


@given(
    num_bits=st.integers(min_value=8, max_value=128),
    seed=st.integers(min_value=0, max_value=2**31-1)
)
@settings(max_examples=50)
def test_binary_vector_auto_packing_rejects_non_binary(num_bits, seed):
    """
    For any vector with non-binary values, auto-packing should raise a ValueError.
    """
    np.random.seed(seed)
    
    # Generate non-binary vectors
    a = np.random.randn(num_bits).astype(np.float32)
    b = np.random.randn(num_bits).astype(np.float32)
    
    # Should raise ValueError for non-binary values
    with pytest.raises(ValueError, match="must contain only 0.0 or 1.0"):
        hdb.hamming_auto(a, b)
    
    with pytest.raises(ValueError, match="must contain only 0.0 or 1.0"):
        hdb.jaccard_auto(a, b)


# ============================================================================
# Additional Binary Vector Tests
# ============================================================================

def test_binary_jaccard_distance_correctness():
    """
    Test Jaccard distance computation for binary vectors with known values.
    """
    # Binary vectors: 10110101 and 10101100
    a = np.array([0b10110101], dtype=np.uint8)
    b = np.array([0b10101100], dtype=np.uint8)
    
    # Count set bits
    # a: 10110101 -> 5 bits set
    # b: 10101100 -> 4 bits set
    # intersection (a & b): 10100100 -> 3 bits set
    # union (a | b): 10111101 -> 6 bits set
    # Jaccard similarity = 3/6 = 0.5
    # Jaccard distance = 1 - 0.5 = 0.5
    
    result = hdb.jaccard_packed(a, b)
    expected = 0.5
    
    assert np.isclose(result, expected, rtol=1e-5), \
        f"Expected Jaccard distance {expected}, got {result}"


def test_binary_jaccard_distance_identical():
    """
    Test that Jaccard distance between identical vectors is 0.
    """
    a = np.array([0b10110101, 0b11001100], dtype=np.uint8)
    
    result = hdb.jaccard_packed(a, a)
    
    assert np.isclose(result, 0.0, rtol=1e-5), \
        f"Jaccard distance to self should be 0, got {result}"


def test_binary_jaccard_distance_all_zeros():
    """
    Test that Jaccard distance between all-zero vectors is 0.
    """
    a = np.array([0, 0, 0], dtype=np.uint8)
    b = np.array([0, 0, 0], dtype=np.uint8)
    
    result = hdb.jaccard_packed(a, b)
    
    assert np.isclose(result, 0.0, rtol=1e-5), \
        f"Jaccard distance between all-zero vectors should be 0, got {result}"


def test_binary_vector_dimension_mismatch():
    """
    Test that dimension mismatch raises ValueError.
    """
    a = np.array([0b10110101], dtype=np.uint8)
    b = np.array([0b10101100, 0b11001100], dtype=np.uint8)
    
    with pytest.raises(ValueError, match="length mismatch"):
        hdb.hamming_packed(a, b)
    
    with pytest.raises(ValueError, match="length mismatch"):
        hdb.jaccard_packed(a, b)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
