#!/usr/bin/env python3
"""
Test script to verify all Python module exports are working correctly.
This validates Task 12.1: Update src/lib.rs to export new Python modules
"""

import sys
import numpy as np

def test_imports():
    """Test that all modules and functions can be imported"""
    print("Testing imports...")
    try:
        import hyperstreamdb as hdb
        print("✓ hyperstreamdb module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import hyperstreamdb: {e}")
        return False
    
    # Test GPU Context API
    print("\nTesting GPU Context API...")
    try:
        ctx = hdb.ComputeContext.auto_detect()
        print(f"✓ ComputeContext.auto_detect() works: backend={ctx.backend}")
        
        backends = hdb.ComputeContext.list_available_backends()
        print(f"✓ ComputeContext.list_available_backends() works: {backends}")
        
        cpu_ctx = hdb.ComputeContext('cpu')
        print(f"✓ ComputeContext('cpu') works: backend={cpu_ctx.backend}")
    except Exception as e:
        print(f"✗ GPU Context API failed: {e}")
        return False
    
    # Test Distance API - Single-pair functions
    print("\nTesting Distance API - Single-pair functions...")
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    
    functions = [
        ('l2', hdb.l2),
        ('cosine', hdb.cosine),
        ('inner_product', hdb.inner_product),
        ('l1', hdb.l1),
        ('hamming', hdb.hamming),
        ('jaccard', hdb.jaccard),
    ]
    
    for name, func in functions:
        try:
            result = func(a, b)
            print(f"✓ {name}(a, b) = {result}")
        except Exception as e:
            print(f"✗ {name} failed: {e}")
            return False
    
    # Test Distance API - Batch functions
    print("\nTesting Distance API - Batch functions...")
    query = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    vectors = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ], dtype=np.float32)
    
    batch_functions = [
        ('l2_batch', hdb.l2_batch),
        ('cosine_batch', hdb.cosine_batch),
        ('inner_product_batch', hdb.inner_product_batch),
        ('l1_batch', hdb.l1_batch),
        ('hamming_batch', hdb.hamming_batch),
        ('jaccard_batch', hdb.jaccard_batch),
    ]
    
    for name, func in batch_functions:
        try:
            result = func(query, vectors)
            print(f"✓ {name}(query, vectors) shape = {result.shape}, values = {result}")
        except Exception as e:
            print(f"✗ {name} failed: {e}")
            return False
    
    # Test Sparse Vector API
    print("\nTesting Sparse Vector API...")
    try:
        indices = np.array([0, 5, 10], dtype=np.uint32)
        values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        sparse_a = hdb.SparseVector(indices, values, 100)
        print(f"✓ SparseVector created: dim={sparse_a.dim}, nnz={len(sparse_a.indices)}")
        
        sparse_b = hdb.SparseVector(np.array([0, 3, 10], dtype=np.uint32), 
                                     np.array([1.0, 4.0, 3.0], dtype=np.float32), 100)
        
        l2_dist = hdb.l2_sparse(sparse_a, sparse_b)
        print(f"✓ l2_sparse(sparse_a, sparse_b) = {l2_dist}")
        
        cos_dist = hdb.cosine_sparse(sparse_a, sparse_b)
        print(f"✓ cosine_sparse(sparse_a, sparse_b) = {cos_dist}")
        
        ip = hdb.inner_product_sparse(sparse_a, sparse_b)
        print(f"✓ inner_product_sparse(sparse_a, sparse_b) = {ip}")
        
        dense = sparse_a.to_dense()
        print(f"✓ sparse_a.to_dense() shape = {dense.shape}")
    except Exception as e:
        print(f"✗ Sparse Vector API failed: {e}")
        return False
    
    # Test Binary Vector API
    print("\nTesting Binary Vector API...")
    try:
        # Test packed binary vectors
        a_packed = np.array([0b10110101], dtype=np.uint8)
        b_packed = np.array([0b10101100], dtype=np.uint8)
        
        hamming_dist = hdb.hamming_packed(a_packed, b_packed)
        print(f"✓ hamming_packed(a_packed, b_packed) = {hamming_dist}")
        
        jaccard_dist = hdb.jaccard_packed(a_packed, b_packed)
        print(f"✓ jaccard_packed(a_packed, b_packed) = {jaccard_dist}")
        
        # Test auto-packing
        a_unpacked = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        b_unpacked = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float32)
        
        hamming_auto = hdb.hamming_auto(a_unpacked, b_unpacked)
        print(f"✓ hamming_auto(a_unpacked, b_unpacked) = {hamming_auto}")
        
        jaccard_auto = hdb.jaccard_auto(a_unpacked, b_unpacked)
        print(f"✓ jaccard_auto(a_unpacked, b_unpacked) = {jaccard_auto}")
    except Exception as e:
        print(f"✗ Binary Vector API failed: {e}")
        return False
    
    print("\n" + "="*60)
    print("✓ All exports verified successfully!")
    print("="*60)
    return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
