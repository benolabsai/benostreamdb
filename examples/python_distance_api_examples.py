"""
Python Distance API Examples

This file demonstrates the usage of HyperStreamDB's Python distance API with GPU acceleration.
It covers:
- Single-pair distance computations
- Batch operations for efficient similarity search
- Sparse vector operations
- Binary vector operations
- GPU context management

Requirements:
- hyperstreamdb installed
- numpy
- Optional: CUDA, ROCm, or Metal (MPS) for GPU acceleration
"""

import numpy as np
import hyperstreamdb as hdb

# ============================================================================
# Example 1: Single-Pair Distance Computations
# ============================================================================

def example_single_pair_distances():
    """Demonstrate computing distances between two vectors"""
    print("=" * 80)
    print("Example 1: Single-Pair Distance Computations")
    print("=" * 80)
    
    # Create two sample vectors
    vec_a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    vec_b = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    
    print(f"Vector A: {vec_a}")
    print(f"Vector B: {vec_b}")
    print()
    
    # Compute various distance metrics
    l2_dist = hdb.l2(vec_a, vec_b)
    print(f"L2 (Euclidean) distance: {l2_dist:.4f}")
    
    cosine_dist = hdb.cosine(vec_a, vec_b)
    print(f"Cosine distance: {cosine_dist:.4f}")
    
    inner_prod = hdb.inner_product(vec_a, vec_b)
    print(f"Inner product: {inner_prod:.4f}")
    
    l1_dist = hdb.l1(vec_a, vec_b)
    print(f"L1 (Manhattan) distance: {l1_dist:.4f}")
    
    hamming_dist = hdb.hamming(vec_a, vec_b)
    print(f"Hamming distance: {hamming_dist:.4f}")
    
    jaccard_dist = hdb.jaccard(vec_a, vec_b)
    print(f"Jaccard distance: {jaccard_dist:.4f}")
    print()


# ============================================================================
# Example 2: GPU Context Management
# ============================================================================

def example_gpu_context():
    """Demonstrate GPU context creation and management"""
    print("=" * 80)
    print("Example 2: GPU Context Management")
    print("=" * 80)
    
    # List available backends
    backends = hdb.ComputeContext.list_available_backends()
    print(f"Available backends: {backends}")
    print()
    
    # Auto-detect best backend
    ctx = hdb.ComputeContext.auto_detect()
    print(f"Auto-detected backend: {ctx.backend}")
    print(f"Device ID: {ctx.device_id}")
    print()
    
    # Create context with specific backend (if available)
    if 'cuda' in backends:
        cuda_ctx = hdb.ComputeContext('cuda', device_id=0)
        print(f"Created CUDA context: {cuda_ctx}")
    
    # Force CPU computation
    cpu_ctx = hdb.ComputeContext('cpu')
    print(f"Created CPU context: {cpu_ctx}")
    print()
    
    # Compute distance with GPU acceleration
    vec_a = np.random.rand(128).astype(np.float32)
    vec_b = np.random.rand(128).astype(np.float32)
    
    dist_gpu = hdb.l2(vec_a, vec_b, context=ctx)
    dist_cpu = hdb.l2(vec_a, vec_b, context=cpu_ctx)
    
    print(f"Distance (GPU): {dist_gpu:.6f}")
    print(f"Distance (CPU): {dist_cpu:.6f}")
    print(f"Difference: {abs(dist_gpu - dist_cpu):.10f}")
    print()


# ============================================================================
# Example 3: Batch Operations for Similarity Search
# ============================================================================

def example_batch_operations():
    """Demonstrate efficient batch distance computations"""
    print("=" * 80)
    print("Example 3: Batch Operations for Similarity Search")
    print("=" * 80)
    
    # Create a query vector and a database of vectors
    dim = 128
    n_vectors = 10000
    
    query = np.random.rand(dim).astype(np.float32)
    database = np.random.rand(n_vectors, dim).astype(np.float32)
    
    print(f"Query vector dimension: {dim}")
    print(f"Database size: {n_vectors} vectors")
    print()
    
    # Create GPU context for acceleration
    ctx = hdb.ComputeContext.auto_detect()
    print(f"Using backend: {ctx.backend}")
    print()
    
    # Compute distances to all database vectors
    import time
    
    start = time.time()
    distances = hdb.l2_batch(query, database, context=ctx)
    elapsed = time.time() - start
    
    print(f"Computed {len(distances)} distances in {elapsed*1000:.2f}ms")
    print(f"Throughput: {len(distances)/elapsed:.0f} distances/sec")
    print()
    
    # Find top-k nearest neighbors
    k = 5
    top_k_indices = np.argsort(distances)[:k]
    top_k_distances = distances[top_k_indices]
    
    print(f"Top-{k} nearest neighbors:")
    for i, (idx, dist) in enumerate(zip(top_k_indices, top_k_distances)):
        print(f"  {i+1}. Index {idx}: distance = {dist:.4f}")
    print()
    
    # Compare different metrics
    print("Comparing different distance metrics:")
    l2_dists = hdb.l2_batch(query, database[:100], context=ctx)
    cosine_dists = hdb.cosine_batch(query, database[:100], context=ctx)
    inner_prods = hdb.inner_product_batch(query, database[:100], context=ctx)
    
    print(f"  L2 distances (first 5): {l2_dists[:5]}")
    print(f"  Cosine distances (first 5): {cosine_dists[:5]}")
    print(f"  Inner products (first 5): {inner_prods[:5]}")
    print()


# ============================================================================
# Example 4: Sparse Vector Operations
# ============================================================================

def example_sparse_vectors():
    """Demonstrate sparse vector distance computations"""
    print("=" * 80)
    print("Example 4: Sparse Vector Operations")
    print("=" * 80)
    
    # Create sparse vectors (high-dimensional with few non-zero elements)
    dim = 10000
    
    # Sparse vector A: non-zero at indices [0, 100, 500, 1000]
    indices_a = np.array([0, 100, 500, 1000], dtype=np.uint32)
    values_a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    sparse_a = hdb.SparseVector(indices_a, values_a, dim)
    
    # Sparse vector B: non-zero at indices [0, 200, 500, 1500]
    indices_b = np.array([0, 200, 500, 1500], dtype=np.uint32)
    values_b = np.array([1.5, 1.0, 3.5, 2.0], dtype=np.float32)
    sparse_b = hdb.SparseVector(indices_b, values_b, dim)
    
    print(f"Sparse vector A: {len(indices_a)} non-zero elements in {dim} dimensions")
    print(f"  Indices: {indices_a}")
    print(f"  Values: {values_a}")
    print()
    
    print(f"Sparse vector B: {len(indices_b)} non-zero elements in {dim} dimensions")
    print(f"  Indices: {indices_b}")
    print(f"  Values: {values_b}")
    print()
    
    # Compute sparse distances
    l2_sparse = hdb.l2_sparse(sparse_a, sparse_b)
    cosine_sparse = hdb.cosine_sparse(sparse_a, sparse_b)
    inner_prod_sparse = hdb.inner_product_sparse(sparse_a, sparse_b)
    
    print("Sparse distance computations:")
    print(f"  L2 distance: {l2_sparse:.4f}")
    print(f"  Cosine distance: {cosine_sparse:.4f}")
    print(f"  Inner product: {inner_prod_sparse:.4f}")
    print()
    
    # Verify equivalence with dense computation
    dense_a = sparse_a.to_dense()
    dense_b = sparse_b.to_dense()
    
    l2_dense = hdb.l2(dense_a, dense_b)
    cosine_dense = hdb.cosine(dense_a, dense_b)
    inner_prod_dense = hdb.inner_product(dense_a, dense_b)
    
    print("Dense distance computations (for verification):")
    print(f"  L2 distance: {l2_dense:.4f}")
    print(f"  Cosine distance: {cosine_dense:.4f}")
    print(f"  Inner product: {inner_prod_dense:.4f}")
    print()
    
    print("Differences (sparse vs dense):")
    print(f"  L2: {abs(l2_sparse - l2_dense):.10f}")
    print(f"  Cosine: {abs(cosine_sparse - cosine_dense):.10f}")
    print(f"  Inner product: {abs(inner_prod_sparse - inner_prod_dense):.10f}")
    print()


# ============================================================================
# Example 5: Binary Vector Operations
# ============================================================================

def example_binary_vectors():
    """Demonstrate binary vector distance computations"""
    print("=" * 80)
    print("Example 5: Binary Vector Operations")
    print("=" * 80)
    
    # Create binary vectors (bit-packed)
    # Each byte stores 8 bits
    binary_a = np.array([0b10110101, 0b11001100, 0b00110011], dtype=np.uint8)
    binary_b = np.array([0b10101100, 0b11110000, 0b00111111], dtype=np.uint8)
    
    print(f"Binary vector A (packed): {binary_a}")
    print(f"Binary vector B (packed): {binary_b}")
    print()
    
    # Compute Hamming distance (count of differing bits)
    hamming_dist = hdb.hamming_packed(binary_a, binary_b)
    print(f"Hamming distance (packed): {hamming_dist} bits")
    
    # Compute Jaccard distance
    jaccard_dist = hdb.jaccard_packed(binary_a, binary_b)
    print(f"Jaccard distance (packed): {jaccard_dist:.4f}")
    print()
    
    # Auto-packing: work with unpacked binary vectors
    print("Auto-packing example:")
    unpacked_a = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    unpacked_b = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    
    print(f"Unpacked vector A: {unpacked_a}")
    print(f"Unpacked vector B: {unpacked_b}")
    print()
    
    # These functions automatically pack the vectors
    hamming_auto = hdb.hamming_auto(unpacked_a, unpacked_b)
    jaccard_auto = hdb.jaccard_auto(unpacked_a, unpacked_b)
    
    print(f"Hamming distance (auto-packed): {hamming_auto} bits")
    print(f"Jaccard distance (auto-packed): {jaccard_auto:.4f}")
    print()


# ============================================================================
# Example 6: Performance Monitoring
# ============================================================================

def example_performance_monitoring():
    """Demonstrate GPU performance monitoring"""
    print("=" * 80)
    print("Example 6: Performance Monitoring")
    print("=" * 80)
    
    # Create GPU context
    ctx = hdb.ComputeContext.auto_detect()
    print(f"Using backend: {ctx.backend}")
    print()
    
    # Reset stats before measurement
    ctx.reset_stats()
    
    # Perform some computations
    query = np.random.rand(256).astype(np.float32)
    database = np.random.rand(50000, 256).astype(np.float32)
    
    print(f"Computing distances for {len(database)} vectors...")
    distances = hdb.l2_batch(query, database, context=ctx)
    print(f"Computed {len(distances)} distances")
    print()
    
    # Get performance statistics
    stats = ctx.get_stats()
    
    print("Performance Statistics:")
    print(f"  Total kernel launches: {stats['total_kernel_launches']}")
    print(f"  Total GPU time: {stats['total_gpu_time_ms']:.2f}ms")
    print(f"  Total CPU time: {stats['total_cpu_time_ms']:.2f}ms")
    print(f"  Vectors processed: {stats['total_vectors_processed']}")
    print(f"  Memory transfers: {stats['memory_transfers_mb']:.2f}MB")
    print()
    
    if stats['total_gpu_time_ms'] > 0:
        throughput = stats['total_vectors_processed'] / (stats['total_gpu_time_ms'] / 1000)
        print(f"  GPU throughput: {throughput:.0f} vectors/sec")
    print()


# ============================================================================
# Example 7: Real-World Use Case - Semantic Search
# ============================================================================

def example_semantic_search():
    """Demonstrate a real-world semantic search use case"""
    print("=" * 80)
    print("Example 7: Real-World Use Case - Semantic Search")
    print("=" * 80)
    
    # Simulate document embeddings (in practice, these would come from a model)
    dim = 384  # Common embedding dimension
    n_documents = 1000
    
    # Create document embeddings
    document_embeddings = np.random.rand(n_documents, dim).astype(np.float32)
    # Normalize for cosine similarity
    document_embeddings = document_embeddings / np.linalg.norm(document_embeddings, axis=1, keepdims=True)
    
    # Create query embedding
    query_embedding = np.random.rand(dim).astype(np.float32)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    
    print(f"Document database: {n_documents} documents")
    print(f"Embedding dimension: {dim}")
    print()
    
    # Use GPU acceleration for search
    ctx = hdb.ComputeContext.auto_detect()
    
    # Compute cosine distances (lower is more similar)
    distances = hdb.cosine_batch(query_embedding, document_embeddings, context=ctx)
    
    # Find top-10 most similar documents
    k = 10
    top_k_indices = np.argsort(distances)[:k]
    top_k_distances = distances[top_k_indices]
    
    print(f"Top-{k} most similar documents:")
    for i, (idx, dist) in enumerate(zip(top_k_indices, top_k_distances)):
        similarity = 1.0 - dist  # Convert distance to similarity
        print(f"  {i+1}. Document {idx}: similarity = {similarity:.4f}")
    print()


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all examples"""
    print("\n")
    print("*" * 80)
    print("HyperStreamDB Python Distance API Examples")
    print("*" * 80)
    print("\n")
    
    try:
        example_single_pair_distances()
        example_gpu_context()
        example_batch_operations()
        example_sparse_vectors()
        example_binary_vectors()
        example_performance_monitoring()
        example_semantic_search()
        
        print("=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
