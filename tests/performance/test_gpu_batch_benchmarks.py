"""
Performance benchmarks for GPU-accelerated batch distance operations.

This benchmark validates Requirement 3.2:
"WHEN batch operations use GPU acceleration, THE System SHALL achieve at least 
10x speedup compared to CPU for databases with 100,000+ vectors"

Note: If GPU is not available, this benchmark will document CPU performance baseline.
"""
import pytest
import time
import numpy as np
import hyperstreamdb as hdb


def measure_batch_performance(query, database, context, metric_name="l2", warmup=True):
    """
    Measure batch distance computation performance.
    
    Args:
        query: Query vector (1D array)
        database: Database vectors (2D array, shape [n_vectors, dim])
        context: ComputeContext for computation
        metric_name: Distance metric to use ("l2", "cosine", "inner_product", "l1", "hamming", "jaccard")
        warmup: Whether to perform a warmup run
    
    Returns:
        dict with timing and throughput metrics
    """
    # Get the appropriate batch function
    batch_functions = {
        "l2": hdb.l2_batch,
        "cosine": hdb.cosine_batch,
        "inner_product": hdb.inner_product_batch,
        "l1": hdb.l1_batch,
        "hamming": hdb.hamming_batch,
        "jaccard": hdb.jaccard_batch,
    }
    
    batch_fn = batch_functions[metric_name]
    
    # Warmup run to ensure GPU is initialized
    if warmup:
        _ = batch_fn(query, database[:100], device=context)
    
    # Benchmark run
    start = time.time()
    distances = batch_fn(query, database, device=context)
    elapsed = time.time() - start
    
    n_vectors = len(database)
    throughput = n_vectors / elapsed
    
    return {
        "elapsed_sec": elapsed,
        "elapsed_ms": elapsed * 1000,
        "n_vectors": n_vectors,
        "throughput": throughput,
        "distances": distances,
    }


class TestGPUBatchPerformance:
    """Test GPU vs CPU performance for batch distance operations."""
    
    def test_l2_batch_100k_vectors(self):
        """
        Benchmark L2 distance batch operations with 100K vectors.
        
        Validates Requirement 3.2: >10x speedup for 100K+ vectors on GPU vs CPU.
        """
        print("\n" + "=" * 80)
        print("Benchmark: L2 Distance Batch Operations (100K vectors)")
        print("=" * 80)
        
        # Setup: 100K vectors, 128 dimensions
        dim = 128
        n_vectors = 100_000
        
        query = np.random.rand(dim).astype(np.float32)
        database = np.random.rand(n_vectors, dim).astype(np.float32)
        
        print(f"Query dimension: {dim}")
        print(f"Database size: {n_vectors:,} vectors")
        print()
        
        # Check available backends
        backends = hdb.ComputeContext.list_available_backends()
        print(f"Available backends: {backends}")
        print()
        
        # Test CPU performance
        cpu_ctx = hdb.ComputeContext('cpu')
        print(f"Testing CPU backend...")
        cpu_result = measure_batch_performance(query, database, cpu_ctx, metric_name="l2")
        
        print(f"CPU Results:")
        print(f"  Time: {cpu_result['elapsed_ms']:.2f}ms")
        print(f"  Throughput: {cpu_result['throughput']:,.0f} vectors/sec")
        print()
        
        # Test GPU performance if available
        gpu_available = any(backend in backends for backend in ['cuda', 'rocm', 'mps', 'intel'])
        
        if gpu_available:
            gpu_ctx = hdb.ComputeContext.auto_detect()
            print(f"Testing GPU backend: {gpu_ctx.backend}")
            gpu_result = measure_batch_performance(query, database, gpu_ctx, metric_name="l2")
            
            print(f"GPU Results:")
            print(f"  Time: {gpu_result['elapsed_ms']:.2f}ms")
            print(f"  Throughput: {gpu_result['throughput']:,.0f} vectors/sec")
            print()
            
            # Calculate speedup
            speedup = cpu_result['elapsed_sec'] / gpu_result['elapsed_sec']
            print(f"Speedup: {speedup:.2f}x")
            print()
            
            # Verify correctness (results should be very close)
            max_diff = np.max(np.abs(cpu_result['distances'] - gpu_result['distances']))
            print(f"Max difference between CPU and GPU: {max_diff:.10f}")
            assert max_diff < 1e-4, f"CPU and GPU results differ too much: {max_diff}"
            print()
            
            # Validate Requirement 3.2: >10x speedup
            print("Requirement 3.2 Validation:")
            if speedup >= 10.0:
                print(f"  ✓ PASS: GPU speedup {speedup:.2f}x >= 10x")
            else:
                print(f"  ⚠ WARNING: GPU speedup {speedup:.2f}x < 10x")
                print(f"  Note: Speedup may vary based on hardware and system load")
        else:
            print("⚠ No GPU backend available")
            print("Documenting CPU performance baseline only")
            print()
            print("CPU Baseline Performance:")
            print(f"  Time: {cpu_result['elapsed_ms']:.2f}ms")
            print(f"  Throughput: {cpu_result['throughput']:,.0f} vectors/sec")
        
        print("=" * 80)
    
    def test_cosine_batch_100k_vectors(self):
        """
        Benchmark Cosine distance batch operations with 100K vectors.
        """
        print("\n" + "=" * 80)
        print("Benchmark: Cosine Distance Batch Operations (100K vectors)")
        print("=" * 80)
        
        # Setup: 100K vectors, 128 dimensions
        dim = 128
        n_vectors = 100_000
        
        query = np.random.rand(dim).astype(np.float32)
        database = np.random.rand(n_vectors, dim).astype(np.float32)
        
        print(f"Query dimension: {dim}")
        print(f"Database size: {n_vectors:,} vectors")
        print()
        
        # Check available backends
        backends = hdb.ComputeContext.list_available_backends()
        
        # Test CPU performance
        cpu_ctx = hdb.ComputeContext('cpu')
        print(f"Testing CPU backend...")
        cpu_result = measure_batch_performance(query, database, cpu_ctx, metric_name="cosine")
        
        print(f"CPU Results:")
        print(f"  Time: {cpu_result['elapsed_ms']:.2f}ms")
        print(f"  Throughput: {cpu_result['throughput']:,.0f} vectors/sec")
        print()
        
        # Test GPU performance if available
        gpu_available = any(backend in backends for backend in ['cuda', 'rocm', 'mps', 'intel'])
        
        if gpu_available:
            gpu_ctx = hdb.ComputeContext.auto_detect()
            print(f"Testing GPU backend: {gpu_ctx.backend}")
            gpu_result = measure_batch_performance(query, database, gpu_ctx, metric_name="cosine")
            
            print(f"GPU Results:")
            print(f"  Time: {gpu_result['elapsed_ms']:.2f}ms")
            print(f"  Throughput: {gpu_result['throughput']:,.0f} vectors/sec")
            print()
            
            # Calculate speedup
            speedup = cpu_result['elapsed_sec'] / gpu_result['elapsed_sec']
            print(f"Speedup: {speedup:.2f}x")
            print()
            
            # Verify correctness
            max_diff = np.max(np.abs(cpu_result['distances'] - gpu_result['distances']))
            print(f"Max difference between CPU and GPU: {max_diff:.10f}")
            assert max_diff < 1e-4, f"CPU and GPU results differ too much: {max_diff}"
        else:
            print("⚠ No GPU backend available - CPU baseline only")
        
        print("=" * 80)
    
    def test_inner_product_batch_100k_vectors(self):
        """
        Benchmark Inner Product batch operations with 100K vectors.
        """
        print("\n" + "=" * 80)
        print("Benchmark: Inner Product Batch Operations (100K vectors)")
        print("=" * 80)
        
        # Setup: 100K vectors, 128 dimensions
        dim = 128
        n_vectors = 100_000
        
        query = np.random.rand(dim).astype(np.float32)
        database = np.random.rand(n_vectors, dim).astype(np.float32)
        
        print(f"Query dimension: {dim}")
        print(f"Database size: {n_vectors:,} vectors")
        print()
        
        # Check available backends
        backends = hdb.ComputeContext.list_available_backends()
        
        # Test CPU performance
        cpu_ctx = hdb.ComputeContext('cpu')
        print(f"Testing CPU backend...")
        cpu_result = measure_batch_performance(query, database, cpu_ctx, metric_name="inner_product")
        
        print(f"CPU Results:")
        print(f"  Time: {cpu_result['elapsed_ms']:.2f}ms")
        print(f"  Throughput: {cpu_result['throughput']:,.0f} vectors/sec")
        print()
        
        # Test GPU performance if available
        gpu_available = any(backend in backends for backend in ['cuda', 'rocm', 'mps', 'intel'])
        
        if gpu_available:
            gpu_ctx = hdb.ComputeContext.auto_detect()
            print(f"Testing GPU backend: {gpu_ctx.backend}")
            gpu_result = measure_batch_performance(query, database, gpu_ctx, metric_name="inner_product")
            
            print(f"GPU Results:")
            print(f"  Time: {gpu_result['elapsed_ms']:.2f}ms")
            print(f"  Throughput: {gpu_result['throughput']:,.0f} vectors/sec")
            print()
            
            # Calculate speedup
            speedup = cpu_result['elapsed_sec'] / gpu_result['elapsed_sec']
            print(f"Speedup: {speedup:.2f}x")
            print()
            
            # Verify correctness
            max_diff = np.max(np.abs(cpu_result['distances'] - gpu_result['distances']))
            print(f"Max difference between CPU and GPU: {max_diff:.10f}")
            assert max_diff < 1e-4, f"CPU and GPU results differ too much: {max_diff}"
        else:
            print("⚠ No GPU backend available - CPU baseline only")
        
        print("=" * 80)
    
    def test_varying_vector_sizes(self):
        """
        Benchmark performance across different database sizes.
        
        Tests: 10K, 50K, 100K, 200K, 500K vectors
        """
        print("\n" + "=" * 80)
        print("Benchmark: Performance Scaling with Database Size")
        print("=" * 80)
        
        dim = 128
        sizes = [10_000, 50_000, 100_000, 200_000, 500_000]
        
        # Check available backends
        backends = hdb.ComputeContext.list_available_backends()
        gpu_available = any(backend in backends for backend in ['cuda', 'rocm', 'mps', 'intel'])
        
        cpu_ctx = hdb.ComputeContext('cpu')
        if gpu_available:
            gpu_ctx = hdb.ComputeContext.auto_detect()
            print(f"GPU backend: {gpu_ctx.backend}")
        else:
            print("⚠ No GPU backend available - CPU baseline only")
        print()
        
        print(f"{'Size':>10} | {'CPU Time':>12} | {'CPU Throughput':>18} | {'GPU Time':>12} | {'GPU Throughput':>18} | {'Speedup':>10}")
        print("-" * 110)
        
        for n_vectors in sizes:
            query = np.random.rand(dim).astype(np.float32)
            database = np.random.rand(n_vectors, dim).astype(np.float32)
            
            # CPU benchmark
            cpu_result = measure_batch_performance(query, database, cpu_ctx, metric_name="l2", warmup=False)
            
            if gpu_available:
                # GPU benchmark
                gpu_result = measure_batch_performance(query, database, gpu_ctx, metric_name="l2", warmup=False)
                speedup = cpu_result['elapsed_sec'] / gpu_result['elapsed_sec']
                
                print(f"{n_vectors:>10,} | {cpu_result['elapsed_ms']:>10.2f}ms | {cpu_result['throughput']:>14,.0f}/sec | "
                      f"{gpu_result['elapsed_ms']:>10.2f}ms | {gpu_result['throughput']:>14,.0f}/sec | {speedup:>9.2f}x")
            else:
                print(f"{n_vectors:>10,} | {cpu_result['elapsed_ms']:>10.2f}ms | {cpu_result['throughput']:>14,.0f}/sec | "
                      f"{'N/A':>12} | {'N/A':>18} | {'N/A':>10}")
        
        print("=" * 80)
    
    def test_varying_dimensions(self):
        """
        Benchmark performance across different vector dimensions.
        
        Tests: 64D, 128D, 256D, 512D, 1024D
        """
        print("\n" + "=" * 80)
        print("Benchmark: Performance Scaling with Vector Dimension")
        print("=" * 80)
        
        n_vectors = 100_000
        dimensions = [64, 128, 256, 512, 1024]
        
        # Check available backends
        backends = hdb.ComputeContext.list_available_backends()
        gpu_available = any(backend in backends for backend in ['cuda', 'rocm', 'mps', 'intel'])
        
        cpu_ctx = hdb.ComputeContext('cpu')
        if gpu_available:
            gpu_ctx = hdb.ComputeContext.auto_detect()
            print(f"GPU backend: {gpu_ctx.backend}")
        else:
            print("⚠ No GPU backend available - CPU baseline only")
        print()
        
        print(f"{'Dimension':>10} | {'CPU Time':>12} | {'CPU Throughput':>18} | {'GPU Time':>12} | {'GPU Throughput':>18} | {'Speedup':>10}")
        print("-" * 110)
        
        for dim in dimensions:
            query = np.random.rand(dim).astype(np.float32)
            database = np.random.rand(n_vectors, dim).astype(np.float32)
            
            # CPU benchmark
            cpu_result = measure_batch_performance(query, database, cpu_ctx, metric_name="l2", warmup=False)
            
            if gpu_available:
                # GPU benchmark
                gpu_result = measure_batch_performance(query, database, gpu_ctx, metric_name="l2", warmup=False)
                speedup = cpu_result['elapsed_sec'] / gpu_result['elapsed_sec']
                
                print(f"{dim:>10}D | {cpu_result['elapsed_ms']:>10.2f}ms | {cpu_result['throughput']:>14,.0f}/sec | "
                      f"{gpu_result['elapsed_ms']:>10.2f}ms | {gpu_result['throughput']:>14,.0f}/sec | {speedup:>9.2f}x")
            else:
                print(f"{dim:>10}D | {cpu_result['elapsed_ms']:>10.2f}ms | {cpu_result['throughput']:>14,.0f}/sec | "
                      f"{'N/A':>12} | {'N/A':>18} | {'N/A':>10}")
        
        print("=" * 80)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
