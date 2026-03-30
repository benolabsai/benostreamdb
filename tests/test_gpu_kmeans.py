import hyperstreamdb as hdb
import numpy as np
import time

def benchmark_kmeans_direct():
    print("="*50)
    print("HYPERSTREAMDB GPU K-MEANS DIRECT BENCHMARK")
    print("="*50)

    # 1. Setup Data
    n_vectors = 100_000 
    dim = 256
    k = 100 # Use more clusters to see GPU advantage
    
    print(f"\n[1] Generating {n_vectors:,} vectors with {dim} dimensions...")
    vectors = np.random.randn(n_vectors, dim).astype(np.float32).tolist()

    # 2. CPU Benchmark
    print(f"\n[2] Running K-Means (k={k}) on CPU...")
    # By default, without activating a context, it uses CPU
    ctx_cpu = hdb.ComputeContext("cpu")
    
    start_time = time.time()
    _, _ = ctx_cpu.kmeans(vectors, k, max_iters=5)
    cpu_time = time.time() - start_time
    print(f"CPU K-Means Time: {cpu_time:.4f}s")

    # 3. GPU Benchmark (MPS)
    print(f"\n[3] Running K-Means (k={k}) on GPU (MPS)...")
    try:
        ctx_gpu = hdb.ComputeContext.auto_detect()
        print(f"Detected backend: {ctx_gpu.backend}")
        
        # Warm up
        print("Warming up GPU...")
        _, _ = ctx_gpu.kmeans(vectors[:1000], k, max_iters=1)
        
        start_time = time.time()
        _, _ = ctx_gpu.kmeans(vectors, k, max_iters=5)
        gpu_time = time.time() - start_time
        print(f"GPU K-Means Time: {gpu_time:.4f}s")
        if gpu_time > 0:
            print(f"Speedup: {cpu_time / gpu_time:.2f}x")
            
        # Check stats
        stats = ctx_gpu.get_stats()
        print(f"\nGPU Stats: {stats}")
        
    except Exception as e:
        print(f"GPU Error: {e}")

if __name__ == "__main__":
    benchmark_kmeans_direct()
