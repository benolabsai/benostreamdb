import hyperstreamdb as hdb
import numpy as np
import time

def test_mps_gpu():
    print("="*50)
    print("HYPERSTREAMDB MPS GPU VALIDATION")
    print("="*50)
    
    # 1. Detect GPU Context
    print("\n[1] Detecting GPU Context...")
    try:
        ctx = hdb.ComputeContext.auto_detect()
        print(f"Detected backend: {ctx.backend}")
        print(f"Device ID: {ctx.device_id}")
        
        if ctx.backend != "mps":
            print(f"WARNING: Expected 'mps' backend on macOS, but got '{ctx.backend}'")
    except Exception as e:
        print(f"Error detecting GPU context: {e}")
        return

    # 2. Prepare Test Data
    print("\n[2] Preparing Test Data...")
    n_vectors = 100_000
    dim = 768
    print(f"Generating {n_vectors:,} vectors of {dim} dimensions...")
    
    query = np.random.randn(dim).astype(np.float32)
    vectors = np.random.randn(n_vectors, dim).astype(np.float32)
    
    # 3. Compute on CPU (Baseline)
    print("\n[3] Computing distances on CPU...")
    cpu_ctx = hdb.ComputeContext("cpu")
    start_time = time.time()
    cpu_distances = hdb.l2_batch(query, vectors, device=cpu_ctx)
    cpu_time = time.time() - start_time
    print(f"CPU Time: {cpu_time*1000:.2f}ms")

    # 4. Compute on GPU (MPS)
    print("\n[4] Computing distances on GPU (MPS)...")
    # Reset stats before measuring
    ctx.reset_stats()
    
    start_time = time.time()
    gpu_distances = hdb.l2_batch(query, vectors, device=ctx)
    gpu_time = time.time() - start_time
    
    stats = ctx.get_stats()
    print(f"GPU Time (Total): {gpu_time*1000:.2f}ms")
    print(f"GPU Kernel Time: {stats['total_gpu_time_ms']:.2f}ms")
    print(f"Vectors Processed: {stats['total_vectors_processed']}")
    print(f"Speedup: {cpu_time / gpu_time:.2f}x")

    # 5. Verify Correctness
    print("\n[5] Verifying Correctness...")
    max_diff = np.max(np.abs(np.array(cpu_distances) - np.array(gpu_distances)))
    print(f"Max difference between CPU and GPU results: {max_diff:.2e}")
    
    if max_diff < 1e-4:
        print("✅ SUCCESS: GPU results match CPU results!")
    else:
        print("❌ FAILURE: Significant difference detected!")

if __name__ == "__main__":
    test_mps_gpu()
