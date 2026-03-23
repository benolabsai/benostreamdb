# Python Vector Distance API Reference

## Overview

HyperStreamDB provides a comprehensive Python API for vector distance computations with GPU acceleration support across multiple hardware backends. This API allows you to compute distances between vectors directly from Python without writing SQL queries, with optional GPU acceleration for high-performance batch operations.

## Supported Distance Metrics

The API supports six distance metrics:

| Metric | Function | Description | Use Case |
|--------|----------|-------------|----------|
| **L2 (Euclidean)** | `l2_distance()` | √Σ(a-b)² | General-purpose similarity |
| **Cosine** | `cosine_distance()` | 1 - (a·b)/(‖a‖‖b‖) | Text embeddings, normalized vectors |
| **Inner Product** | `inner_product()` | -a·b | Maximum similarity search |
| **L1 (Manhattan)** | `l1_distance()` | Σ\|a-b\| | Robust to outliers |
| **Hamming** | `hamming_distance()` | Count of differing bits | Binary vectors, hashing |
| **Jaccard** | `jaccard_distance()` | 1 - \|A∩B\|/\|A∪B\| | Set similarity |

## GPU Backend Support

### Supported Hardware

| Backend | Hardware | Platform | Status |
|---------|----------|----------|--------|
| **CUDA** | NVIDIA GPUs | Linux, Windows | ✅ Supported |
| **ROCm** | AMD GPUs | Linux | ✅ Supported |
| **Metal (MPS)** | Apple Silicon | macOS | ✅ Supported |
| **OpenCL** | Intel GPUs | Linux, Windows | ✅ Supported |
| **CPU** | All platforms | Fallback | ✅ Always available |

### Backend Priority

When using `GPUContext.auto_detect()`, backends are selected in this priority order:
1. CUDA (NVIDIA)
2. Metal/MPS (Apple Silicon)
3. ROCm (AMD)
4. OpenCL (Intel)
5. CPU (fallback)

## Installation

### Basic Installation

```bash
pip install hyperstreamdb
```

### GPU Backend Requirements

#### NVIDIA CUDA

**Requirements:**
- NVIDIA GPU with compute capability 6.0+ (Pascal or newer)
- CUDA Toolkit 11.0 or later
- NVIDIA driver 450.80.02 or later

**Installation (Linux):**
```bash
# Ubuntu/Debian
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-12-3

# Verify installation
nvidia-smi
nvcc --version
```

**Installation (Windows):**
1. Download CUDA Toolkit from [NVIDIA website](https://developer.nvidia.com/cuda-downloads)
2. Run installer and follow prompts
3. Verify with `nvidia-smi` in Command Prompt

#### AMD ROCm

**Requirements:**
- AMD GPU (Radeon RX 5000 series or newer, or Radeon Instinct)
- ROCm 5.0 or later
- Linux only (Ubuntu 20.04/22.04, RHEL 8/9)

**Installation (Ubuntu):**
```bash
# Add ROCm repository
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_5.7.50700-1_all.deb
sudo apt-get install ./amdgpu-install_5.7.50700-1_all.deb

# Install ROCm
sudo amdgpu-install --usecase=rocm

# Add user to video and render groups
sudo usermod -a -G video,render $USER

# Verify installation
rocm-smi
```

#### Apple Metal (MPS)

**Requirements:**
- Apple Silicon Mac (M1, M2, M3, or newer)
- macOS 12.3 or later
- No additional installation required

**Verification:**
```python
import hyperstreamdb as hdb
ctx = hdb.GPUContext.auto_detect()
print(ctx.backend)  # Should show "mps" on Apple Silicon
```

#### Intel OpenCL

**Requirements:**
- Intel GPU (Iris Xe or newer recommended)
- Intel Graphics Driver with OpenCL support
- Linux or Windows

**Installation (Linux):**
```bash
# Ubuntu/Debian
sudo apt-get install intel-opencl-icd

# Verify installation
clinfo
```

**Installation (Windows):**
- Install latest Intel Graphics Driver from [Intel website](https://www.intel.com/content/www/us/en/download-center/home.html)
- OpenCL support is included in modern drivers

## Quick Start

### Basic Distance Computation

```python
import hyperstreamdb as hdb
import numpy as np

# Create two vectors
vec1 = np.array([1.0, 2.0, 3.0])
vec2 = np.array([4.0, 5.0, 6.0])

# Compute L2 distance
distance = hdb.l2_distance(vec1, vec2)
print(f"L2 distance: {distance}")

# Compute cosine distance
distance = hdb.cosine_distance(vec1, vec2)
print(f"Cosine distance: {distance}")
```

### GPU-Accelerated Batch Operations

```python
import hyperstreamdb as hdb
import numpy as np

# Create GPU context (auto-detect best backend)
ctx = hdb.GPUContext.auto_detect()
print(f"Using backend: {ctx.backend}")

# Create query vector and database
query = np.random.randn(768).astype(np.float32)
database = np.random.randn(100000, 768).astype(np.float32)

# Compute distances on GPU (10x+ faster for large databases)
distances = hdb.l2_distance_batch(query, database, context=ctx)

# Find top-k nearest neighbors
k = 10
top_k_indices = np.argsort(distances)[:k]
top_k_distances = distances[top_k_indices]

print(f"Top {k} nearest neighbors:")
for idx, dist in zip(top_k_indices, top_k_distances):
    print(f"  Index {idx}: distance {dist:.4f}")
```

### Sparse Vector Operations

```python
import hyperstreamdb as hdb
import numpy as np

# Create sparse vectors (only store non-zero elements)
# Useful for high-dimensional sparse data (e.g., TF-IDF, bag-of-words)
sparse1 = hdb.SparseVector(
    indices=np.array([0, 5, 100, 500], dtype=np.int32),
    values=np.array([1.0, 2.5, 0.8, 3.2], dtype=np.float32),
    dim=1000
)

sparse2 = hdb.SparseVector(
    indices=np.array([5, 50, 100, 600], dtype=np.int32),
    values=np.array([2.0, 1.5, 0.9, 2.1], dtype=np.float32),
    dim=1000
)

# Compute sparse distance (only processes non-zero elements)
distance = hdb.l2_distance_sparse(sparse1, sparse2)
print(f"Sparse L2 distance: {distance}")

# Convert to dense if needed
dense1 = sparse1.to_dense()
```

### Binary Vector Operations

```python
import hyperstreamdb as hdb
import numpy as np

# Binary vectors for efficient similarity search
# Each bit represents a feature (e.g., SimHash, LSH)

# Create bit-packed binary vectors (8 bits per byte)
binary1 = np.packbits(np.random.randint(0, 2, 128))  # 128 bits = 16 bytes
binary2 = np.packbits(np.random.randint(0, 2, 128))

# Compute Hamming distance (counts differing bits)
distance = hdb.hamming_distance_packed(binary1, binary2)
print(f"Hamming distance: {distance} bits differ")

# Compute Jaccard distance for binary vectors
distance = hdb.jaccard_distance_packed(binary1, binary2)
print(f"Jaccard distance: {distance}")

# Auto-packing: provide unpacked binary vectors (0/1 values)
# The API will automatically pack them for efficiency
unpacked1 = np.random.randint(0, 2, 128, dtype=np.uint8)
unpacked2 = np.random.randint(0, 2, 128, dtype=np.uint8)
distance = hdb.hamming_distance(unpacked1, unpacked2)  # Auto-packed internally
```

## API Reference

### Distance Functions

#### Single-Pair Distance Functions

All single-pair functions accept two vectors and return a scalar distance:

```python
l2_distance(vec1, vec2, context=None) -> float
cosine_distance(vec1, vec2, context=None) -> float
inner_product(vec1, vec2, context=None) -> float
l1_distance(vec1, vec2, context=None) -> float
hamming_distance(vec1, vec2, context=None) -> float
jaccard_distance(vec1, vec2, context=None) -> float
```

**Parameters:**
- `vec1`, `vec2`: NumPy arrays, Python lists, or any array-like objects
- `context` (optional): `GPUContext` for GPU acceleration

**Returns:** `float` - The computed distance

**Raises:**
- `ValueError`: If vectors have different dimensions
- `ValueError`: If vectors contain NaN or infinite values
- `TypeError`: If input types are invalid

#### Batch Distance Functions

Compute distances between one query vector and multiple database vectors:

```python
l2_distance_batch(query, database, context=None) -> np.ndarray
cosine_distance_batch(query, database, context=None) -> np.ndarray
inner_product_batch(query, database, context=None) -> np.ndarray
l1_distance_batch(query, database, context=None) -> np.ndarray
hamming_distance_batch(query, database, context=None) -> np.ndarray
jaccard_distance_batch(query, database, context=None) -> np.ndarray
```

**Parameters:**
- `query`: 1D NumPy array (shape: `[dim]`)
- `database`: 2D NumPy array (shape: `[n_vectors, dim]`)
- `context` (optional): `GPUContext` for GPU acceleration

**Returns:** `np.ndarray` - 1D array of distances (shape: `[n_vectors]`)

**Performance:** GPU acceleration provides 10x+ speedup for databases with 100,000+ vectors

#### Sparse Distance Functions

Efficient distance computation for sparse vectors:

```python
l2_distance_sparse(sparse1, sparse2, context=None) -> float
cosine_distance_sparse(sparse1, sparse2, context=None) -> float
inner_product_sparse(sparse1, sparse2, context=None) -> float
```

**Parameters:**
- `sparse1`, `sparse2`: `SparseVector` objects
- `context` (optional): `GPUContext` for GPU acceleration

**Returns:** `float` - The computed distance

#### Binary Distance Functions

Efficient distance computation for bit-packed binary vectors:

```python
hamming_distance_packed(binary1, binary2, context=None) -> int
jaccard_distance_packed(binary1, binary2, context=None) -> float
```

**Parameters:**
- `binary1`, `binary2`: NumPy uint8 arrays (bit-packed)
- `context` (optional): `GPUContext` for GPU acceleration

**Returns:** Distance value (int for Hamming, float for Jaccard)

### GPU Context Management

#### GPUContext Class

```python
class GPUContext:
    """Manages GPU backend selection and device configuration."""
    
    @staticmethod
    def auto_detect() -> GPUContext:
        """Detect and return the highest-priority available GPU backend."""
    
    def __init__(self, backend: str, device_id: int = 0):
        """
        Create GPU context with specific backend.
        
        Args:
            backend: Backend name ("cuda", "rocm", "mps", "opencl", "cpu")
            device_id: GPU device ID for multi-GPU systems (default: 0)
        
        Raises:
            RuntimeError: If backend is not available
        """
    
    @property
    def backend(self) -> str:
        """Get current backend name."""
    
    @property
    def device_id(self) -> int:
        """Get current device ID."""
    
    def list_available_backends(self) -> list[str]:
        """List all detected GPU backends."""
    
    def get_stats(self) -> dict:
        """
        Get performance statistics.
        
        Returns:
            dict with keys:
                - total_gpu_time_ms: Total GPU computation time
                - kernel_launches: Number of GPU kernel launches
                - vectors_processed: Total vectors processed
        """
    
    def reset_stats(self):
        """Clear all accumulated performance metrics."""
```

#### Usage Examples

```python
# Auto-detect best backend
ctx = hdb.GPUContext.auto_detect()

# Create specific backend
ctx = hdb.GPUContext("cuda", device_id=0)

# List available backends
backends = ctx.list_available_backends()
print(f"Available backends: {backends}")

# Monitor performance
distances = hdb.l2_distance_batch(query, database, context=ctx)
stats = ctx.get_stats()
print(f"GPU time: {stats['total_gpu_time_ms']}ms")
print(f"Kernel launches: {stats['kernel_launches']}")

# Reset stats for next benchmark
ctx.reset_stats()
```

### Sparse Vector Class

```python
class SparseVector:
    """Sparse vector representation storing only non-zero elements."""
    
    def __init__(self, indices: np.ndarray, values: np.ndarray, dim: int):
        """
        Create sparse vector.
        
        Args:
            indices: 1D int32 array of non-zero indices (must be sorted)
            values: 1D float32 array of non-zero values
            dim: Total dimension of the vector
        
        Raises:
            ValueError: If indices are not sorted or out of bounds
        """
    
    @property
    def indices(self) -> np.ndarray:
        """Get non-zero indices."""
    
    @property
    def values(self) -> np.ndarray:
        """Get non-zero values."""
    
    @property
    def dim(self) -> int:
        """Get total dimension."""
    
    def to_dense(self) -> np.ndarray:
        """Convert to dense NumPy array."""
```

## Performance Optimization

### When to Use GPU Acceleration

GPU acceleration provides significant speedups for:
- **Batch operations** with 10,000+ vectors
- **High-dimensional vectors** (512+ dimensions)
- **Repeated queries** on the same database

For small operations (< 1,000 vectors), CPU may be faster due to GPU transfer overhead.

### Memory Considerations

GPU memory limits:
- **NVIDIA RTX 3090**: 24 GB
- **Apple M1 Max**: 32-64 GB (shared)
- **AMD RX 7900 XTX**: 24 GB

For large databases, consider:
1. **Batch processing**: Process database in chunks
2. **Sparse vectors**: Reduce memory usage for sparse data
3. **Binary vectors**: Use bit-packing for binary features

### Best Practices

```python
# ✅ Good: Reuse GPU context
ctx = hdb.GPUContext.auto_detect()
for query in queries:
    distances = hdb.l2_distance_batch(query, database, context=ctx)

# ❌ Bad: Create new context each time
for query in queries:
    ctx = hdb.GPUContext.auto_detect()  # Overhead!
    distances = hdb.l2_distance_batch(query, database, context=ctx)

# ✅ Good: Use appropriate data types
query = np.array(data, dtype=np.float32)  # float32 is faster on GPU

# ❌ Bad: Use float64 unnecessarily
query = np.array(data, dtype=np.float64)  # Slower and uses more memory
```

## Integration with SQL

The Python API shares the same GPU context with SQL queries:

```python
import hyperstreamdb as hdb

# Set global GPU context
ctx = hdb.GPUContext.auto_detect()
hdb.set_global_gpu_context(ctx)

# SQL queries now use GPU acceleration
session = hdb.Session()
session.register("documents", table)

results = session.sql("""
    SELECT id, content,
           embedding <-> '[0.1, 0.2, 0.3]'::vector AS distance
    FROM documents
    ORDER BY distance
    LIMIT 10
""")

# Check GPU usage
stats = ctx.get_stats()
print(f"GPU time: {stats['total_gpu_time_ms']}ms")
```

## Troubleshooting

### GPU Not Detected

```python
ctx = hdb.GPUContext.auto_detect()
print(ctx.backend)  # Shows "cpu" instead of GPU backend
```

**Solutions:**
1. Verify GPU drivers are installed: `nvidia-smi`, `rocm-smi`, or check System Preferences on macOS
2. Check backend availability: `ctx.list_available_backends()`
3. Verify CUDA/ROCm installation: `nvcc --version` or `rocminfo`

### Out of Memory Errors

```python
# Error: RuntimeError: GPU out of memory
distances = hdb.l2_distance_batch(query, huge_database, context=ctx)
```

**Solutions:**
```python
# Process in chunks
chunk_size = 10000
all_distances = []
for i in range(0, len(database), chunk_size):
    chunk = database[i:i+chunk_size]
    distances = hdb.l2_distance_batch(query, chunk, context=ctx)
    all_distances.append(distances)
all_distances = np.concatenate(all_distances)
```

### Dimension Mismatch

```python
# Error: ValueError: Vector dimensions must match
distance = hdb.l2_distance(vec1, vec2)
```

**Solutions:**
```python
# Check dimensions
print(f"vec1 shape: {vec1.shape}, vec2 shape: {vec2.shape}")

# Ensure same dimension
assert vec1.shape == vec2.shape
```

## Examples

See [examples/python_distance_api_examples.py](../examples/python_distance_api_examples.py) for complete working examples.

## See Also

- [pgvector SQL Guide](PGVECTOR_SQL_GUIDE.md) - SQL syntax for vector operations
- [Vector Configuration](VECTOR_CONFIGURATION.md) - Index configuration and tuning
- [Benchmarking Guide](BENCHMARKING.md) - Performance testing and optimization
