# GPU Setup Guide for HyperStreamDB

This guide provides detailed instructions for setting up GPU acceleration for vector operations in HyperStreamDB.

## Overview

HyperStreamDB supports GPU acceleration for vector distance computations across multiple hardware backends:

- **NVIDIA CUDA** - For NVIDIA GPUs (GeForce, Quadro, Tesla)
- **AMD ROCm** - For AMD Radeon GPUs
- **Apple Metal (MPS)** - For Apple Silicon Macs
- **Intel OpenCL** - For Intel integrated and discrete GPUs

GPU acceleration provides 10x+ speedup for batch distance operations on large vector databases (100,000+ vectors).

## Quick Start

```python
import hyperstreamdb as hdb

# Auto-detect and use best available GPU backend
ctx = hdb.GPUContext.auto_detect()
print(f"Using GPU backend: {ctx.backend}")

# Check what backends are available
print(f"Available backends: {ctx.list_available_backends()}")
```

## NVIDIA CUDA Setup

### Requirements

- **GPU**: NVIDIA GPU with compute capability 6.0 or higher
  - Pascal (GTX 10 series) or newer
  - Recommended: RTX 20/30/40 series, A100, H100
- **Driver**: NVIDIA driver 450.80.02 or later
- **CUDA Toolkit**: Version 11.0 or later (12.x recommended)

### Supported GPUs

| Series | Compute Capability | Supported |
|--------|-------------------|-----------|
| GTX 10 series (Pascal) | 6.0-6.1 | ✅ Yes |
| GTX 16 series (Turing) | 7.5 | ✅ Yes |
| RTX 20 series (Turing) | 7.5 | ✅ Yes |
| RTX 30 series (Ampere) | 8.6 | ✅ Yes |
| RTX 40 series (Ada) | 8.9 | ✅ Yes |
| Tesla V100 | 7.0 | ✅ Yes |
| A100 | 8.0 | ✅ Yes |
| H100 | 9.0 | ✅ Yes |

### Installation on Linux (Ubuntu/Debian)

```bash
# Add NVIDIA package repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install CUDA Toolkit
sudo apt-get install cuda-toolkit-12-3

# Add to PATH (add to ~/.bashrc for persistence)
export PATH=/usr/local/cuda-12.3/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH

# Verify installation
nvidia-smi
nvcc --version
```

### Installation on Windows

1. Download CUDA Toolkit from [NVIDIA website](https://developer.nvidia.com/cuda-downloads)
2. Run the installer (cuda_12.3.0_windows.exe)
3. Follow the installation wizard
4. Verify installation:
   ```cmd
   nvidia-smi
   nvcc --version
   ```

### Verification

```python
import hyperstreamdb as hdb
import numpy as np

# Create CUDA context
ctx = hdb.GPUContext("cuda")
print(f"CUDA backend initialized: {ctx.backend}")

# Test GPU computation
query = np.random.randn(768).astype(np.float32)
database = np.random.randn(10000, 768).astype(np.float32)
distances = hdb.l2_distance_batch(query, database, context=ctx)
print(f"Computed {len(distances)} distances on GPU")

# Check performance stats
stats = ctx.get_stats()
print(f"GPU time: {stats['total_gpu_time_ms']}ms")
```

## AMD ROCm Setup

### Requirements

- **GPU**: AMD Radeon RX 5000 series or newer, or Radeon Instinct
  - Recommended: RX 6000/7000 series, MI100/MI200 series
- **OS**: Linux only (Ubuntu 20.04/22.04, RHEL 8/9, SLES 15)
- **ROCm**: Version 5.0 or later (6.x recommended)

### Supported GPUs

| Series | Architecture | Supported |
|--------|-------------|-----------|
| RX 5000 series | RDNA 1 | ✅ Yes |
| RX 6000 series | RDNA 2 | ✅ Yes |
| RX 7000 series | RDNA 3 | ✅ Yes |
| Radeon VII | GCN 5.1 | ✅ Yes |
| MI100 | CDNA 1 | ✅ Yes |
| MI200 series | CDNA 2 | ✅ Yes |

### Installation on Ubuntu

```bash
# Download and install AMD GPU driver
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_5.7.50700-1_all.deb
sudo apt-get install ./amdgpu-install_5.7.50700-1_all.deb

# Install ROCm
sudo amdgpu-install --usecase=rocm

# Add user to video and render groups
sudo usermod -a -G video,render $USER

# Reboot to apply changes
sudo reboot

# Verify installation
rocm-smi
rocminfo
```

### Installation on RHEL/CentOS

```bash
# Add ROCm repository
sudo tee /etc/yum.repos.d/rocm.repo <<EOF
[ROCm]
name=ROCm
baseurl=https://repo.radeon.com/rocm/rhel8/rpm
enabled=1
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
EOF

# Install ROCm
sudo yum install rocm-hip-sdk

# Add user to video and render groups
sudo usermod -a -G video,render $USER

# Reboot
sudo reboot
```

### Verification

```python
import hyperstreamdb as hdb

# Create ROCm context
ctx = hdb.GPUContext("rocm")
print(f"ROCm backend initialized: {ctx.backend}")

# Test computation
import numpy as np
query = np.random.randn(768).astype(np.float32)
database = np.random.randn(10000, 768).astype(np.float32)
distances = hdb.l2_distance_batch(query, database, context=ctx)
print(f"Computed {len(distances)} distances on AMD GPU")
```

## Apple Metal (MPS) Setup

### Requirements

- **Hardware**: Apple Silicon Mac (M1, M2, M3, or newer)
- **OS**: macOS 12.3 (Monterey) or later
- **No additional installation required** - Metal is built into macOS

### Supported Devices

| Device | Chip | Supported |
|--------|------|-----------|
| MacBook Air (2020+) | M1/M2/M3 | ✅ Yes |
| MacBook Pro (2020+) | M1/M2/M3 Pro/Max/Ultra | ✅ Yes |
| Mac mini (2020+) | M1/M2/M2 Pro | ✅ Yes |
| Mac Studio | M1/M2 Max/Ultra | ✅ Yes |
| iMac (2021+) | M1/M3 | ✅ Yes |
| Mac Pro (2023+) | M2 Ultra | ✅ Yes |

### Verification

```python
import hyperstreamdb as hdb

# Auto-detect should find Metal on Apple Silicon
ctx = hdb.GPUContext.auto_detect()
print(f"Backend: {ctx.backend}")  # Should show "mps"

# Or explicitly create Metal context
ctx = hdb.GPUContext("mps")
print(f"Metal backend initialized")

# Test computation
import numpy as np
query = np.random.randn(768).astype(np.float32)
database = np.random.randn(10000, 768).astype(np.float32)
distances = hdb.l2_distance_batch(query, database, context=ctx)
print(f"Computed {len(distances)} distances on Apple GPU")
```

### Performance Notes

- Apple Silicon uses unified memory (shared between CPU and GPU)
- No explicit memory transfer overhead
- M1 Max/Ultra and M2 Max/Ultra have more GPU cores for better performance
- Recommended for databases up to available system memory

## Intel OpenCL Setup

### Requirements

- **GPU**: Intel Iris Xe or newer (integrated or discrete)
  - Recommended: Arc A-series discrete GPUs
- **Driver**: Intel Graphics Driver with OpenCL support
- **OS**: Linux or Windows

### Supported GPUs

| Series | Architecture | Supported |
|--------|-------------|-----------|
| Iris Xe (11th gen+) | Xe-LP | ✅ Yes |
| Arc A-series | Xe-HPG | ✅ Yes |
| Data Center GPU Max | Xe-HPC | ✅ Yes |

### Installation on Linux (Ubuntu/Debian)

```bash
# Install Intel OpenCL runtime
sudo apt-get update
sudo apt-get install intel-opencl-icd

# Install OpenCL headers (for development)
sudo apt-get install opencl-headers

# Verify installation
clinfo
```

### Installation on Windows

1. Download latest Intel Graphics Driver from [Intel Download Center](https://www.intel.com/content/www/us/en/download-center/home.html)
2. Run the installer
3. OpenCL support is included in modern Intel drivers
4. Verify with `clinfo` (install from [GitHub](https://github.com/Oblomov/clinfo))

### Verification

```python
import hyperstreamdb as hdb

# Create OpenCL context
ctx = hdb.GPUContext("opencl")
print(f"OpenCL backend initialized: {ctx.backend}")

# Test computation
import numpy as np
query = np.random.randn(768).astype(np.float32)
database = np.random.randn(10000, 768).astype(np.float32)
distances = hdb.l2_distance_batch(query, database, context=ctx)
print(f"Computed {len(distances)} distances on Intel GPU")
```

## Multi-GPU Systems

For systems with multiple GPUs, specify the device ID:

```python
import hyperstreamdb as hdb

# List available backends
ctx = hdb.GPUContext.auto_detect()
print(f"Available backends: {ctx.list_available_backends()}")

# Use specific GPU device
ctx = hdb.GPUContext("cuda", device_id=0)  # First GPU
ctx = hdb.GPUContext("cuda", device_id=1)  # Second GPU

# Check which device is being used
print(f"Using device: {ctx.device_id}")
```

## Troubleshooting

### GPU Not Detected

**Symptom:** `auto_detect()` returns CPU backend

**Solutions:**
1. Verify GPU drivers are installed:
   - NVIDIA: `nvidia-smi`
   - AMD: `rocm-smi`
   - Intel: `clinfo`
   - Apple: Check "About This Mac" → ensure Apple Silicon

2. Check backend availability:
   ```python
   ctx = hdb.GPUContext.auto_detect()
   print(ctx.list_available_backends())
   ```

3. Try creating backend explicitly:
   ```python
   try:
       ctx = hdb.GPUContext("cuda")
   except RuntimeError as e:
       print(f"CUDA not available: {e}")
   ```

### Out of Memory Errors

**Symptom:** `RuntimeError: GPU out of memory`

**Solutions:**
1. Process database in chunks:
   ```python
   chunk_size = 10000
   all_distances = []
   for i in range(0, len(database), chunk_size):
       chunk = database[i:i+chunk_size]
       distances = hdb.l2_distance_batch(query, chunk, context=ctx)
       all_distances.append(distances)
   all_distances = np.concatenate(all_distances)
   ```

2. Use smaller data types (float32 instead of float64)
3. Use sparse vectors for sparse data
4. Use binary vectors for binary features

### Slow Performance

**Symptom:** GPU is slower than CPU

**Possible causes:**
1. **Small batch size**: GPU overhead dominates for < 1,000 vectors
   - Solution: Use CPU for small batches, GPU for large batches

2. **Data type mismatch**: Using float64 instead of float32
   - Solution: Convert to float32: `data.astype(np.float32)`

3. **Memory transfer overhead**: Creating new context each time
   - Solution: Reuse GPU context across multiple operations

4. **Wrong backend**: Using OpenCL on NVIDIA GPU
   - Solution: Use CUDA for NVIDIA, ROCm for AMD

### Driver Version Mismatch

**Symptom:** `RuntimeError: CUDA driver version is insufficient`

**Solution:**
```bash
# Check current driver version
nvidia-smi

# Update NVIDIA driver (Ubuntu)
sudo apt-get update
sudo apt-get install --only-upgrade nvidia-driver-535

# Or install latest driver
sudo ubuntu-drivers autoinstall
```

## Performance Benchmarks

Expected speedups for batch operations (100,000 vectors, 768 dimensions):

| Backend | Hardware | Speedup vs CPU |
|---------|----------|----------------|
| CUDA | RTX 3090 | 15-20x |
| CUDA | RTX 4090 | 20-30x |
| CUDA | A100 | 25-35x |
| ROCm | RX 7900 XTX | 12-18x |
| Metal | M1 Max | 8-12x |
| Metal | M2 Ultra | 15-20x |
| OpenCL | Arc A770 | 6-10x |

*Benchmarks measured with float32 data, L2 distance metric*

## Best Practices

1. **Reuse GPU context** across multiple operations
2. **Use float32** instead of float64 for better GPU performance
3. **Batch operations** when possible (process multiple queries together)
4. **Profile your workload** using `ctx.get_stats()`
5. **Choose appropriate backend** for your hardware
6. **Monitor GPU memory** usage for large databases
7. **Use sparse/binary vectors** when applicable to reduce memory

## See Also

- [Python Vector API Documentation](PYTHON_VECTOR_API.md) - Complete API reference
- [Vector Configuration Guide](VECTOR_CONFIGURATION.md) - Index tuning
- [Benchmarking Guide](BENCHMARKING.md) - Performance testing
