"""
Unit tests for GPU Context API

Tests Requirements: 2.1, 2.2, 2.3, 2.6
"""
import pytest
import hyperstreamdb as hdb


def test_auto_detect():
    """Test that auto_detect returns a valid ComputeContext"""
    ctx = hdb.ComputeContext.auto_detect()
    assert ctx is not None
    assert isinstance(ctx.backend, str)
    assert ctx.backend in ['cpu', 'cuda', 'rocm', 'mps', 'intel']
    assert isinstance(ctx.device_id, int)
    print(f"Auto-detected backend: {ctx.backend}, device_id: {ctx.device_id}")


def test_cpu_backend_creation():
    """Test creating a CPU backend context"""
    ctx = hdb.ComputeContext('cpu')
    assert ctx.backend == 'cpu'
    assert ctx.device_id == 0  # Default device_id


def test_cpu_backend_with_device_id():
    """Test creating a CPU backend with custom device_id"""
    ctx = hdb.ComputeContext('cpu', device_id=-1)
    assert ctx.backend == 'cpu'
    assert ctx.device_id == -1


def test_backend_property():
    """Test that backend property returns the correct backend name"""
    ctx = hdb.ComputeContext('cpu')
    assert ctx.backend == 'cpu'
    
    # Test with auto_detect
    ctx2 = hdb.ComputeContext.auto_detect()
    backend = ctx2.backend
    assert backend in ['cpu', 'cuda', 'rocm', 'mps', 'intel']


def test_device_id_property():
    """Test that device_id property returns the correct device ID"""
    ctx = hdb.ComputeContext('cpu', device_id=5)
    assert ctx.device_id == 5


def test_list_available_backends():
    """Test that list_available_backends returns a list of backend names"""
    backends = hdb.ComputeContext.list_available_backends()
    assert isinstance(backends, list)
    assert len(backends) > 0
    assert 'cpu' in backends  # CPU should always be available
    
    # All backends should be valid strings
    for backend in backends:
        assert isinstance(backend, str)
        assert backend in ['cpu', 'cuda', 'rocm', 'mps', 'intel']
    
    print(f"Available backends: {backends}")


def test_unavailable_backend_error():
    """Test that requesting an unavailable backend raises RuntimeError"""
    backends = hdb.ComputeContext.list_available_backends()
    
    # Try to create a context with a backend that's not available
    # We'll try all possible backends and expect errors for unavailable ones
    all_backends = ['cuda', 'rocm', 'mps', 'intel']
    unavailable = [b for b in all_backends if b not in backends]
    
    for backend in unavailable:
        with pytest.raises(RuntimeError) as exc_info:
            hdb.ComputeContext(backend)
        
        # Check that error message mentions available backends
        error_msg = str(exc_info.value)
        assert 'not available' in error_msg.lower()
        assert 'available backends' in error_msg.lower()
        print(f"Correctly raised error for unavailable backend '{backend}': {error_msg}")


def test_invalid_backend_error():
    """Test that requesting an invalid backend raises ValueError"""
    with pytest.raises(ValueError) as exc_info:
        hdb.ComputeContext('invalid_backend')
    
    error_msg = str(exc_info.value)
    assert 'unknown backend' in error_msg.lower()
    print(f"Correctly raised error for invalid backend: {error_msg}")


def test_get_stats():
    """Test that get_stats returns a dictionary with performance metrics"""
    ctx = hdb.ComputeContext.auto_detect()
    stats = ctx.get_stats()
    
    assert isinstance(stats, dict)
    assert 'total_kernel_launches' in stats
    assert 'total_gpu_time_ms' in stats
    assert 'total_cpu_time_ms' in stats
    assert 'total_vectors_processed' in stats
    assert 'memory_transfers_mb' in stats
    
    # All values should be numeric
    assert isinstance(stats['total_kernel_launches'], int)
    assert isinstance(stats['total_gpu_time_ms'], float)
    assert isinstance(stats['total_cpu_time_ms'], float)
    assert isinstance(stats['total_vectors_processed'], int)
    assert isinstance(stats['memory_transfers_mb'], float)
    
    # Initial values should be zero
    assert stats['total_kernel_launches'] == 0
    assert stats['total_gpu_time_ms'] == 0.0
    assert stats['total_cpu_time_ms'] == 0.0
    assert stats['total_vectors_processed'] == 0
    assert stats['memory_transfers_mb'] == 0.0
    
    print(f"Stats: {stats}")


def test_reset_stats():
    """Test that reset_stats clears all performance counters"""
    ctx = hdb.ComputeContext.auto_detect()
    
    # Get initial stats
    stats1 = ctx.get_stats()
    assert stats1['total_kernel_launches'] == 0
    
    # Reset stats
    ctx.reset_stats()
    
    # Get stats again
    stats2 = ctx.get_stats()
    assert stats2['total_kernel_launches'] == 0
    assert stats2['total_gpu_time_ms'] == 0.0
    
    print("reset_stats() works correctly")


def test_repr():
    """Test that __repr__ returns a useful string representation"""
    ctx = hdb.ComputeContext('cpu', device_id=0)
    repr_str = repr(ctx)
    
    assert isinstance(repr_str, str)
    assert 'ComputeContext' in repr_str
    assert 'cpu' in repr_str
    assert '0' in repr_str
    
    print(f"repr: {repr_str}")


def test_case_insensitive_backend():
    """Test that backend names are case-insensitive"""
    ctx1 = hdb.ComputeContext('CPU')
    assert ctx1.backend == 'cpu'
    
    ctx2 = hdb.ComputeContext('Cpu')
    assert ctx2.backend == 'cpu'
    
    ctx3 = hdb.ComputeContext('cpu')
    assert ctx3.backend == 'cpu'


if __name__ == "__main__":
    # Run tests
    test_auto_detect()
    test_cpu_backend_creation()
    test_cpu_backend_with_device_id()
    test_backend_property()
    test_device_id_property()
    test_list_available_backends()
    test_unavailable_backend_error()
    test_invalid_backend_error()
    test_get_stats()
    test_reset_stats()
    test_repr()
    test_case_insensitive_backend()
    print("\nAll tests passed!")
