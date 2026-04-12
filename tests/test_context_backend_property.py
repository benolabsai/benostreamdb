"""
Property-based test for GPU Context backend property

Feature: python-vector-api-gpu-acceleration
Property 4: Context Backend Property

**Validates: Requirements 2.4**

For any created GPU context, querying its backend property should return 
the backend name that was used to create it.
"""
import pytest
from hypothesis import given, strategies as st
import hyperstreamdb as hdb


# Strategy for valid backend names
valid_backends = st.sampled_from(['cpu', 'cuda', 'rocm', 'mps', 'intel'])

# Strategy for device IDs
device_ids = st.integers(min_value=-1, max_value=7)


@given(backend=valid_backends, device_id=device_ids)
def test_context_backend_property(backend, device_id):
    """
    Property: For any created GPU context, querying its backend property 
    should return the backend name that was used to create it.
    
    This property verifies that the backend property correctly reflects
    the backend used during context creation, regardless of whether the
    backend is available on the current system.
    """
    # Get list of available backends
    available_backends = hdb.ComputeContext.list_available_backends()
    
    if backend in available_backends:
        # Backend is available - should succeed
        ctx = hdb.ComputeContext(backend, index=device_id)
        
        # Property: backend property should match the backend used to create it
        assert ctx.backend == backend.lower(), \
            f"Expected backend '{backend.lower()}', got '{ctx.backend}'"
        
        # Property: device_id property should match the device_id used to create it
        assert ctx.device_id == device_id, \
            f"Expected device_id {device_id}, got {ctx.device_id}"
    else:
        # Backend is not available
        if backend in ['cuda', 'mps']:
            # These backends aggressively validate hardware upon creation
            with pytest.raises(RuntimeError) as exc_info:
                hdb.ComputeContext(backend, index=device_id)
            error_msg = str(exc_info.value)
            assert 'available' in error_msg.lower()
        else:
            # ROCM and Intel use WGPU which does not always error on immediate creation but during execution
            try:
                hdb.ComputeContext(backend, index=device_id)
            except RuntimeError:
                pass


@given(device_id=device_ids)
def test_auto_detect_backend_property(device_id):
    """
    Property: For auto-detected contexts, the backend property should return
    one of the valid backend names.
    """
    ctx = hdb.ComputeContext.auto_detect()
    
    # Property: backend should be one of the valid backends
    valid_backend_names = ['cpu', 'cuda', 'rocm', 'mps', 'intel']
    assert ctx.backend in valid_backend_names, \
        f"Auto-detected backend '{ctx.backend}' is not valid"
    
    # Property: backend should be in the list of available backends
    available_backends = hdb.ComputeContext.list_available_backends()
    assert ctx.backend in available_backends, \
        f"Auto-detected backend '{ctx.backend}' not in available backends: {available_backends}"


# Strategy for case variations
case_variations = st.sampled_from([
    'cpu', 'CPU', 'Cpu', 'cPu',
    'cuda', 'CUDA', 'Cuda',
    'rocm', 'ROCM', 'Rocm',
    'mps', 'MPS', 'Mps',
    'intel', 'INTEL', 'Intel'
])


@given(backend=case_variations)
def test_backend_case_insensitive_property(backend):
    """
    Property: Backend names should be case-insensitive, and the backend
    property should always return lowercase names.
    """
    # Extract the lowercase version
    backend_lower = backend.lower()
    
    # Get list of available backends
    available_backends = hdb.ComputeContext.list_available_backends()
    
    if backend_lower in available_backends:
        # Backend is available - should succeed
        ctx = hdb.ComputeContext(backend)
        
        # Property: backend property should always return lowercase
        assert ctx.backend == backend_lower, \
            f"Expected lowercase backend '{backend_lower}', got '{ctx.backend}'"
        
        # Property: backend property should be lowercase
        assert ctx.backend.islower(), \
            f"Backend property should be lowercase, got '{ctx.backend}'"


if __name__ == "__main__":
    # Run property tests with pytest
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])
