/// Returns `true` if GPU tests should be skipped.
///
/// Checks (in order):
/// 1. The `SKIP_GPU_TESTS` environment variable — if set, always skip.
/// 2. Runtime CUDA availability — attempts to create a CUDA backend via
///    `ComputeContext::from_backend(ComputeBackend::Cuda)`.  If that fails
///    (feature not compiled in, no driver, no device) the test is skipped.
pub fn should_skip_gpu_tests() -> bool {
    if std::env::var("SKIP_GPU_TESTS").is_ok() {
        return true;
    }

    // Probe for a working CUDA device at runtime
    use hyperstreamdb::core::index::gpu::{ComputeBackend, ComputeContext};
    ComputeContext::from_backend(ComputeBackend::Cuda).is_err()
}
