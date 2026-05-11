/// Returns `true` if GPU tests should be skipped (e.g., on CI without a real GPU).
/// Controlled by the `SKIP_GPU_TESTS` environment variable.
pub fn should_skip_gpu_tests() -> bool {
    std::env::var("SKIP_GPU_TESTS").is_ok()
}
