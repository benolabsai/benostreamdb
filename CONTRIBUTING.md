# Contributing to HyperStreamDB

Thank you for your interest in contributing! HyperStreamDB is a serverless index-streaming database built in Rust with Python and Java bindings. This document outlines how to set up your development environment and the conventions we follow.

## Project Principles

Before contributing, please read [STEERING.md](STEERING.md) for our development philosophy. It covers our approach to code quality, design, and release readiness based on *The Pragmatic Programmer*.

---

## Development Setup

### Prerequisites

- **Rust** (stable, 1.80+) — install via [rustup](https://rustup.rs/)
- **Python** (3.10–3.14) — for binding development and testing
- **Docker** — for integration tests against MinIO and Nessie
- **Cargo** and **maturin** — for building Python wheels

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install maturin (Python-Rust build tool)
pip install maturin

# Clone and build
git clone https://github.com/rla3rd/hyperstreamdb.git
cd hyperstreamdb
cargo build --release
```

### Python Bindings

```bash
# Develop Python bindings locally
maturin develop

# Run Python tests
pytest tests/
```

### Docker Services

```bash
# Start MinIO (S3-compatible storage) and Nessie (Iceberg catalog)
docker compose -f docker-compose-minio-nessie.yml up -d

# Run integration tests
pytest tests/integration/
```

### GPU Acceleration (Optional)

GPU features are optional and feature-gated. To build with GPU support:

```bash
# With CUDA (NVIDIA)
cargo build --features cuda

# With WGPU (Vulkan/Metal/ROCm)
cargo build --features wgpu
```

See the [README](README.md#gpu-acceleration-optional) for hardware-specific installation guides.

---

## Coding Standards

### Rust

- **Formatting**: Run `cargo fmt` before committing. We use `rustfmt` defaults.
- **Linting**: Run `cargo clippy --all-features` and resolve all warnings. Treat clippy warnings as errors.
- **Naming**:
  - Modules: `snake_case`
  - Types/Structs: `PascalCase`
  - Functions/methods: `snake_case`
  - Constants: `SCREAMING_SNAKE_CASE`
- **Error handling**: Use `Result<T, HyperstreamError>` throughout. Never use `.unwrap()` in library code — prefer `.expect()` with a descriptive message or proper error propagation.
- **Documentation**: All `pub` items must have rustdoc comments. Run `cargo doc --no-deps` to verify.

### Python

- Follow [PEP 8](https://peps.python.org/pep-0008/) for style
- Use type hints on all public functions and methods
- Docstrings should follow [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)

### Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/):

```
<type>(<scope>): <description>

[optional body]
```

**Types:**
| Type       | Description                                      |
|------------|--------------------------------------------------|
| `feat`     | New feature or functionality                     |
| `fix`      | Bug fix                                          |
| `perf`     | Performance improvement                          |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `docs`     | Documentation only changes                       |
| `test`     | Adding or correcting tests                       |
| `chore`    | Maintenance, CI, build, dependency updates       |
| `ci`       | Continuous Integration configuration changes     |

**Examples:**
```
feat(indexing): stabilize HNSW SIMD indexing and remove legacy simdeez dependency
fix: resolve rustdoc warnings and pyo3 API compatibility
refactor(gpu): implement hardware-agnostic compute dispatch with thread-local context
docs: switch to GitHub Actions for Pages deployment
test: add comprehensive unit test suites for Spark and Trino connectors
chore: bump version to v0.4.0
```

---

## Branch Naming Conventions

```
feat/<feature-name>        # New features
fix/<issue-or-bug>         # Bug fixes
refactor/<component>       # Refactoring work
docs/<topic>               # Documentation changes
test/<scope>               # Test additions or changes
chore/<task>               # Maintenance tasks
```

**Examples:**
- `feat/tiered-manifests`
- `fix/hybrid-search-fallback`
- `refactor/gpu-context`
- `test/spark-connector`

---

## Pull Request Process

### 1. Create a Draft PR

Start with a **Draft PR** while work is in progress. This allows early feedback without triggering full CI review.

### 2. Self-Review Checklist

Before marking your PR as **Ready for Review**:

- [ ] Code passes `cargo fmt` and `cargo clippy` with zero warnings
- [ ] All tests pass (`cargo test` and `pytest tests/`)
- [ ] New features include corresponding unit and/or integration tests
- [ ] Public API changes have rustdoc documentation
- [ ] `CHANGELOG.md` is updated (if this is a user-facing change)
- [ ] No broken windows: compiler warnings, failing tests, or suboptimal patterns

### 3. CI Validation

The CI pipeline runs:
- **Rust checks**: `cargo check`, `cargo clippy`, `cargo test`
- **Python tests**: `pytest`
- **Documentation build**: `cargo doc --no-deps`
- **Integration tests**: against MinIO and Nessie (when applicable)

All CI checks must pass before merge.

### 4. Review and Merge

- PRs require at least one approval from a maintainer
- Address all review comments before merging
- Squash-merge to keep history clean

---

## Testing Requirements

### Unit Tests

Every new feature or bug fix must include unit tests. Use `#[cfg(test)]` modules within the relevant source file:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_behavior() {
        // ...
    }
}
```

### Integration Tests

For features that span multiple modules or involve external systems (storage, catalogs), add integration tests under `tests/integration/`:

```bash
# Run all integration tests
pytest tests/integration/

# Run specific test
pytest tests/integration/test_hybrid_search.py -v
```

### Benchmarks

Performance-critical changes should include benchmark updates:

```bash
# Run Criterion benchmarks
cargo bench

# Quick benchmark suite
python benchmarks/benchmark_suite.py --quick
```

---

## Building Connectors

The Spark and Trino connectors require a separate build step:

```bash
# Build all connector variants
./build-connectors.sh

# Build with CUDA support
./build-connectors.sh --cuda
```

---

## Release Process

Releases are tagged and published by the maintainer. The process involves:

1. Bump version in `Cargo.toml` (single source of truth)
2. Update `CHANGELOG.md` with release date
3. Create git tag: `git tag v0.X.Y`
4. Push tag: `git push origin v0.X.Y`
5. CI automatically publishes to crates.io and PyPI

---

## Questions or Issues?

- Open a [GitHub Issue](https://github.com/rla3rd/hyperstreamdb/issues) for bugs or feature requests
- For discussions, use the Discussions tab or reach out directly

---

**Stay Pragmatic. Stay Antigravity.**
