# Getting Started with Real-World Testing

This guide will walk you through running your first HyperStreamDB benchmarks.

## Prerequisites

```bash
# 1. Build the Rust library
cargo build --release

# 2. Install Python bindings
pip install maturin
maturin develop

# 3. Install test dependencies
pip install pyarrow pandas numpy
```

## Quick Test: Synthetic Data

Let's start with a small synthetic dataset to verify everything works:

```bash
# Run the benchmark suite
cargo bench
```

This will test:
- Ingest throughput (1K, 10K, 100K rows)
- Query latency with indexes
- Vector search performance

## Real-World Test 1: NYC Taxi (Optional - 200GB download)

**Warning:** This downloads ~200GB of data. Only run if you have space and bandwidth.

```bash
# Download NYC Taxi data
./tests/data/download_nyc_taxi.sh

# Run integration test
python tests/integration/test_nyc_taxi.py
```

**Expected results:**
- Ingest: >100K rows/sec
- Query: <100ms
- Compaction: <5min per 10GB

## Real-World Test 2: Vector Embeddings

Generate 10M synthetic embeddings (simulates BERT):

```bash
# Generate embeddings (~7GB)
python tests/data/generate_embeddings.py

# Run vector search benchmark
python tests/benchmarks/vector_search/test_parallel_search.py
```

## Viewing Results

Benchmark results are saved to:
```
target/criterion/
├── ingest/
│   └── report/index.html
├── query_indexed/
│   └── report/index.html
└── vector_search/
    └── report/index.html
```

Open in browser:
```bash
open target/criterion/report/index.html
```

## Next Steps

1. ✅ Run synthetic benchmarks
2. ✅ Review performance results
3. ✅ Identify bottlenecks
4. 🔄 Optimize and re-run
5. 🔄 Run NYC Taxi test (if desired)

## Troubleshooting

**Error: "maturin: command not found"**
```bash
pip install maturin
```

**Error: "cannot find -lpython3.x"**
```bash
# Install Python dev headers
sudo apt-get install python3-dev  # Ubuntu/Debian
brew install python@3.11           # macOS
```

**Error: "failed to compile hyperstreamdb"**
```bash
# Check Rust version
rustc --version  # Should be 1.80+
rustup update
```

## Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| Ingest | >100K rows/sec | >10K rows/sec (CPU) |
| Query (indexed) | <100ms p99 | ⏱️ In progress |
| Vector search | <50ms (k=10) | 819ms (100K, 768D, CPU) |
| Compaction | <5min/10GB | ⏱️ In progress |

Fill in "Current" after running benchmarks!
