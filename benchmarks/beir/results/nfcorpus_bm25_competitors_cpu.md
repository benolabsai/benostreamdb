# BEIR Lexical / BM25 Benchmark Results

- **Dataset**: `nfcorpus`
- **Corpus Documents**: 3,633
- **Evaluated Queries**: 323
- **Top-K**: 10
- **Host**: x86_64 (Linux)
- **Resource Envelope**: 8 CPUs, 16g RAM (containerized)
- **GPUs**: none (CPU-only host)
- **Methodology**: every engine runs in a Docker container under the same `--cpus`/`--memory` envelope (see `benchmarks/competitors/docker_bench.sh --workload beir`), so no participant gets more cores or RAM than another.

| Engine | Backend | Status | Build Time | Index Size | QPS | p50 Latency | p99 Latency | Recall@10 | nDCG@10 |
|---|---|---|---|---|---|---|---|---|---|
| **benostreamdb** | `cpu` | ✅ Pass | 0.30s | 2.5 MB | **655.5** | **1.34 ms** | 2.36 ms | 0.1491 | 0.3067 |
| **tantivy** | `cpu` | ✅ Pass | 0.15s | 6.2 MB | **10697.8** | **0.05 ms** | 0.19 ms | 0.1452 | 0.2998 |
| **opensearch** | `cpu` | ✅ Pass | 0.56s | 4.8 MB | **1171.9** | **0.85 ms** | 1.20 ms | 0.1532 | 0.3215 |

### Differential Oracle & Result Agreement

- **Top-10 Jaccard Overlap vs tantivy**: **92.2%**.
- **Top-10 Jaccard Overlap vs opensearch**: **61.9%**.
- High ranking agreement validates correct Okapi BM25 implementation across vocabulary, inverted postings, and document length normalization sidecars.
