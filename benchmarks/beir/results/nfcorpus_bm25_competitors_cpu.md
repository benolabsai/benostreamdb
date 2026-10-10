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
| **benostreamdb** | `cpu` | ✅ Pass | 0.28s | 2.5 MB | **1041.6** | **0.73 ms** | 1.69 ms | 0.1491 | 0.3067 |
| **tantivy** | `cpu` | ✅ Pass | 0.17s | 6.2 MB | **11310.4** | **0.05 ms** | 0.19 ms | 0.1441 | 0.2997 |
| **opensearch** | `cpu` | ✅ Pass | 0.53s | 4.8 MB | **1466.7** | **0.70 ms** | 0.96 ms | 0.1532 | 0.3215 |

### Differential Oracle & Result Agreement

- **Top-10 Jaccard Overlap vs tantivy**: **92.3%**.
- **Top-10 Jaccard Overlap vs opensearch**: **61.9%**.
- High ranking agreement validates correct Okapi BM25 implementation across vocabulary, inverted postings, and document length normalization sidecars.
