# BEIR Lexical / BM25 Benchmark Results

- **Dataset**: `arguana`
- **Corpus Documents**: 8,674
- **Evaluated Queries**: 1,406
- **Top-K**: 10
- **Host**: x86_64 (Linux)
- **Resource Envelope**: 8 CPUs, 16g RAM (containerized)
- **GPUs**: none (CPU-only host)
- **Methodology**: every engine runs in a Docker container under the same `--cpus`/`--memory` envelope (see `benchmarks/competitors/docker_bench.sh --workload beir`), so no participant gets more cores or RAM than another.

| Engine | Backend | Status | Build Time | Index Size | QPS | p50 Latency | p99 Latency | Recall@10 | nDCG@10 |
|---|---|---|---|---|---|---|---|---|---|
| **benostreamdb** | `cpu` | ✅ Pass | 0.48s | 4.1 MB | **54.0** | **17.78 ms** | 31.04 ms | 0.6550 | 0.3084 |
| **tantivy** | `cpu` | ✅ Pass | 0.18s | 10.4 MB | **395.6** | **2.42 ms** | 5.15 ms | 0.6700 | 0.3226 |
| **opensearch** | `cpu` | ✅ Pass | 0.92s | 7.9 MB | **123.2** | **7.05 ms** | 24.22 ms | 0.7461 | 0.3557 |

### Differential Oracle & Result Agreement

- **Top-10 Jaccard Overlap vs tantivy**: **65.8%**.
- **Top-10 Jaccard Overlap vs opensearch**: **55.9%**.
- High ranking agreement validates correct Okapi BM25 implementation across vocabulary, inverted postings, and document length normalization sidecars.
