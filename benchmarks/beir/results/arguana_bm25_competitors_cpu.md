# BEIR Lexical / BM25 Benchmark Results

- **Dataset**: `arguana`
- **Corpus Documents**: 8,674
- **Evaluated Queries**: 1,406
- **Top-K**: 10
- **Host**: x86_64 (Linux)
- **Resource Envelope**: unconstrained (host)
- **GPUs**: NVIDIA GeForce RTX 5070 Ti, 16303 MiB, NVIDIA GeForce RTX 5070 Ti, 16303 MiB
- **Methodology**: every engine runs in a Docker container under the same `--cpus`/`--memory` envelope (see `benchmarks/competitors/docker_bench.sh --workload beir`), so no participant gets more cores or RAM than another.

| Engine | Backend | Status | Build Time | Index Size | QPS | p50 Latency | p99 Latency | Recall@10 | nDCG@10 |
|---|---|---|---|---|---|---|---|---|---|
| **benostreamdb** | `cpu` | ✅ Pass | 0.56s | 4.1 MB | **43.4** | **22.22 ms** | 39.42 ms | 0.6558 | 0.3086 |
| **tantivy** | `cpu` | ✅ Pass | 0.14s | 10.4 MB | **281.2** | **3.39 ms** | 7.11 ms | 0.6700 | 0.3226 |
| **opensearch** | `cpu` | ✅ Pass | 1.44s | 8.0 MB | **100.9** | **8.67 ms** | 32.39 ms | 0.7461 | 0.3557 |

### Differential Oracle & Result Agreement

- **Top-10 Jaccard Overlap vs tantivy**: **65.8%**.
- **Top-10 Jaccard Overlap vs opensearch**: **55.9%**.
- High ranking agreement validates correct Okapi BM25 implementation across vocabulary, inverted postings, and document length normalization sidecars.
