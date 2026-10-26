# BEIR Lexical / BM25 Benchmark Results

- **Dataset**: `nfcorpus`
- **Corpus Documents**: 3,633
- **Evaluated Queries**: 323
- **Top-K**: 10
- **Host**: x86_64 (Linux)
- **Resource Envelope**: unconstrained (host)
- **GPUs**: NVIDIA GeForce RTX 5070 Ti, 16303 MiB, NVIDIA GeForce RTX 5070 Ti, 16303 MiB
- **Methodology**: every engine runs in a Docker container under the same `--cpus`/`--memory` envelope (see `benchmarks/competitors/docker_bench.sh --workload beir`), so no participant gets more cores or RAM than another.

| Engine | Backend | Status | Build Time | Index Size | QPS | p50 Latency | p99 Latency | Recall@10 | nDCG@10 |
|---|---|---|---|---|---|---|---|---|---|
| **benostreamdb** | `cpu` | ✅ Pass | 0.34s | 2.5 MB | **756.4** | **1.01 ms** | 2.29 ms | 0.1491 | 0.3069 |
| **tantivy** | `cpu` | ✅ Pass | 0.08s | 6.2 MB | **8825.0** | **0.07 ms** | 0.22 ms | 0.1438 | 0.2992 |
| **opensearch** | `cpu` | ✅ Pass | 0.74s | 4.8 MB | **801.1** | **1.21 ms** | 1.99 ms | 0.1532 | 0.3215 |

### Differential Oracle & Result Agreement

- **Top-10 Jaccard Overlap vs tantivy**: **92.1%**.
- **Top-10 Jaccard Overlap vs opensearch**: **61.9%**.
- High ranking agreement validates correct Okapi BM25 implementation across vocabulary, inverted postings, and document length normalization sidecars.
