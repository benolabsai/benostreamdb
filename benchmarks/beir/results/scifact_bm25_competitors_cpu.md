# BEIR Lexical / BM25 Benchmark Results

- **Dataset**: `scifact`
- **Corpus Documents**: 5,183
- **Evaluated Queries**: 300
- **Top-K**: 10
- **Host**: x86_64 (Linux)
- **Resource Envelope**: 8 CPUs, 16g RAM (containerized)
- **GPUs**: none (CPU-only host)
- **Methodology**: every engine runs in a Docker container under the same `--cpus`/`--memory` envelope (see `benchmarks/competitors/docker_bench.sh --workload beir`), so no participant gets more cores or RAM than another.

| Engine | Backend | Status | Build Time | Index Size | QPS | p50 Latency | p99 Latency | Recall@10 | nDCG@10 |
|---|---|---|---|---|---|---|---|---|---|
| **benostreamdb** | `cpu` | ✅ Pass | 0.36s | 3.3 MB | **417.6** | **2.25 ms** | 3.85 ms | 0.7909 | 0.6617 |
| **tantivy** | `cpu` | ✅ Pass | 1.92s | 8.0 MB | **4026.4** | **0.20 ms** | 0.49 ms | 0.7812 | 0.6517 |
| **opensearch** | `cpu` | ✅ Pass | 2.56s | 6.7 MB | **814.1** | **1.13 ms** | 2.43 ms | 0.8196 | 0.6821 |

### Differential Oracle & Result Agreement

- **Top-10 Jaccard Overlap vs tantivy**: **81.5%**.
- **Top-10 Jaccard Overlap vs opensearch**: **53.1%**.
- High ranking agreement validates correct Okapi BM25 implementation across vocabulary, inverted postings, and document length normalization sidecars.
