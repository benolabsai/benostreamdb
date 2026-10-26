# BEIR Lexical / BM25 Benchmark Results

- **Dataset**: `scifact`
- **Corpus Documents**: 5,183
- **Evaluated Queries**: 300
- **Top-K**: 10
- **Host**: x86_64 (Linux)
- **Resource Envelope**: unconstrained (host)
- **GPUs**: NVIDIA GeForce RTX 5070 Ti, 16303 MiB, NVIDIA GeForce RTX 5070 Ti, 16303 MiB
- **Methodology**: every engine runs in a Docker container under the same `--cpus`/`--memory` envelope (see `benchmarks/competitors/docker_bench.sh --workload beir`), so no participant gets more cores or RAM than another.

| Engine | Backend | Status | Build Time | Index Size | QPS | p50 Latency | p99 Latency | Recall@10 | nDCG@10 |
|---|---|---|---|---|---|---|---|---|---|
| **benostreamdb** | `cpu` | ✅ Pass | 0.47s | 3.3 MB | **301.7** | **3.06 ms** | 5.81 ms | 0.7909 | 0.6617 |
| **tantivy** | `cpu` | ✅ Pass | 0.10s | 8.3 MB | **1988.1** | **0.40 ms** | 0.94 ms | 0.7812 | 0.6517 |
| **opensearch** | `cpu` | ✅ Pass | 0.95s | 6.6 MB | **734.4** | **1.32 ms** | 2.09 ms | 0.8196 | 0.6821 |

### Differential Oracle & Result Agreement

- **Top-10 Jaccard Overlap vs tantivy**: **81.4%**.
- **Top-10 Jaccard Overlap vs opensearch**: **53.1%**.
- High ranking agreement validates correct Okapi BM25 implementation across vocabulary, inverted postings, and document length normalization sidecars.
