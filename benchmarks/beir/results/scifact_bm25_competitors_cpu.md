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
| **benostreamdb** | `cpu` | ✅ Pass | 0.37s | 3.3 MB | **422.9** | **2.22 ms** | 3.77 ms | 0.7909 | 0.6617 |
| **tantivy** | `cpu` | ✅ Pass | 0.16s | 8.3 MB | **3520.2** | **0.23 ms** | 0.53 ms | 0.7812 | 0.6517 |
| **opensearch** | `cpu` | ✅ Pass | 0.85s | 6.6 MB | **507.9** | **1.72 ms** | 4.03 ms | 0.8196 | 0.6821 |

### Differential Oracle & Result Agreement

- **Top-10 Jaccard Overlap vs tantivy**: **81.5%**.
- **Top-10 Jaccard Overlap vs opensearch**: **53.1%**.
- High ranking agreement validates correct Okapi BM25 implementation across vocabulary, inverted postings, and document length normalization sidecars.
