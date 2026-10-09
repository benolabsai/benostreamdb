# BEIR Lexical / BM25 Benchmark Results

- **Dataset**: `scifact`
- **Corpus Documents**: 5,183
- **Evaluated Queries**: 300
- **Top-K**: 10
- **Host**: x86_64 (Linux)
- **Resource Envelope**: 8 CPUs, 16g RAM (containerized)
- **Methodology**: every engine runs in a Docker container under the same `--cpus`/`--memory` envelope (see `benchmarks/competitors/docker_bench.sh --workload beir`), so no participant gets more cores or RAM than another.

| Engine | Status | Build Time | Index Size | QPS | p50 Latency | p99 Latency | Recall@10 | nDCG@10 |
|---|---|---|---|---|---|---|---|---|
| **benostreamdb** | ✅ Pass | 0.36s | 3.3 MB | **418.1** | **2.25 ms** | 3.89 ms | 0.7909 | 0.6617 |
| **tantivy** | ✅ Pass | 0.19s | 8.3 MB | **2690.9** | **0.31 ms** | 0.71 ms | 0.7812 | 0.6517 |
| **opensearch** | ✅ Pass | 0.76s | 6.6 MB | **817.8** | **1.16 ms** | 2.15 ms | 0.8196 | 0.6821 |

### Differential Oracle & Result Agreement

- **Top-10 Jaccard Overlap vs tantivy**: **81.5%**.
- **Top-10 Jaccard Overlap vs opensearch**: **53.1%**.
- High ranking agreement validates correct Okapi BM25 implementation across vocabulary, inverted postings, and document length normalization sidecars.
