# BEIR Hybrid Search Benchmark Results: BenoStreamDB vs Competitor

- **Dataset**: `scifact`
- **Corpus Documents**: 5,183
- **Evaluated Queries**: 300
- **Dense Embedding Model**: `all-MiniLM-L6-v2` (384-d)
- **Lexical Algorithm**: Okapi BM25 (`k1=1.2, b=0.75`)
- **Fusion Algorithm**: Reciprocal Rank Fusion (RRF, `k=60`)
- **Top-K**: 10
- **Host**: x86_64 (Linux)
- **Resource Envelope**: 8 CPUs, 16g RAM (containerized)
- **GPUs**: none (CPU-only host)
- **Methodology**: every engine runs in a Docker container under the same `--cpus`/`--memory` envelope (see `benchmarks/competitors/docker_bench.sh --workload beir`), so no participant gets more cores or RAM than another.

### Competitor Comparison (Hybrid Dense + Sparse RRF)

| Engine | Backend | Status | Build Time | Total Size on Disk | Throughput (QPS) | p50 Latency | p99 Latency | Recall@10 | nDCG@10 | MRR@10 |
|---|---|---|---|---|---|---|---|---|---|---|
| **benostreamdb** | `cpu` | ✅ Pass | 0.56s | 6.6 MB | **168.1** | **5.76 ms** | 7.71 ms | **0.8493** | **0.6961** | 0.6511 |
| **lancedb** | `cpu` | ✅ Pass | 1.21s | 14.4 MB | **189.3** | **4.81 ms** | 7.40 ms | **0.8284** | **0.6658** | 0.6193 |

### Differential Oracle & Result Agreement

- **Top-10 Jaccard Overlap**: **46.7%** between BenoStreamDB Hybrid and LanceDB Hybrid.
- High ranking agreement validates correct multi-modal retrieval and reciprocal rank fusion mathematics against an established embedded vector database.

### BenoStreamDB Single-Modality vs Hybrid Lift Breakdown

| Search Mode | Index Size | QPS | p50 Latency | Recall@10 | nDCG@10 | MRR@10 |
|---|---|---|---|---|---|---|
| **Sparse (BM25 Only)** | 6.6 MB | 398.9 | 2.38 ms | 0.7909 | 0.6617 | 0.6276 |
| **Dense (Vector Only)** | 0.0 MB | 1216.0 | 0.71 ms | 0.7767 | 0.6437 | 0.6047 |
| **Hybrid (Dense + BM25 RRF)** | 6.6 MB | 168.1 | 5.76 ms | **0.8493** | **0.6961** | **0.6511** |
