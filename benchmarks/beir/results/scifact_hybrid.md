# BEIR Hybrid Search Benchmark Results: BenoStreamDB vs Competitor

- **Dataset**: `scifact`
- **Corpus Documents**: 5,183
- **Evaluated Queries**: 300
- **Dense Embedding Model**: `all-MiniLM-L6-v2` (384-d)
- **Lexical Algorithm**: Okapi BM25 (`k1=1.2, b=0.75`)
- **Fusion Algorithm**: Reciprocal Rank Fusion (RRF, `k=60`)
- **Top-K**: 10
- **Host**: x86_64 (Linux)

### Competitor Comparison (Hybrid Dense + Sparse RRF)

| Engine | Status | Build Time | Total Size on Disk | Throughput (QPS) | p50 Latency | p99 Latency | Recall@10 | nDCG@10 | MRR@10 |
|---|---|---|---|---|---|---|---|---|---|
| **benostreamdb** | ✅ Pass | 0.98s | 6.7 MB | **190.7** | **5.12 ms** | 6.94 ms | **0.8460** | **0.6859** | 0.6394 |
| **lancedb** | ✅ Pass | 0.71s | 13.9 MB | **167.2** | **5.70 ms** | 7.36 ms | **0.8367** | **0.7118** | 0.6783 |

### Differential Oracle & Result Agreement

- **Top-10 Jaccard Overlap**: **60.2%** between BenoStreamDB Hybrid and LanceDB Hybrid.
- High ranking agreement validates correct multi-modal retrieval and reciprocal rank fusion mathematics against an established embedded vector database.

### BenoStreamDB Single-Modality vs Hybrid Lift Breakdown

| Search Mode | Index Size | QPS | p50 Latency | Recall@10 | nDCG@10 | MRR@10 |
|---|---|---|---|---|---|---|
| **Sparse (BM25 Only)** | 6.7 MB | 414.3 | 2.29 ms | 0.7909 | 0.6617 | 0.6276 |
| **Dense (Vector Only)** | 0.0 MB | 1308.8 | 0.67 ms | 0.7517 | 0.6290 | 0.5935 |
| **Hybrid (Dense + BM25 RRF)** | 6.7 MB | 190.7 | 5.12 ms | **0.8460** | **0.6859** | **0.6394** |
