# BEIR Lexical / BM25 Benchmark Results

- **Dataset**: `scifact`
- **Corpus Documents**: 5,183
- **Evaluated Queries**: 300
- **Top-K**: 10
- **Host**: x86_64 (Linux)

| Engine | Status | Build Time | Index Size | QPS | p50 Latency | p99 Latency | Recall@10 | nDCG@10 |
|---|---|---|---|---|---|---|---|---|
| **benostreamdb** | ✅ Pass | 0.34s | 3.3 MB | **435.8** | **2.16 ms** | 3.82 ms | 0.7909 | 0.6617 |
| **tantivy** | ✅ Pass | 0.26s | 8.4 MB | **3798.0** | **0.22 ms** | 0.50 ms | 0.7812 | 0.6517 |

### Differential Oracle & Result Agreement

- **Top-10 Jaccard Overlap**: **81.4%** between BenoStreamDB and Tantivy.
- High ranking agreement validates correct Okapi BM25 implementation across vocabulary, inverted postings, and document length normalization sidecars.
