# HyperStreamDB — full-site Wikipedia Graph RAG demo

A Streamlit UI that puts HyperStreamDB's hybrid engine through its paces on the
**entire English-Wikipedia link graph** (~51.8M live pages, ~383M edges):

| Tab | Capability exercised |
|---|---|
| Browse | engine-side SQL over 51M rows (keyset pagination, no client-side table) |
| Semantic search | HNSW + TurboQuant-8 dense vectors fused with BM25 keyword search via RRF (`hybrid_search`) |
| Graph RAG (local) | seed discovery → multi-hop induced subgraph → Personalized PageRank (HippoRAG-style) via `graph_rag_search` |
| DRIFT (regional) | Louvain community rollup (`summarize_communities`) + iterative-deepening `drift_search` |
| Graph traversals | microsecond CSR-index hops: `shortest_path`, `connecting_paths`, `graph_neighbors`, `subgraph` |

The LLM (any OpenAI-compatible endpoint, e.g. a local vLLM server) is **optional**
— it powers query parsing/answer synthesis only. Every tab degrades gracefully
without it, and without the embedder.

## 1. Prepare the data (one-time, ~4–6 h)

From the repository root:

```bash
pip install -e ".[dev]"          # hyperstreamdb + polars/sentence-transformers etc.
python scripts/prepare_demo.py
```

The script is **idempotent and resumable** — rerun it any time; completed stages
are skipped. Stages:

1. **download** — all 19 `enwiki pages-meta-current` chunks (~48 GB) from
   `dumps.wikimedia.org` into `data/` (parallel, `.part`-resumable, 429-aware —
   see `scripts/download_wiki_dumps.py`).
2. **parse** — `cargo run --release --bin ingest_wikipedia` streams the XML dumps
   into `data_full/{nodes,edges}.parquet` (bounded memory, per-chunk flush).
3. **resolve** — polars joins mixed curid/title link endpoints to integer
   `curid`s and drops redirect pages → `data/wiki_nodes.parquet`,
   `data/wiki_edges.parquet`.
4. **embed** — sentence-transformers (`all-MiniLM-L6-v2` by default;
   `--embed-model BAAI/bge-large-en-v1.5` for higher quality at ~10× cost)
   sharded across `--workers` processes → `data/embeddings/part-*.parquet`
   (resumable per shard).
5. **load** — builds the two persistent HyperStreamDB tables under
   `data/wiki_graph_db/`:
   - `edges` — `(source, target) int64` + CSR graph index
   - `nodes` — `(id, title, summary, embedding)` + `hnsw_tq8` vector index +
     BM25 inverted index on `title`

Useful invocations:

```bash
python scripts/prepare_demo.py --stage embed --workers 8   # one stage
python scripts/prepare_demo.py --stage load --rebuild      # rebuild the tables
df -h .                                                    # watch disk (needs ~200 GB)
```

## 2. Launch the demo

```bash
streamlit run examples/web_ui/app.py
```

Open <http://localhost:8501>. The first query per tab warms the indexes; CSR
traversals and vector search respond in milliseconds on the full graph.

### Optional: LLM endpoint

Start any OpenAI-compatible server (e.g. vLLM) and point the app at it via
`examples/web_ui/../../.streamlit/secrets.toml` (repo root `.streamlit/`):

```toml
[llm]
base_url = "http://127.0.0.1:18020/v1"
api_key  = "empty"
model    = "qwen3.8-27b"
```

Toggle the LLM off in the sidebar to run purely engine-side.

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `HDB_DEMO_DB` | `data/wiki_graph_db` | location of the prepared tables |
| `HDB_DEMO_EMBED_MODEL` | `all-MiniLM-L6-v2` | must match the model used in the embed stage |
| `HDB_DEMO_LLM` | `1` | `0` disables LLM features by default |

## Timings & system stats (measured on this machine)

Reference hardware: **32-core x86-64 Linux, 121 GB RAM, NVMe** (`/tmp` is a
61 GB tmpfs). Dataset: the **whole** `enwiki` current-pages dump — 66.1M raw
pages / 498M raw links → **51.8M live pages / 383M clean int64 edges** after
redirect filtering and endpoint resolution.

| Stage | Wall time | Peak memory | Notes |
|---|---|---|---|
| download (14 chunks, ~35 GB) | ~1.8 h | <1 GB | 3 parallel streams; Wikimedia returns HTTP 429 beyond ~3 — the downloader is 429-aware and resumes via `.part` |
| parse (streaming Rust) | ~2.6 h | **< 2 GB** | 66.1M pages + 498M edges; the old accumulate-then-write design OOM-killed at ~110 GB — streaming per-chunk flushes fixed it |
| resolve (polars) | **106 s** | ~47 GB system | two hash joins + streaming sinks; replaced a pandas `map` pass that took ~17 min at 1/5 the scale |
| embed (MiniLM-L6-v2, 6 workers) | ~2 h (est.) | ~55 GB | measured 1,424 sent/s/worker at 8 threads; chunked encode+write keeps workers at 6–9 GB RSS each |
| load (hdb tables + indexes) | ~2–4 h (est.) | — | ingest benchmarked below; HNSW-TQ8 builds per segment |

### Engine benchmarks (91.7M-edge graph, debug build)

| Operation | Time |
|---|---|
| hdb ingest + commit, 91.7M edges | **37.8 s** |
| `connected_components()` (pointer jumping + edge contraction) | **137.4 s** → 509 components, largest 3.93M nodes |
| `subgraph()` frontier BFS, hops=1 | **98.4 s** → 884,700 nodes / 18.7M edges |
| CSR traversals (`shortest_path`, `graph_neighbors`, `connecting_paths`) | milliseconds (memory-mapped `.graph.csr.*`, zero-copy) |

For scale: the previous naive per-hop CC design (two full edge joins per round,
O(diameter) rounds, extra join per convergence check) never completed at this
size; the rewritten pointer-jumping + contraction algorithm finishes in ~2
minutes and spills to disk instead of pinning the edge set in RAM.

### Disk footprint (end state)

| Artifact | Size |
|---|---|
| `data/*.xml.bz2` (19 chunks) | ~48 GB |
| `data/wiki_nodes.parquet` / `wiki_edges.parquet` | 3.6 GB / 1.7 GB |
| `data/embeddings/` (51.8M × 384d fp32) | ~80 GB |
| `data/wiki_graph_db/` (segments + CSR + HNSW-TQ8 + BM25) | ~110 GB |
| **Total** | **~240 GB** (plan `df` accordingly; `data_full/` is safe to delete after resolve) |

## Troubleshooting

- **“Demo tables not found”** → run `python scripts/prepare_demo.py` first.
- **Semantic tab warns “Embedder unavailable”** → `pip install sentence-transformers`,
  and ensure `HDB_DEMO_EMBED_MODEL` matches the embed stage model.
- **DRIFT tab fails on huge regions** → lower *Region seeds*; the region is a
  1-hop induced subgraph around the query's top pages.
- **Out of disk** → the pipeline needs ~200 GB free (dumps 48 GB, intermediates
  ~36 GB, embeddings ~80 GB, tables ~110 GB). `data_full/` is safe to delete
  once `data/wiki_*.parquet` exist.
