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
4. **embed** — sentence-transformers (`all-MiniLM-L6-v2` by default — the
   two-level design only needs a cheap 384-d **seed index**, and reranking is
   bitmap-filtered to the retrieved neighborhood; `--embed-model
   BAAI/bge-large-en-v1.5` costs ~20× more: measured 279 vs 6,065 sent/s)
   on GPU if present → `data/embeddings/part-*.parquet` (resumable per shard).
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

> ⏳ Wall-time for **load** below is an estimate; the whole-site load
> (HNSW-TQ8 build over 51.8M vectors) is in progress and will be replaced with
> a measured number when it completes. Embed is now measured.

| Stage | Wall time | Peak memory | Notes |
|---|---|---|---|
| download (19 chunks, ~48 GB) | ~2 h | <1 GB | 3 parallel streams; Wikimedia returns HTTP 429 beyond ~3 — the downloader is 429-aware and resumes via `.part` |
| parse (streaming Rust) | ~2.6 h | **< 2 GB** | 66.1M pages + 498M edges; the old accumulate-then-write design OOM-killed at ~110 GB — streaming per-chunk flushes fixed it |
| resolve (polars, chunked) | ~3–5 min | **~5–8 GB** | per-edge-chunk join against a single broadcast title→curid map; replaced a pandas pass that OOM'd at 47 GB / 17 min at ⅕ scale |
| embed (all-MiniLM-L6-v2, 384-d, **RTX 3090**) | **~2.3 h** | **< 4 GB RAM**, ~6 GB VRAM | streaming per-row-group, fp16, batch 512; **6,065 sent/s measured** (bge-large-1024: 279 sent/s = 50 h — rejected for the seed index) |
| load (hdb tables + HNSW-TQ8 + CSR + BM25) | ~2–4 h (est.) | ~4–8 GB | streams embeddings into the table and **deletes each shard after ingest** to bound peak disk |

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

## Minimum requirements to run this

Two distinct workloads with very different bars: **preparing** the dataset
(offline, one-time) and **serving** the demo (online, per query).

### Serving the whole-site demo (the app)

| Resource | Absolute floor | Comfortable | Why |
|---|---|---|---|
| RAM | **16 GB** | 24–32 GB | Vector + CSR indexes are **mmap-backed** (`use_mmap` default + `MADV_RANDOM`), so only pages a query touches are resident (upper HNSW layers + visited nodes ≈ a few GB). 16 GB runs graph + keyword + dense search, just with more page faults. |
| Disk | **~150 GB** | 250 GB | nodes table with 384-d f32 embeddings ≈ 80 GB + TQ8 index ~25 GB + edges/CSR (383M) ~10 GB + BM25 ~5 GB. |
| CPU | 4 cores | 8+ | hybrid/PPR/DRIFT are engine-side; DRIFT over a big region is the heaviest. |
| GPU | none | optional | GPU only speeds *prep* (embedding); serving reads the prebuilt index. |

**Laptop verdict (16 GB, no GPU, 512 GB SSD):** serving the whole-site graph +
keyword + dense search **works** (mmap keeps RAM bounded; disk is the tight
constraint). What a laptop can't do is *prepare* it fast — see below.

### Preparing the whole-site dataset (offline)

| Resource | Floor | Notes |
|---|---|---|
| RAM | **8 GB** | every stage is now streaming/chunked (parse <2 GB, resolve ~5–8 GB, embed <4 GB, load ~4–8 GB). The old 47–55 GB spikes are gone. |
| Disk | **~200 GB free** | peak = ~80 GB embedding shards + growing table before shard-delete; dumps (48 GB) are re-fetchable and can be deleted after parse. |
| GPU | strongly advised | MiniLM-384 on a 3090 ≈ **2.5 h** (measured); on CPU expect ~10–20 h. bge-large-1024 would take 50 h on the same GPU. |
| Time | **~9–11 h** end-to-end | parse ~2.6 h; embed ~2.3 h; load ~2–4 h; download ~2 h. |

**If you have neither a GPU nor ~300 GB disk**, use the pruned profile instead
(`scripts/build_demo_dataset.py`, ~50k-node hub-centered subgraph → ~155 MB
tables, embeds in ~10 min on CPU, runs anywhere). The whole-site path and the
pruned path share the same app and engine APIs — only scale differs.

## Troubleshooting

- **“Demo tables not found”** → run `python scripts/prepare_demo.py` first.
- **Semantic tab warns “Embedder unavailable”** → `pip install sentence-transformers`,
  and ensure `HDB_DEMO_EMBED_MODEL` matches the embed stage model.
- **DRIFT tab fails on huge regions** → lower *Region seeds*; the region is a
  1-hop induced subgraph around the query's top pages.
- **Out of disk** → the pipeline needs ~200 GB free (dumps 48 GB, intermediates
  ~36 GB, embedding shards ~80 GB, tables ~105 GB). `data_full/` is safe to
  delete once `data/wiki_*.parquet` exist.
