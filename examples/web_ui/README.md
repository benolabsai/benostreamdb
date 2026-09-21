# HyperStreamDB — full-site Wikipedia Graph RAG demo

A Streamlit UI that puts HyperStreamDB's hybrid engine through its paces on the
**entire English-Wikipedia link graph** (~51.8M live pages, ~383M edges):

| Tab | Capability exercised |
|---|---|
| Browse | engine-side SQL over 51M rows (keyset pagination, no client-side table) |
| Semantic search | HNSW + TurboQuant-8 dense vectors fused with BM25 keyword search via RRF (`hybrid_search`) |
| Graph RAG (local) | two-level: seed discovery → multi-hop induced subgraph → Personalized PageRank (HippoRAG-style) via `graph_rag_search`, then a **bitmap-filtered rerank** (`id IN (...)` → RoaringBitmap predicate on the HNSW index: topology prunes, semantics orders) |
| DRIFT (regional) | Louvain community rollup (`summarize_communities`) + iterative-deepening `drift_search` |
| Graph traversals | microsecond CSR-index hops: `shortest_path`, `connecting_paths`, `graph_neighbors`, `subgraph` |

The LLM (any OpenAI-compatible endpoint, e.g. a local vLLM server) is **optional**
— it powers query parsing/answer synthesis only. Every tab degrades gracefully
without it, and without the embedder.

## 1. Prepare the data (one-time, ~4–6 h)

> No env-var prefixes needed: the engine self-tunes glibc malloc arenas at
> import (`mallopt(M_ARENA_MAX, 2)`), and the load stage writes the nodes table
> in **fresh-process chunks of 10M rows** (`--load-chunk-rows`) — in-process
> HNSW/TQ8 builders strand freed memory in glibc's main heap (~2.6 GB per
> million rows measured), so a new process per chunk resets the high-water mark
> and keeps peak RSS ~25 GB regardless of dataset size.

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
4. **embed** — sentence-transformers over each article's lead
   (`--lead-chars 256` by default; `all-MiniLM-L6-v2` — the two-level design
   only needs a cheap 384-d **seed index**, and reranking is bitmap-filtered
   to the retrieved neighborhood; `--embed-model BAAI/bge-large-en-v1.5` costs
   ~20× more: measured 279 vs ~6,700 sent/s) on GPU if present. Writes
   `data/embeddings/part-NNN.parquet` shards of 5M rows
   (`HDB_EMBED_SHARD_ROWS`; 51.8M pages → 11 shards ≈ 36 GB). The skip check is
   all-or-nothing: if an interrupted run left shards behind, delete
   `data/embeddings/` and rerun. `--embed-dims 256/128` MRL-truncates.
5. **load** — builds the two persistent HyperStreamDB tables under
   `data/wiki_graph_db/` (nodes in fresh-process 10M-row chunks, see above):
   - `edges` — `(source, target) int64` + CSR graph index
   - `nodes` — `(id, title, summary, embedding)` + `hnsw_tq8` vector index
     (`--quant tq4|tq8|none`) + BM25 inverted index on `title`
   Embedding shards are streamed in 250k-row batches and deleted only **after**
   the table commits. If load dies mid-run: `rm -rf data/wiki_graph_db/nodes`
   and rerun `--stage load` (edges table and shards are untouched).

Useful invocations:

```bash
python scripts/prepare_demo.py --stage download --workers 6   # --workers = download only
python scripts/prepare_demo.py --stage embed --embed-batch 1024
python scripts/prepare_demo.py --stage load --rebuild         # rebuild both tables
python scripts/prepare_demo.py --stage load --load-chunk-rows 0   # single process (small datasets)
python scripts/prepare_demo.py --stage load --keep-shards     # don't delete embeddings after commit
df -h .                                                       # watch disk (needs ~200 GB)
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
| `MALLOC_ARENA_MAX` | engine sets 2 at import | override only if you know better |
| `HDB_EMBED_SHARD_ROWS` | `5000000` | embed-stage shard rotation size |

## Timings & system stats (measured on this machine)

Reference hardware: **32-core x86-64 Linux, 121 GB RAM, NVMe** (`/tmp` is a
61 GB tmpfs). Dataset: the **whole** `enwiki` current-pages dump — 66.1M raw
pages / 498M raw links → **51.8M live pages / 383M clean int64 edges** after
redirect filtering and endpoint resolution.

> ⏳ The **load (nodes)** row below is mid-measurement (HNSW-TQ8 build over
> 51.8M vectors); it will be finalized when the run completes. Everything else
> in this table is measured.

| Stage | Wall time | Peak memory | Notes |
|---|---|---|---|
| download (19 chunks, ~48 GB) | ~2 h | <1 GB | 3 parallel streams; Wikimedia returns HTTP 429 beyond ~3 — the downloader is 429-aware and resumes via `.part` |
| parse (streaming Rust) | ~2.6 h | **< 2 GB** | 66.1M pages + 498M edges; the old accumulate-then-write design OOM-killed at ~110 GB — streaming per-chunk flushes fixed it |
| resolve (polars, chunked) | ~3–5 min | **~5–8 GB** | per-edge-chunk join against a single broadcast title→curid map; replaced a pandas pass that OOM'd at 47 GB / 17 min at ⅕ scale |
| embed (all-MiniLM-L6-v2, 384-d, **RTX 3090**) | **2.1 h** | **< 4 GB RAM**, ~7 GB VRAM | fp16, batch 512, 256-char leads; **6,750 sent/s avg** → 11 shards / 36 GB (bge-large-1024: 279 sent/s = 50 h — rejected for the seed index) |
| load — edges (383M + CSR) | **529 s** | ~6 GB | 15 GB table incl. CSR sidecars |
| load — nodes (51.8M + HNSW-TQ8 + BM25) | ~2 h (in progress) | ~25 GB per 10M-row chunk (fresh process each; single-process runs ratchet ~2.6 GB/M rows in glibc's main heap) | chunked parent spawns children sequentially; shards deleted after all chunks commit |

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
| RAM | **8 GB** | every stage is streaming/chunked (parse <2 GB, resolve ~5–8 GB, embed <4 GB, load ~25 GB per fresh-process chunk). The old 47–82 GB single-process spikes are gone. |
| Disk | **~200 GB free** | peak = ~36 GB embedding shards + growing tables; shards are deleted after the nodes table commits; dumps (48 GB) are re-fetchable and can be deleted after parse. |
| GPU | strongly advised | MiniLM-384 on a 3090 ≈ **2.1 h** (measured, 6,750 sent/s); on CPU expect ~10–20 h. bge-large-1024 would take 50 h on the same GPU. |
| Time | **~8–10 h** end-to-end | download ~2 h; parse ~2.6 h; embed ~2.1 h; load ~1.5–2.5 h. |

**If you have neither a GPU nor ~300 GB disk**, use the pruned profile instead
(`scripts/build_demo_dataset.py`, ~50k-node hub-centered subgraph → ~155 MB
tables, embeds in ~10 min on CPU, runs anywhere). The whole-site path and the
pruned path share the same app and engine APIs — only scale differs.

## Troubleshooting

- **“Demo tables not found”** → run `python scripts/prepare_demo.py` first.
- **Load stage RAM climbs into tens of GB** → expected only if you forced
  `--load-chunk-rows 0`; the default fresh-process chunking exists precisely
  because glibc strands freed index-build memory in the main heap.
- **Load died mid-run** → `rm -rf data/wiki_graph_db/nodes && MALLOC_ARENA_MAX=2
  python scripts/prepare_demo.py --stage load` — embedding shards survive until
  post-commit deletion, so only the (re-runnable) load is redone.
- **Embed interrupted** → `rm -rf data/embeddings/` and rerun `--stage embed`
  (the skip check is all-or-nothing).
- **`nvrtc` panic in engine logs** → harmless: cudarc probes for GPU kernel
  compilation, doesn't recognize pip's `libnvrtc.so.13` filename, falls back to
  CPU. To enable the GPU build path: symlink `libnvrtc.so → libnvrtc.so.13` in
  `site-packages/nvidia/cu13/lib/` and add that directory to `LD_LIBRARY_PATH`.
- **Semantic tab warns “Embedder unavailable”** → `pip install sentence-transformers`,
  and ensure `HDB_DEMO_EMBED_MODEL` matches the embed stage model.
- **DRIFT tab fails on huge regions** → lower *Region seeds*; the region is a
  1-hop induced subgraph around the query's top pages.
- **Out of disk** → the pipeline needs ~200 GB free (dumps 48 GB, intermediates
  ~36 GB, embedding shards ~36 GB, tables ~105 GB). `data_full/` is safe to
  delete once `data/wiki_*.parquet` exist.
