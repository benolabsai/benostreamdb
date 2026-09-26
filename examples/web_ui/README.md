# BenoStreamDB — full-site Wikipedia Graph RAG demo

A Streamlit UI that puts BenoStreamDB's hybrid engine through its paces on the
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
> in **fresh-process chunks** whose size auto-scales from available RAM
> (`--load-chunk-rows` / `BSDB_LOAD_CHUNK_ROWS` to override) — in-process
> HNSW/TQ8 builders strand freed memory in glibc's main heap (~4.5 GB per
> million rows measured), so a new process per chunk resets the high-water
> mark. On this 121 GB box: 10M-row chunks ≈ 30 GB peak at 12.6k rows/s; an
> 8 GB laptop auto-selects ~250k-row chunks; 500k-row chunks ≈ 3-4 GB
> (serverless-task sized). Chunks must be ≥ ~10 flush-segments (750k rows
> each) so index builds overlap writes instead of serializing into a tail.

From the repository root:

```bash
pip install -e ".[dev]"          # benostreamdb + polars/sentence-transformers etc.
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
   (`BSDB_EMBED_SHARD_ROWS`; 51.8M pages → 11 shards ≈ 36 GB). The skip check is
   all-or-nothing: if an interrupted run left shards behind, delete
   `data/embeddings/` and rerun. `--embed-dims 256/128` MRL-truncates.
5. **load** — builds the two persistent BenoStreamDB tables under
   `data/wiki_graph_db/` (nodes in fresh-process 10M-row chunks, see above):
   - `edges` — `(source, target) int64` + CSR graph index
   - `nodes` — `(id, title, summary, embedding)` + `hnsw_tq8` vector index
     (`--quant tq4|tq8|none`) + BM25 inverted index on `title`
   Embedding shards are streamed in 250k-row batches and deleted only **after**
   the table commits. If load dies mid-run: `rm -rf data/wiki_graph_db/nodes`
   and rerun `--stage load` (edges table and shards are untouched).
   The index config is persisted in the table (manifest schema) at the first
   commit and restored on open, so every chunk process indexes its own segments.
6. **compact** — final step: `rewrite_data_files` merges the many small
   segments chunked loading produces (each query fans out over all of them) into
   ~2×`--compact-min-bytes` (default 2 GB → 4 GB) segments, **rebuilding their
   vector/inverted indexes** as it rewrites. Runs automatically at the end of
   `--stage all`.

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

### Live tracing, resources & bottlenecks

Each search tab instruments itself so you can see what the engine is doing:

- **Engine trace** — a live, auto-scrolling, fixed-height log of every pipeline
  stage (embed → search → rerank → fetch), timestamped with per-stage cost and
  the process RSS at each step.
- **Query plan (explain)** — the engine's `EXPLAIN` output: segments scanned vs.
  pruned (with per-rule reasons) and the index access path chosen (Inverted
  Index / Bitmap / Full scan).
- **Process resources** — a live RSS / peak-RSS / disk-read / disk-write / CPU
  readout (sampled from `/proc`) with deltas since the query started, plus DB
  filesystem headroom. It re-samples every 0.4s while the query runs.
- **Bottleneck breakdown** — stages ranked by wall time with the dominant stage
  called out, a bar chart, and resource deltas across the run.
- **End-to-end timing** — each tab prints its total wall time.

Toggle the trace and plan panels from the sidebar. Engine calls run on a worker
thread, so these panels keep updating *during* a query rather than only after it
returns.

### Optional: LLM endpoint

Any OpenAI-compatible provider works — a local vLLM or a hosted one. Set it
with the **standard OpenAI environment variables** (precedence: env >
`.streamlit/secrets.toml` > defaults):

```bash
# local vLLM
export OPENAI_BASE_URL=http://127.0.0.1:18020/v1
export OPENAI_API_KEY=empty
export OPENAI_MODEL=qwen3.8-27b

# or a hosted provider, e.g. OpenRouter (free Qwen 27B):
export OPENAI_BASE_URL=https://openrouter.ai/api/v1
export OPENAI_API_KEY=sk-or-v1-...
export OPENAI_MODEL=qwen/qwen3.8-27b:free
```

Alternatively, put the same keys under `[llm]` in the repo-root
`.streamlit/secrets.toml`. Toggle the LLM off in the sidebar (or
`BSDB_DEMO_LLM=0`) to run purely engine-side.

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `BSDB_DEMO_DB` | `data/wiki_graph_db` | location of the prepared tables |
| `BSDB_DEMO_EMBED_MODEL` | `all-MiniLM-L6-v2` | must match the model used in the embed stage |
| `BSDB_DEMO_LLM` | `1` | `0` disables LLM features by default |
| `OPENAI_BASE_URL` | `http://127.0.0.1:18020/v1` | any OpenAI-compatible endpoint (e.g. `https://openrouter.ai/api/v1`) |
| `OPENAI_API_KEY` | `empty` | provider key (OpenRouter: `sk-or-v1-…`) |
| `OPENAI_MODEL` | `qwen3.8-27b` | e.g. `qwen/qwen3.8-27b:free` on OpenRouter |
| `MALLOC_ARENA_MAX` | engine sets 2 at import | override only if you know better |
| `BSDB_EMBED_SHARD_ROWS` | `5000000` | embed-stage shard rotation size |
| `BSDB_LOAD_CHUNK_ROWS` | auto (from RAM) | load chunk size; `--load-chunk-rows` flag wins over this |

## Timings & system stats (measured on this machine)

Reference hardware: **32-core x86-64 Linux, 121 GB RAM, NVMe** (`/tmp` is a
61 GB tmpfs). Dataset: the **whole** `enwiki` current-pages dump — 66.1M raw
pages / 498M raw links → **51.8M live pages / 383M clean int64 edges** after
redirect filtering and endpoint resolution.

> The **load (nodes)** row and the per-chunk table below are measured on a
> **from-scratch reload** (all 51.8M rows) after the idempotent-backfill fix
> (`add_index` no longer re-indexes every prior segment on each fresh chunk
> process). Before the fix, per-chunk time grew 293 s → 885 s → 2071 s and the
> run was OOM-killed partway through.

| Stage | Wall time | Peak memory | Notes |
|---|---|---|---|
| download (19 chunks, ~48 GB) | ~2 h | <1 GB | 3 parallel streams; Wikimedia returns HTTP 429 beyond ~3 — the downloader is 429-aware and resumes via `.part` |
| parse (streaming Rust) | ~2.6 h | **< 2 GB** | 66.1M pages + 498M edges; the old accumulate-then-write design OOM-killed at ~110 GB — streaming per-chunk flushes fixed it |
| resolve (polars, chunked) | ~3–5 min | **~5–8 GB** | per-edge-chunk join against a single broadcast title→curid map; replaced a pandas pass that OOM'd at 47 GB / 17 min at ⅕ scale |
| embed (all-MiniLM-L6-v2, 384-d, **RTX 3090**) | **2.1 h** | **8.3 GB peak RSS**, 5.9 GB VRAM reserved | fp16, batch 512, 256-char leads; **~6,300 sent/s avg** (measured 6,606 → 6,189) → 11 shards / ~40 GB (bge-large-1024: 279 sent/s = 50 h — rejected for the seed index) |
| load — edges (383M + CSR) | **245 s** | ~6 GB | 15 GB table incl. CSR sidecars |
| load — nodes (51.8M + HNSW-TQ8 + BM25) | **38.7 min** | **13.3–16.2 GB/chunk (10M rows)** (measured peak RSS, ~1.5 GB/M rows) | **~22k rows/s sustained**, ~36–52 s per M rows; the auto-sizer *estimated* ~45 GB/chunk (its `4.5 GB/M` constant is ~3× conservative) — builds overlap writes inside the chunk; shards deleted after all chunks commit |

Per-chunk node-load timings (from-scratch reload, post-fix) — note the flat rate
and the real peak RSS vs the ~36 GB the auto-sizer predicted:

| Chunk | Rows | Wall time | Peak RSS | s / M rows |
|---|---|---|---|---|
| [0, 8.0M) | 8,000,000 | 269 s | 14.6 GB | 33.6 |
| [8.0M, 16.0M) | 8,000,000 | 336 s | 15.2 GB | 42.0 |
| [16.0M, 24.0M) | 8,000,000 | 355 s | 15.3 GB | 44.4 |
| [24.0M, 32.0M) | 8,000,000 | 346 s | 17.4 GB | 43.3 |
| [32.0M, 40.0M) | 8,000,000 | 421 s | 17.0 GB | 52.6 |
| [40.0M, 48.0M) | 8,000,000 | 404 s | 16.3 GB | 50.5 |
| [48.0M, 51.79M) | 3,793,809 | 337 s | 18.0 GB | 88.9 |
| **total** | **51,793,809** | **2,468 s (41 min)** | **max 18.0 GB** | **47.6** |

Before the fix the same chunks escalated (37 → 110 → 259 s / M rows) and the
process was killed. The real peak (~2.0 GB/M rows) is ~2.2× below the
`4.5 GB/M` sizing constant, so the chunk could be roughly doubled for the same
memory budget.

### Engine benchmarks (91.7M-edge graph, debug build)

| Operation | Time |
|---|---|
| bsdb ingest + commit, 91.7M edges | **37.8 s** |
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
| RAM | **8 GB** | every stage is streaming/chunked (parse <2 GB, resolve ~5–8 GB, embed <4 GB, load auto-sized from MemAvailable — ~2–3 GB per chunk on an 8 GB host; `--load-chunk-rows` overrides). The old 47–82 GB single-process spikes are gone. |
| Disk | **~200 GB free** | peak = ~36 GB embedding shards + growing tables; shards are deleted after the nodes table commits; dumps (48 GB) are re-fetchable and can be deleted after parse. |
| GPU | strongly advised | MiniLM-384 on a 3090 ≈ **2.1 h** (measured, 6,750 sent/s); on CPU expect ~10–20 h. bge-large-1024 would take 50 h on the same GPU. |
| Time | **~8–10 h** end-to-end | download ~2 h; parse ~2.6 h; embed ~2.1 h; load ~1.5–2.5 h. |

**If you have neither a GPU nor ~300 GB disk**, use the pruned profile instead
(`scripts/build_demo_dataset.py`, ~50k-node hub-centered subgraph → ~155 MB
tables, embeds in ~10 min on CPU, runs anywhere). The whole-site path and the
pruned path share the same app and engine APIs — only scale differs.

## Troubleshooting

- **“Demo tables not found”** → run `python scripts/prepare_demo.py` first.
- **Engine fixes seem to have no effect after `maturin develop`** → verify the
  *installed* extension timestamp; maturin has been observed reporting
  “Installed” while leaving a stale `.so`, so edits appear inert (this cost us
  an evening). Force it explicitly:
  `cargo build --features python && cp target/debug/libbenostreamdb.so
  .venv*/lib/python*/site-packages/benostreamdb/benostreamdb.abi3.so`
- **Queries are slow / reading hundreds of GB** → check that segments actually
  carry indexes: a nodes dir with far more `seg_*.parquet` files than
  `*.tq8.centroids.parquet` files means segments were written without the index
  config (fixed: the config persists in the manifest and is restored on open).
  `--stage compact` rewrites and re-indexes them.
- **Queries slow despite indexes** → the engine's index cache defaults to 1 GB;
  the app sets `BENOSTREAM_CACHE_GB=40` (69 segment indexes ≈ 20 GB). Set it
  for any other client too.
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
  and ensure `BSDB_DEMO_EMBED_MODEL` matches the embed stage model.
- **DRIFT tab fails on huge regions** → lower *Region seeds*; the region is a
  1-hop induced subgraph around the query's top pages.
- **Out of disk** → the pipeline needs ~200 GB free (dumps 48 GB, intermediates
  ~36 GB, embedding shards ~36 GB, tables ~105 GB). `data_full/` is safe to
  delete once `data/wiki_*.parquet` exist.
