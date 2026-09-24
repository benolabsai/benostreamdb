# Benchmarking

The only performance numbers BenoStreamDB stands behind are from the **full-site
English-Wikipedia Graph-RAG demo** ([`examples/web_ui/`](../examples/web_ui/README.md)).
It is end-to-end, reproducible, and measured on documented hardware, so it doubles
as the project's scale test.

> **Everything else was removed.** The earlier OpenSearch / Elasticsearch / LanceDB
> comparisons, the NYC-Taxi micro-runs, the `tests/benchmarks` suite, and the
> Criterion `benches/` targets were stale and are no longer maintained. If a number
> is not in the demo README below, treat it as unverified.

## The measured workload

Reference hardware: **32-core x86-64 Linux, 121 GB RAM, NVMe**, NVIDIA RTX 3090.
Dataset: the whole `enwiki` current-pages dump — 66.1M raw pages / 498M raw links
→ **51.8M live pages / 383M clean int64 edges** after redirect filtering and
endpoint resolution.

| Stage | Wall time | Peak memory |
| --- | --- | --- |
| download (19 chunks, ~48 GB) | ~2 h | <1 GB |
| parse (streaming Rust) | ~2.6 h | < 2 GB |
| resolve (polars, chunked) | ~3–5 min | ~5–8 GB |
| embed (all-MiniLM-L6-v2, 384-d, RTX 3090) | 2.1 h | **8.3 GB RSS**, 5.9 GB VRAM |
| load — edges (383M + CSR) | 271 s | ~6 GB |
| load — nodes (51.8M + HNSW-TQ8 + BM25) | **41 min** | **14.6–18.0 GB/chunk** |

The per-chunk node-load table, the embed throughput (~6,300 sent/s), and the full
reproduction commands live in
[`examples/web_ui/README.md`](../examples/web_ui/README.md) — that is the single
source of truth for these numbers.

## Reproducing

```bash
# Full pipeline (idempotent and resumable; every stage skips finished work)
python scripts/prepare_demo.py

# One stage at a time
python scripts/prepare_demo.py --stage embed
python scripts/prepare_demo.py --stage load --rebuild --keep-shards
python scripts/prepare_demo.py --stage compact
```

### Logs

Write run logs to the git-ignored **`logs/`** directory — never the repo root:

```bash
mkdir -p logs
python scripts/prepare_demo.py --stage load 2>&1 | tee logs/load.log
```

`logs/` is in `.gitignore`, so nothing you capture there can pollute the tree.

## Measurement methodology

The demo pipeline reports real, in-process numbers rather than estimates:

- **Peak host memory** is each load chunk's own `/proc/self/status` `VmHWM` (the
  chunk runs in a fresh process, so its high-water mark is exactly that chunk's
  peak), logged at chunk end. Live `VmRSS` is logged on each progress line.
- **Peak GPU memory** is `torch.cuda.max_memory_reserved()` / `max_memory_allocated()`
  reported at the end of the embed stage.
- **Chunk size** is auto-sized from `MemAvailable` at load start
  ([`_auto_chunk_rows`](../scripts/prepare_demo.py:64)); the logged estimate is the
  heuristic, the logged `VmHWM` is the measured peak.

Because every number is emitted by the run itself, the documentation is updated
from logs rather than hand-copied — see [`RESOURCE_LIMITS.md`](RESOURCE_LIMITS.md)
for how the engine's memory guards shape these figures.
