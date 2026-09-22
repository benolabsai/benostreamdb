#!/usr/bin/env python3
"""Prepare the FULL English-Wikipedia Graph-RAG demo dataset for HyperStreamDB.

No pruning: the whole site goes in. Every stage is idempotent and resumable —
rerun the script any time and it skips finished work.

Stages:
  1. download  - all enwiki pages-meta-current chunks (scripts/download_wiki_dumps.py)
  2. parse     - XML -> data/full/{nodes,edges}.parquet (Rust: ingest_wikipedia)
  3. resolve   - polars: mixed curid/title endpoints -> int64 curids, redirect
                 pages dropped -> data/wiki_{nodes,edges}.parquet (chunked, low RAM)
  4. embed     - sentence-transformers on GPU (RTX 3090) or CPU, streaming per
                 row-group, sharded + resumable -> data/embeddings/part-*.parquet
  5. load      - persistent HyperStreamDB tables under data/wiki_graph_db/:
                 edges (source,target int64 + CSR graph index)
                 nodes (id,title,summary,embedding + HNSW-TQ vector index + BM25)

bge-large-en-v1.5 (1024-d) is the production-realistic default. Whole-site f32
vectors are ~212 GB, so load deletes each embedding shard after ingesting it to
keep peak disk bounded. Use --embed-dims 512/256 (bge-large supports MRL
truncation) to halve/quarter that if disk is tight.

Usage:
  python scripts/prepare_demo.py                 # everything
  python scripts/prepare_demo.py --stage load    # one stage
  python scripts/prepare_demo.py --embed-dims 512
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data")
DB = os.path.join(DATA, "wiki_graph_db")   # persistent hdb tables
EMB = os.path.join(DATA, "embeddings")

log = lambda m: print(f"[prepare] {m}", flush=True)


def _avail_ram_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            return next(int(l.split()[1]) for l in f if l.startswith("MemAvailable")) / 1048576
    except OSError:
        return 8.0  # conservative default


def _auto_chunk_rows(gb: float | None = None) -> int:
    """Size the fresh-process load chunk from available RAM.

    Measured: the chunked node load needs ~4.5 GB per million rows (live
    flush/index-build buffers + allocator-stranded churn). Use half of
    MemAvailable, rounded to 250k-row batches, clamped to [250k, 10M].
    """
    if gb is None:
        gb = _avail_ram_gb()
    rows = int(gb * 0.5 / 4.5 * 1_000_000)
    chunk = max(250_000, min(10_000_000, rows // 250_000 * 250_000))
    log(f"load: chunk size {chunk:,} rows (auto: {gb:.0f} GB RAM available, "
        f"~{chunk * 4.5 / 1e6:.0f} GB peak)")
    return chunk


def _disk_guard(input_bytes: int, factor: float = 1.3):
    """Warn (not abort) if free disk on the data volume can't hold the output."""
    need = input_bytes * factor
    free = shutil.disk_usage(DATA).free
    if free < need:
        log(f"WARNING: disk guard — need ~{need / 1e9:.0f} GB free "
            f"(inputs × {factor}), have {free / 1e9:.0f} GB. Delete dumps / "
            f"data_full/ if present.")
    else:
        log(f"load: disk ok ({free / 1e9:.0f} GB free ≥ {need / 1e9:.0f} GB needed)")


def _run(cmd, **kw):
    log("$ " + " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, cwd=REPO, check=True, **kw)


def _full_dir() -> str:
    """Parser output dir: prefer data/full, then legacy data_full."""
    for d in (os.path.join(DATA, "full"), os.path.join(REPO, "data_full")):
        if os.path.exists(os.path.join(d, "nodes.parquet")):
            return d
    return os.path.join(DATA, "full")


# ── 1. download ─────────────────────────────────────────────────────────────
def stage_download(workers: int):
    _run([sys.executable, "scripts/download_wiki_dumps.py", "--workers", str(workers)])


# ── 2. parse ────────────────────────────────────────────────────────────────
def stage_parse():
    full = _full_dir()
    if os.path.exists(os.path.join(full, "nodes.parquet")) and \
       os.path.exists(os.path.join(full, "edges.parquet")):
        log(f"parse: already done ({full}) — skipping")
        return
    os.makedirs(full, exist_ok=True)
    _run(["cargo", "run", "--release", "--bin", "ingest_wikipedia", "--",
          "--input-dir", DATA, "--output-dir", full])


# ── 3. resolve (polars, chunked to bound memory) ────────────────────────────
def stage_resolve():
    full = _full_dir()
    src_nodes = os.path.join(full, "nodes.parquet")
    src_edges = os.path.join(full, "edges.parquet")
    out_nodes = os.path.join(DATA, "wiki_nodes.parquet")
    out_edges = os.path.join(DATA, "wiki_edges.parquet")
    if os.path.exists(out_nodes) and os.path.exists(out_edges):
        log("resolve: already done — skipping")
        return

    t0 = time.time()
    nodes_lf = (
        pl.scan_parquet(src_nodes)
        .with_columns(pl.col("id").cast(pl.Int64, strict=False).alias("id"))
        .with_columns(
            pl.col("summary").str.strip_chars_start().str.starts_with("#REDIRECT")
            .fill_null(False).alias("is_redirect")
        )
    )
    # Broadcast title->curid map, built once and reused across edge chunks.
    live_titles = nodes_lf.filter(~pl.col("is_redirect")).select(["title", "id"]).collect()
    lower_titles = live_titles.with_columns(
        (pl.col("title").str.slice(0, 1).str.to_lowercase() + pl.col("title").str.slice(1)).alias("title")
    )
    tm = pl.concat([live_titles, lower_titles]).unique(subset=["title"], keep="first")
    redirect_ids = (
        nodes_lf.filter(pl.col("is_redirect")).select("id").collect()["id"].to_numpy()
    )
    log(f"resolve: title map {len(tm):,} keys, {len(redirect_ids):,} redirects ({time.time()-t0:.0f}s)")

    # Write live nodes (streaming, per row group).
    redir_set = pl.DataFrame({"id": redirect_ids}) if len(redirect_ids) else None
    nw = pq.ParquetWriter(out_nodes, pa.schema([
        ("id", pa.int64()), ("title", pa.large_string()), ("summary", pa.large_string())]))
    for chunk in (nodes_lf.filter(~pl.col("is_redirect"))
                  .select(["id", "title", "summary"])
                  .collect_batches(chunk_size=1_000_000)):
        nw.write_table(chunk.to_arrow())
    nw.close()

    # Resolve edges chunk-by-chunk (bounded RAM: one chunk + the broadcast map).
    edge_schema = pa.schema([("source", pa.int64()), ("target", pa.int64())])
    ew = pq.ParquetWriter(out_edges, edge_schema)
    total = unresolved = 0
    t1 = time.time()
    for i, chunk in enumerate(pl.scan_parquet(src_edges).collect_batches(chunk_size=4_000_000)):
        e = (chunk
             .with_columns([
                 pl.col("source").cast(pl.Int64, strict=False).alias("src_num"),
                 pl.col("target").cast(pl.Int64, strict=False).alias("dst_num"),
                 pl.col("source").str.replace_all("_", " ").alias("src_norm"),
                 pl.col("target").str.replace_all("_", " ").alias("dst_norm"),
             ])
             .join(tm.rename({"title": "src_norm", "id": "src_t"}), on="src_norm", how="left")
             .join(tm.rename({"title": "dst_norm", "id": "dst_t"}), on="dst_norm", how="left")
             .with_columns([
                 pl.coalesce(["src_num", "src_t"]).alias("source"),
                 pl.coalesce(["dst_num", "dst_t"]).alias("target"),
             ])
             .filter(
                 pl.col("source").is_not_null() & pl.col("target").is_not_null()
                 & (pl.col("source") > 0) & (pl.col("target") > 0)
                 & (pl.col("source") != pl.col("target"))
             ))
        unresolved += int(chunk.height - e.height)
        if redir_set is not None:
            e = (e.join(redir_set.rename({"id": "source"}), on="source", how="anti")
                 .join(redir_set.rename({"id": "target"}), on="target", how="anti"))
        out = e.select(["source", "target"]).to_arrow()
        ew.write_table(out)
        total += out.num_rows
        log(f"  resolve edges: chunk {i+1} -> {total:,} kept ({time.time()-t1:.0f}s)")
    ew.close()
    log(f"resolve: {total:,} clean edges, {unresolved:,} dropped")


# ── 4. embed (GPU if available, streaming, resumable) ────────────────────────
def stage_embed(model: str, dims: int, batch: int, lead_chars: int = 256):
    os.makedirs(EMB, exist_ok=True)
    src = os.path.join(DATA, "wiki_nodes.parquet")
    for stale in os.listdir(EMB):  # leftovers from an interrupted run
        if stale.endswith(".tmp"):
            os.remove(os.path.join(EMB, stale))
    done_shards = sorted(f for f in os.listdir(EMB)
                         if f.startswith("part-") and f.endswith(".parquet"))
    # RESUME: shards are contiguous row-order slices, so finished ones tell us
    # exactly how many source rows are already embedded — continue from there
    # instead of redoing hours of GPU work.
    done_rows = sum(pq.ParquetFile(os.path.join(EMB, s)).metadata.num_rows
                    for s in done_shards)
    total_rows = pq.ParquetFile(src).metadata.num_rows
    if done_rows >= total_rows:
        log(f"embed: all {done_rows:,} rows embedded across {len(done_shards)} shard(s) — done")
        return
    if done_shards:
        log(f"embed: resuming — {done_rows:,}/{total_rows:,} rows already embedded "
            f"({len(done_shards)} shard(s))")

    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = device == "cuda"
    log(f"embed: model={model} device={device} dims={dims or 'auto'} on {src}")
    model_obj = SentenceTransformer(model, device=device)
    if fp16:
        model_obj = model_obj.half()

    # Truncate (MRL) + renormalize if a smaller dim is requested.
    def enc(texts):
        v = model_obj.encode(texts, batch_size=batch, convert_to_numpy=True,
                             normalize_embeddings=False, show_progress_bar=False)
        if dims and dims < v.shape[1]:
            v = v[:, :dims]
        v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
        return v.astype("float32")

    # Rotate shards every SHARD_ROWS so the load stage can delete each shard
    # after ingesting it (bounds peak disk); tmp+rename keeps files atomic.
    SHARD_ROWS = int(os.environ.get("HDB_EMBED_SHARD_ROWS", 5_000_000))
    shard = len(done_shards)          # continue numbering after finished shards
    writer = None
    cur_tmp = cur_final = None
    rows_in_shard = 0
    dim = None
    done = done_rows                  # rows already on disk (resume)
    skip = done_rows                  # source rows to fast-forward past
    t0 = time.time()
    pf = pq.ParquetFile(src)
    n_rows = pf.metadata.num_rows
    for batch_df in pl.scan_parquet(src).collect_batches(chunk_size=200_000):
        if skip > 0:                  # fast-forward over already-embedded rows
            h = batch_df.height
            if skip >= h:
                skip -= h
                continue
            batch_df = batch_df.slice(skip, h - skip)
            skip = 0
        titles = batch_df["title"].to_list()
        summaries = batch_df["summary"].to_list()
        combined = [f"{t}. {(s or '')[:lead_chars]}" for t, s in zip(titles, summaries)]
        vecs = enc(combined)
        dim = vecs.shape[1]
        tbl = pa.Table.from_arrays(
            [pa.array(batch_df["id"].to_list(), type=pa.int64()),
             pa.FixedSizeListArray.from_arrays(pa.array(vecs.reshape(-1), type=pa.float32()), dim)],
            names=["id", "embedding"])
        if writer is not None and rows_in_shard >= SHARD_ROWS:
            writer.close()
            os.replace(cur_tmp, cur_final)
            log(f"embed: {os.path.basename(cur_final)} ({rows_in_shard:,} x {dim}d)")
            shard += 1
            writer = None
        if writer is None:
            cur_final = os.path.join(EMB, f"part-{shard:03d}.parquet")
            cur_tmp = cur_final + ".tmp"
            writer = pq.ParquetWriter(cur_tmp, tbl.schema)
            rows_in_shard = 0
        writer.write_table(tbl)
        rows_in_shard += tbl.num_rows
        done += len(combined)
        if done % 1_000_000 < 200_000:
            rate = done / (time.time() - t0)
            eta = (n_rows - done) / rate / 3600 if rate else 0
            log(f"  embed {done:,}/{n_rows:,} ({rate:.0f} sent/s, ETA {eta:.1f} h)")
    if writer is not None:
        writer.close()
        os.replace(cur_tmp, cur_final)
        log(f"embed: {os.path.basename(cur_final)} ({rows_in_shard:,} x {dim}d)")
    log(f"embed: {done:,} vectors across {shard + 1} shard(s)")


# ── 5. load into persistent HyperStreamDB tables ────────────────────────────
def _table_loaded(d: str) -> bool:
    """Existence is not enough: a crashed run can leave an empty table shell
    (metadata/_wal/_manifest only) that would otherwise be 'skipped'."""
    return os.path.isdir(d) and any(
        f not in ("metadata", "_wal", "_manifest") for f in os.listdir(d))


def _delete_shards():
    if not os.path.isdir(EMB):
        return
    removed = 0
    for p in os.listdir(EMB):
        if p.endswith(".parquet"):
            os.remove(os.path.join(EMB, p))
            removed += 1
    if removed:
        log(f"load nodes: deleted {removed} embedding shard(s) (reclaim disk)")


def _load_nodes_child(quant, delete_shards, row_start, row_end):
    """Write wiki_nodes rows [row_start, row_end) into the nodes table.

    Runs in a FRESH process per chunk: the in-process HNSW/TQ8 builders do
    millions of small allocations, and glibc strands freed memory inside the
    main heap — RSS ratcheted ~2.6 GB per million rows regardless of arena
    caps, OOM-killing single-process runs around 30-40M rows. A new process
    resets the allocator high-water mark.
    """
    import hyperstreamdb as hdb

    nodes_dir = os.path.join(DB, "nodes")
    nodes_path = os.path.join(DATA, "wiki_nodes.parquet")
    total_rows = pq.ParquetFile(nodes_path).metadata.num_rows
    if row_end <= 0:
        row_end = total_rows

    parts = sorted(f for f in os.listdir(EMB) if f.endswith(".parquet")) if os.path.isdir(EMB) else []
    has_vec = bool(parts)
    # Derive schema from the ACTUAL source files so string/large_string and
    # the FixedSizeList inner field name always line up (avoids ArrowInvalid).
    schema = pq.ParquetFile(nodes_path).schema_arrow
    emb_field = None
    if has_vec:
        emb_field = pq.ParquetFile(os.path.join(EMB, parts[0])).schema_arrow.field("embedding")
        schema = schema.append(emb_field)

    if _table_loaded(nodes_dir):
        t = hdb.Table(f"file://{nodes_dir}")           # later chunks: open existing
    else:
        shutil.rmtree(nodes_dir, ignore_errors=True)   # clear crashed-run shell
        t = hdb.Table.create(f"file://{nodes_dir}", schema)
    # Index config must be applied in EVERY process: a freshly opened table does
    # not inherit it, and segments written without it are silently unindexed
    # (queries then flat-scan them — measured 197 GB read for one search).
    if has_vec:
        t.add_index("embedding", f"hnsw_{quant}" if quant != "none" else "hnsw")
    t.add_index("title", "inverted")  # BM25 -> hybrid_search (keyword+vector RRF)

    def ranged_batches(path, s, e, batch=250_000, columns=None):
        """Batches from `path` clipped to the row range [s, e)."""
        pos = 0
        for b in pq.ParquetFile(path).iter_batches(batch_size=batch, columns=columns):
            if pos + b.num_rows > s and pos < e:
                lo = max(0, s - pos)
                hi = min(b.num_rows, e - pos)
                yield b.slice(lo, hi - lo)
            pos += b.num_rows
            if pos >= e:
                break

    t0 = time.time(); n = 0
    if has_vec:
        # Embedding shards are contiguous row-order slices of wiki_nodes, so
        # the same [s, e) grid on both sides keeps batches position-aligned.
        def emb_arrays():
            pos = 0
            for p in parts:
                for b in pq.ParquetFile(os.path.join(EMB, p)).iter_batches(
                        batch_size=250_000, columns=["embedding"]):
                    arr = b["embedding"]
                    if pos + len(arr) > row_start and pos < row_end:
                        lo = max(0, row_start - pos)
                        hi = min(len(arr), row_end - pos)
                        yield arr.slice(lo, hi - lo)
                    pos += len(arr)
                    if pos >= row_end:
                        return

        emb_it = emb_arrays()
        for batch in ranged_batches(nodes_path, row_start, row_end):
            col = next(emb_it, None)
            if col is None:
                raise RuntimeError("embeddings shorter than nodes")
            assert len(col) == batch.num_rows, "node/embed alignment lost"
            out = batch.append_column(emb_field, col)
            t.write(pa.Table.from_batches([out]))
            n += batch.num_rows
            if n % 2_000_000 < 250_000:
                log(f"load nodes: {row_start + n:,}/{total_rows:,} ({time.time()-t0:.0f}s)")
    else:
        log("load nodes: no embeddings — text only (semantic search off)")
        for batch in ranged_batches(nodes_path, row_start, row_end):
            t.write(pa.Table.from_batches([batch]))
            n += batch.num_rows
    t.commit(); t.wait_for_background_tasks()
    log(f"load nodes: wrote {n:,} rows [{row_start:,}, {row_end:,}) in {time.time()-t0:.0f}s")
    if has_vec and delete_shards and row_start == 0 and row_end == total_rows:
        _delete_shards()


def stage_load(rebuild: bool, quant: str, delete_shards: bool,
               chunk_rows: int, row_start: int, row_end: int):
    # hyperstreamdb self-tunes glibc arenas (mallopt M_ARENA_MAX=2) at import;
    # the chunked design below additionally resets allocator high-water marks.
    import hyperstreamdb as hdb

    edges_dir = os.path.join(DB, "edges")
    nodes_dir = os.path.join(DB, "nodes")
    if rebuild:
        shutil.rmtree(edges_dir, ignore_errors=True)
        shutil.rmtree(nodes_dir, ignore_errors=True)
    os.makedirs(DB, exist_ok=True)

    # ── edges table + CSR graph index (single pass, cheap memory) ──
    if not _table_loaded(edges_dir):
        schema = pa.schema([("source", pa.int64()), ("target", pa.int64())])
        t = hdb.Table.create(f"file://{edges_dir}", schema)
        t.add_index("source", {"type": "graph", "src_column": "source", "dst_column": "target"})
        t0 = time.time()
        n = 0
        for batch in pq.ParquetFile(os.path.join(DATA, "wiki_edges.parquet")).iter_batches(1_000_000):
            t.write(pa.Table.from_batches([batch], schema=schema))
            n += batch.num_rows
            if n % 20_000_000 < 1_000_000:
                log(f"load edges: {n:,} ({time.time()-t0:.0f}s)")
        t.commit(); t.wait_for_background_tasks()
        log(f"load edges: {n:,} edges in {time.time()-t0:.0f}s")
    else:
        log("load edges: table exists — skipping (use --rebuild)")

    # ── child / single-process mode ──
    if chunk_rows <= 0 or row_start > 0:
        _load_nodes_child(quant, delete_shards, row_start, row_end)
        return

    # ── nodes: parent orchestrates fresh-process chunks ──
    if _table_loaded(nodes_dir):
        log("load nodes: table exists — skipping (use --rebuild)")
        return
    shutil.rmtree(nodes_dir, ignore_errors=True)  # clear crashed-run shell
    total = pq.ParquetFile(os.path.join(DATA, "wiki_nodes.parquet")).metadata.num_rows
    s = 0
    while s < total:
        e = min(s + chunk_rows, total)
        log(f"load nodes: chunk [{s:,}, {e:,}) in a fresh process")
        rc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--stage", "load",
             "--load-chunk-rows", "0", "--row-start", str(s), "--row-end", str(e),
             "--quant", quant, "--keep-shards"])
        if rc.returncode != 0:
            raise SystemExit(f"load chunk [{s:,}, {e:,}) failed (rc={rc.returncode})")
        s = e
    if delete_shards:
        _delete_shards()
    log(f"load nodes: all chunks complete ({total:,} rows)")


def _count_segments(nodes_dir: str) -> int:
    if not os.path.isdir(nodes_dir):
        return 0
    return len([f for f in os.listdir(nodes_dir)
                if f.endswith(".parquet")
                and not any(x in f for x in ("title", "centroids", "tq8", ".inv."))])


def stage_compact(min_file_size_bytes: int = 2_000_000_000):
    """Final step: compact the nodes table into fewer, larger indexed segments.

    Chunked loads create many small segments and every query fans out over all
    of them. Compaction rewrites them into ~2x-min-size segments and rebuilds
    their vector/inverted indexes (the engine carries the table's index config
    into the compactor; the config itself is restored from the manifest on open).
    """
    import hyperstreamdb as hdb

    nodes_dir = os.path.join(DB, "nodes")
    if not _table_loaded(nodes_dir):
        log("compact: nodes table missing — run --stage load first")
        return
    before = _count_segments(nodes_dir)
    log(f"compact: {before} segments -> target {min_file_size_bytes * 2 / 1e9:.1f} GB "
        f"(min {min_file_size_bytes / 1e9:.1f} GB)")
    t = hdb.Table(f"file://{nodes_dir}")
    t0 = time.time()
    t.rewrite_data_files(min_file_size_bytes)
    after = _count_segments(nodes_dir)
    log(f"compact: {before} -> {after} segments in {time.time()-t0:.0f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["download", "parse", "resolve", "embed", "load", "compact", "all"], default="all")
    ap.add_argument("--workers", type=int, default=3, help="download parallelism")
    ap.add_argument("--embed-model", default="all-MiniLM-L6-v2",
                    help="384-d centroid embedder (fast+small). bge-small-en-v1.5 (384d) or "
                         "BAAI/bge-large-en-v1.5 (1024d, ~4x slower/bigger) are options")
    ap.add_argument("--embed-dims", type=int, default=0, help="0=native; 512/256 = MRL truncate (bge-large/small support this)")
    ap.add_argument("--embed-batch", type=int, default=512)
    ap.add_argument("--lead-chars", type=int, default=256, help="embed only the article lead (chars) for the centroid")
    ap.add_argument("--quant", choices=["tq4", "tq8", "none"], default="tq8")
    ap.add_argument("--keep-shards", action="store_true", help="don't delete embeddings after load")
    ap.add_argument("--rebuild", action="store_true", help="force-rebuild hdb tables in stage load")
    ap.add_argument("--load-chunk-rows", type=int, default=None,
                    help="rows per fresh-process load chunk (resets allocator high-water marks; "
                         "index builds strand ~2.6-4.5 GB freed-but-unreturnable memory per "
                         "million rows otherwise). Default: auto from available RAM (~half of "
                         "MemAvailable at 4.5 GB/M rows, clamped 250k..10M). -1 = single process")
    ap.add_argument("--row-start", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--row-end", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--compact-min-bytes", type=int, default=2_000_000_000,
                    help="compaction candidate threshold; output segments target 2x this")
    args = ap.parse_args()

    stages = (["download", "parse", "resolve", "embed", "load", "compact"]
              if args.stage == "all" else [args.stage])
    for s in stages:
        log(f"=== stage {s} ===")
        if s == "download":
            stage_download(args.workers)
        elif s == "parse":
            stage_parse()
        elif s == "resolve":
            stage_resolve()
        elif s == "embed":
            stage_embed(args.embed_model, args.embed_dims, args.embed_batch, args.lead_chars)
        elif s == "load":
            # precedence: --load-chunk-rows flag > HDB_LOAD_CHUNK_ROWS env > RAM auto-size
            chunk = args.load_chunk_rows
            if chunk is None:
                env = os.environ.get("HDB_LOAD_CHUNK_ROWS", "").strip()
                chunk = int(env) if env else _auto_chunk_rows()
            if not args.row_start:  # parent only (children pass explicit ranges)
                inputs = os.path.getsize(os.path.join(DATA, "wiki_nodes.parquet"))
                if os.path.isdir(EMB):
                    inputs += sum(os.path.getsize(os.path.join(EMB, f))
                                  for f in os.listdir(EMB) if f.endswith(".parquet"))
                _disk_guard(inputs)
            stage_load(args.rebuild, args.quant, not args.keep_shards,
                       chunk, args.row_start, args.row_end)
        elif s == "compact":
            stage_compact(args.compact_min_bytes)
    log("requested stages complete.")


if __name__ == "__main__":
    main()
