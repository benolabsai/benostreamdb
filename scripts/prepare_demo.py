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
    done_shards = sorted(f for f in os.listdir(EMB)
                         if f.startswith("part-") and f.endswith(".parquet"))
    if done_shards:
        log(f"embed: {len(done_shards)} shard(s) present — skipping (delete data/embeddings/ to redo)")
        return
    for stale in os.listdir(EMB):  # leftovers from a crashed run
        if stale.endswith(".tmp"):
            os.remove(os.path.join(EMB, stale))

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
    shard = 0
    writer = None
    cur_tmp = cur_final = None
    rows_in_shard = 0
    dim = None
    done = 0
    t0 = time.time()
    pf = pq.ParquetFile(src)
    n_rows = pf.metadata.num_rows
    for batch_df in pl.scan_parquet(src).collect_batches(chunk_size=200_000):
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


def stage_load(rebuild: bool, quant: str, delete_shards: bool):
    import hyperstreamdb as hdb

    edges_dir = os.path.join(DB, "edges")
    nodes_dir = os.path.join(DB, "nodes")
    if rebuild:
        shutil.rmtree(edges_dir, ignore_errors=True)
        shutil.rmtree(nodes_dir, ignore_errors=True)
    os.makedirs(DB, exist_ok=True)

    # ── edges table + CSR graph index ──
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

    # ── nodes table + HNSW(quant) vector index + BM25 title index ──
    if not _table_loaded(nodes_dir):
        shutil.rmtree(nodes_dir, ignore_errors=True)  # clear crashed-run shell
        parts = sorted(f for f in os.listdir(EMB) if f.endswith(".parquet")) if os.path.isdir(EMB) else []
        has_vec = bool(parts)
        nodes_path = os.path.join(DATA, "wiki_nodes.parquet")
        # Derive schema from the ACTUAL source files so string/large_string and
        # the FixedSizeList inner field name always line up (avoids ArrowInvalid).
        schema = pq.ParquetFile(nodes_path).schema_arrow
        emb_field = None
        if has_vec:
            emb_field = pq.ParquetFile(os.path.join(EMB, parts[0])).schema_arrow.field("embedding")
            schema = schema.append(emb_field)
        t = hdb.Table.create(f"file://{nodes_dir}", schema)
        if has_vec:
            t.add_index("embedding", f"hnsw_{quant}" if quant != "none" else "hnsw")
        t.add_index("title", "inverted")  # BM25 -> hybrid_search (keyword+vector RRF)
        t0 = time.time(); n = 0

        if has_vec:
            # Embedding shards are contiguous row-order slices of wiki_nodes, so
            # zip by position — STREAMING batches (a whole-shard read_table OOM-
            # killed the 37 GB single-shard run). Shards are deleted only AFTER
            # the table commits, so a crash never destroys unrecoverable work.
            def emb_batches():
                for p in parts:
                    path = os.path.join(EMB, p)
                    for b in pq.ParquetFile(path).iter_batches(batch_size=250_000):
                        yield b["embedding"]

            emb_iter = emb_batches()
            emb_cur = None  # current embeddings batch (ChunkedArray column)
            emb_off = 0
            for batch in pq.ParquetFile(nodes_path).iter_batches(batch_size=250_000):
                pieces = []; need = batch.num_rows
                while need > 0:
                    if emb_cur is None or emb_off >= len(emb_cur):
                        try:
                            emb_cur = next(emb_iter)
                        except StopIteration:
                            raise RuntimeError("embeddings shorter than nodes")
                        emb_off = 0
                        continue
                    take = min(need, len(emb_cur) - emb_off)
                    pieces.append(emb_cur.slice(emb_off, take))  # Array (batch col)
                    emb_off += take; need -= take
                col = pieces[0] if len(pieces) == 1 else pa.concat_arrays(pieces)
                out = batch.append_column(emb_field, col)
                t.write(pa.Table.from_batches([out]))
                n += batch.num_rows
                if n % 2_000_000 < 250_000:
                    log(f"load nodes: {n:,} ({time.time()-t0:.0f}s)")
        else:
            log("load nodes: no embeddings — text only (semantic search off)")
            for batch in pq.ParquetFile(nodes_path).iter_batches(250_000):
                t.write(pa.Table.from_batches([batch]))
                n += batch.num_rows
        t.commit(); t.wait_for_background_tasks()
        log(f"load nodes: {n:,} pages in {time.time()-t0:.0f}s")
        if has_vec and delete_shards:
            for p in parts:
                fp = os.path.join(EMB, p)
                if os.path.exists(fp):
                    os.remove(fp)
            log("load nodes: deleted embedding shards (reclaim disk)")
    else:
        log("load nodes: table exists — skipping (use --rebuild)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["download", "parse", "resolve", "embed", "load", "all"], default="all")
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
    args = ap.parse_args()

    stages = ["download", "parse", "resolve", "embed", "load"] if args.stage == "all" else [args.stage]
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
            stage_load(args.rebuild, args.quant, not args.keep_shards)
    log("requested stages complete.")


if __name__ == "__main__":
    main()
