#!/usr/bin/env python3
"""Prepare the FULL English-Wikipedia Graph-RAG demo dataset for HyperStreamDB.

No pruning: the whole site goes in. Every stage is idempotent and resumable —
rerun the script any time and it skips finished work.

Stages:
  1. download  - fetch all enwiki pages-meta-current chunks (scripts/download_wiki_dumps.py)
  2. parse     - XML -> data_full/{nodes,edges}.parquet (Rust: ingest_wikipedia)
  3. resolve   - polars lazy join: mixed curid/title endpoints -> int64 ids,
                 redirect pages dropped -> data/wiki_{nodes,edges}.parquet
  4. embed     - parallel, sharded, resumable sentence-transformers pass over
                 all article summaries -> data/embeddings/part-*.parquet
  5. load      - build the persistent HyperStreamDB tables the web UI opens:
                 data/wiki_graph_db/edges  (CSR graph index)
                 data/wiki_graph_db/nodes  (HNSW + TurboQuant-8 vector index)

Usage:
  python scripts/prepare_demo.py                # everything
  python scripts/prepare_demo.py --stage load   # one stage
  python scripts/prepare_demo.py --embed-model BAAI/bge-large-en-v1.5
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

import polars as pl

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data")
DB = os.path.join(DATA, "wiki_graph_db")   # persistent hdb tables
EMB = os.path.join(DATA, "embeddings")

log = lambda m: print(f"[prepare] {m}", flush=True)


def _full_dir() -> str:
    """Parser output dir: prefer data_full (what the chain script writes), then data/full."""
    for d in (os.path.join(REPO, "data_full"), os.path.join(DATA, "full")):
        if os.path.exists(os.path.join(d, "nodes.parquet")):
            return d
    return os.path.join(REPO, "data_full")


def _run(cmd, **kw):
    log("$ " + " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, cwd=REPO, check=True, **kw)


# ── 1. download ─────────────────────────────────────────────────────────────
def stage_download(workers: int):
    _run([sys.executable, "scripts/download_wiki_dumps.py", "--workers", str(workers)])


# ── 2. parse ────────────────────────────────────────────────────────────────
def stage_parse():
    full = _full_dir()
    out_nodes = os.path.join(full, "nodes.parquet")
    out_edges = os.path.join(full, "edges.parquet")
    if os.path.exists(out_nodes) and os.path.exists(out_edges):
        log(f"parse: already done ({out_nodes}) — skipping")
        return
    _run(["cargo", "run", "--release", "--bin", "ingest_wikipedia", "--",
          "--input-dir", DATA, "--output-dir", full])


# ── 3. resolve (polars) ─────────────────────────────────────────────────────
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
    nodes = (
        pl.scan_parquet(src_nodes)
        .with_columns(pl.col("id").cast(pl.Int64, strict=False).alias("id"))
        .with_columns(
            pl.col("summary").str.strip_chars_start().str.starts_with("#REDIRECT").fill_null(False).alias("is_redirect")
        )
    )
    live = nodes.filter(~pl.col("is_redirect")).select(["id", "title", "summary"])
    redir = nodes.filter(pl.col("is_redirect")).select("id")
    log(f"resolve: nodes scanned in {time.time() - t0:.0f}s")

    t1 = time.time()
    # Single title->curid map covering both casing variants (exact title first
    # so exact matches win); keeps the edge plan to 2 joins, streaming-friendly.
    tm = pl.concat([
        live.select(["title", "id"]),
        live.select(
            (pl.col("title").str.slice(0, 1).str.to_lowercase() + pl.col("title").str.slice(1)).alias("title"),
            pl.col("id"),
        ),
    ]).unique(subset=["title"], keep="first")

    edges = (
        pl.scan_parquet(src_edges)
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
        )
        .join(redir.rename({"id": "source"}), on="source", how="anti")
        .join(redir.rename({"id": "target"}), on="target", how="anti")
        .select(["source", "target"])
    )
    edges.sink_parquet(out_edges)
    live.select(["id", "title", "summary"]).sink_parquet(out_nodes)
    log(f"resolve: joins + sink done in {time.time() - t1:.0f}s")

    n_e = pl.scan_parquet(out_edges).select(pl.len()).collect().item()
    n_n = pl.scan_parquet(out_nodes).select(pl.len()).collect().item()
    log(f"resolve: {n_n:,} live pages, {n_e:,} clean int64 edges")


# ── 4. embed (parallel, sharded, resumable) ─────────────────────────────────
_EMBED = {}


def _embed_worker(args):
    idx, total, model_name, out_path = args
    if os.path.exists(out_path):
        return (idx, "cached")
    import os as _os
    import torch
    torch.set_num_threads(max(2, (_os.cpu_count() or 8) // max(1, total)))
    from sentence_transformers import SentenceTransformer
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    model = _EMBED.get(model_name)
    if model is None:
        model = SentenceTransformer(model_name, device="cpu")
        _EMBED[model_name] = model

    # Deterministic slice per shard.
    n_rows = pq.ParquetFile(os.path.join(DATA, "wiki_nodes.parquet")).metadata.num_rows
    lo, hi = idx * n_rows // total, (idx + 1) * n_rows // total
    df = (
        pl.scan_parquet(os.path.join(DATA, "wiki_nodes.parquet"))
        .slice(lo, hi - lo)
        .collect(streaming=True)
    )
    ids = df["id"].to_numpy()
    titles = df["title"].to_list()
    summaries = df["summary"].to_list()
    del df

    # Encode + write in chunks: a single encode() of millions of texts would
    # materialize a ~13GB float32 array per worker (OOM with several workers).
    tmp = out_path + ".tmp"
    writer = None
    dim = None
    done = 0
    CH = 100_000
    for lo in range(0, len(ids), CH):
        hi = min(lo + CH, len(ids))
        combined = [f"{titles[i]}. {(summaries[i] or '')[:512]}" for i in range(lo, hi)]
        vecs = model.encode(combined, batch_size=512, show_progress_bar=False,
                            convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        dim = vecs.shape[1]
        offsets = pa.array(np.arange(0, (hi - lo + 1) * dim, dim, dtype=np.int64))
        tbl = pa.Table.from_arrays(
            [pa.array(ids[lo:hi], type=pa.int64()),
             pa.LargeListArray.from_arrays(offsets, pa.array(vecs.reshape(-1), type=pa.float32()))],
            names=["id", "embedding"],
        )
        if writer is None:
            writer = pq.ParquetWriter(tmp, tbl.schema)
        writer.write_table(tbl)
        done += hi - lo
        if done % 1_000_000 == 0:
            log(f"embed: shard {idx} at {done / 1e6:.0f}M vectors")
    if writer is not None:
        writer.close()
    os.replace(tmp, out_path)
    return (idx, f"{done:,} vectors x {dim}d")


def stage_embed(workers: int, model: str):
    os.makedirs(EMB, exist_ok=True)
    parts = [os.path.join(EMB, f"part-{i:03d}.parquet") for i in range(workers)]
    todo = [(i, workers, model, p) for i, p in enumerate(parts) if not os.path.exists(p)]
    if not todo:
        log("embed: all shards present — skipping")
        return
    import multiprocessing as mp
    log(f"embed: {len(todo)}/{workers} shards to go with {model}")
    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        for idx, msg in pool.imap_unordered(_embed_worker, todo):
            log(f"embed: shard {idx} -> {msg} ({time.time() - t0:.0f}s elapsed)")
    log("embed: complete")


# ── 5. load into persistent HyperStreamDB tables ────────────────────────────
def stage_load(rebuild: bool):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import hyperstreamdb as hdb

    edges_dir = os.path.join(DB, "edges")
    nodes_dir = os.path.join(DB, "nodes")
    if rebuild:
        shutil.rmtree(edges_dir, ignore_errors=True)
        shutil.rmtree(nodes_dir, ignore_errors=True)

    # ── edges table + CSR graph index ──
    if not os.path.exists(edges_dir):
        os.makedirs(os.path.dirname(edges_dir), exist_ok=True)
        schema = pa.schema([("source", pa.int64()), ("target", pa.int64())])
        t = hdb.Table.create(f"file://{edges_dir}", schema)
        t.add_index("source", {"type": "graph", "src_column": "source", "dst_column": "target"})
        t0 = time.time()
        pf = pq.ParquetFile(os.path.join(DATA, "wiki_edges.parquet"))
        n = 0
        for batch in pf.iter_batches(batch_size=1_000_000):
            t.write(pa.Table.from_batches([batch]))
            n += batch.num_rows
            if n % 20_000_000 < 1_000_000:
                log(f"load edges: {n:,} ({time.time() - t0:.0f}s)")
        t.commit()
        t.wait_for_background_tasks()
        log(f"load edges: done {n:,} edges in {time.time() - t0:.0f}s")
    else:
        log("load edges: table exists — skipping (use --rebuild)")

    # ── nodes table + TQ8 HNSW vector index ──
    if not os.path.exists(nodes_dir):
        os.makedirs(os.path.dirname(nodes_dir), exist_ok=True)
        parts = sorted(f for f in os.listdir(EMB) if f.endswith(".parquet")) if os.path.isdir(EMB) else []
        has_vec = bool(parts)
        if has_vec:
            n_nodes = pq.ParquetFile(os.path.join(DATA, "wiki_nodes.parquet")).metadata.num_rows
            n_emb = sum(pq.ParquetFile(os.path.join(EMB, p)).metadata.num_rows for p in parts)
            if n_nodes != n_emb:
                raise RuntimeError(
                    f"embeddings out of sync with nodes ({n_emb:,} vs {n_nodes:,}) — "
                    "delete data/embeddings/ and rerun stage embed"
                )
        fields = [("id", pa.int64()), ("title", pa.large_string()), ("summary", pa.large_string())]
        if has_vec:
            fields.append(("embedding", pa.large_list(pa.float32())))
        schema = pa.schema(fields)
        t = hdb.Table.create(f"file://{nodes_dir}", schema)
        if has_vec:
            t.add_index("embedding", "hnsw_tq8")
        # BM25 inverted index on titles -> enables hybrid_search (keyword+vector RRF)
        t.add_index("title", "inverted")
        t0 = time.time()
        n = 0

        if has_vec:
            # Embedding shards are contiguous slices of wiki_nodes.parquet in
            # row order, so zip by position — a hash join on 66M x 384d
            # vectors would need ~100GB of build-side memory.
            def emb_batches():
                for p in parts:
                    for b in pq.ParquetFile(os.path.join(EMB, p)).iter_batches(batch_size=250_000):
                        yield b

            emb_iter = emb_batches()
            emb_buf = None
            emb_off = 0
            for batch in pq.ParquetFile(os.path.join(DATA, "wiki_nodes.parquet")).iter_batches(batch_size=250_000):
                rows = batch.num_rows
                pieces = []
                need = rows
                while need > 0:
                    if emb_buf is None or emb_off >= emb_buf.num_rows:
                        emb_buf = next(emb_iter, None)
                        emb_off = 0
                        if emb_buf is None:
                            raise RuntimeError("embeddings shorter than nodes — rerun stage embed")
                        if emb_buf.num_rows == 0:
                            continue
                    take = min(need, emb_buf.num_rows - emb_off)
                    pieces.append(emb_buf.slice(emb_off, take).column("embedding"))
                    emb_off += take
                    need -= take
                col = pieces[0] if len(pieces) == 1 else pa.concat_arrays(pieces)
                out = batch.append_column("embedding", col)
                t.write(pa.Table.from_batches([out], schema=schema))
                n += rows
                if n % 2_000_000 < 250_000:
                    log(f"load nodes: {n:,} ({time.time() - t0:.0f}s)")
        else:
            log("load nodes: no embeddings found — loading text only (semantic search unavailable)")
            for batch in pq.ParquetFile(os.path.join(DATA, "wiki_nodes.parquet")).iter_batches(batch_size=250_000):
                t.write(pa.Table.from_batches([batch], schema=schema))
                n += batch.num_rows
                if n % 2_000_000 < 250_000:
                    log(f"load nodes: {n:,} ({time.time() - t0:.0f}s)")

        t.commit()
        t.wait_for_background_tasks()
        log(f"load nodes: done {n:,} pages in {time.time() - t0:.0f}s")
    else:
        log("load nodes: table exists — skipping (use --rebuild)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["download", "parse", "resolve", "embed", "load", "all"], default="all")
    ap.add_argument("--workers", type=int, default=4, help="parallel processes for download/embed")
    ap.add_argument("--embed-model", default="all-MiniLM-L6-v2",
                    help="'all-MiniLM-L6-v2' (384d, fast) or 'BAAI/bge-large-en-v1.5' (1024d, ~10x slower)")
    ap.add_argument("--rebuild", action="store_true", help="force-rebuild the hdb tables in stage load")
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
            stage_embed(args.workers, args.embed_model)
        elif s == "load":
            stage_load(args.rebuild)
    log("all requested stages complete.")


if __name__ == "__main__":
    main()
