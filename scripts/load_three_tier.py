#!/usr/bin/env python3
"""Load the 3-tier Wikipedia dataset into BenoStreamDB with edge tables.

Implements three JanusGraph/Dgraph-inspired optimizations:
  1. Vertex-centric indexing — sections modeled as an edge table
     (article --[has_section]--> section), leveraging BenoStreamDB's
     CSR graph index for O(1) article→sections traversal.
  2. Bulk pre-sort — data sorted by join key before writing for
     sequential I/O and better compression.
  3. Index preload — warm all caches (HNSW, CSR, inverted) at load
     time so first query is instant (Dgraph-style in-memory indexes).

Schema:
  articles       — node table  (id, page_id, title)
  sections       — edge table  (source=article_id, target=section_id,
                                section_title, level)
  section_texts  — node table  (section_id, text)
  section_embeds — node table  (section_id, embedding)  [HNSW indexed]

Usage:
  python scripts/load_three_tier.py --data-dir data/three_tier
"""

import argparse
import os
import sys
import time

import numpy as np
import polars as pl
import benostreamdb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/three_tier")
    ap.add_argument("--table-dir", default="data/three_tier_db")
    ap.add_argument("--batch-size", type=int, default=100_000)
    ap.add_argument("--no-index", action="store_true",
                    help="Skip HNSW index build (faster load, query later)")
    ap.add_argument("--no-preload", action="store_true",
                    help="Skip index cache warming")
    args = ap.parse_args()

    os.makedirs(args.table_dir, exist_ok=True)

    # ── GPU activation ───────────────────────────────────────────────
    try:
        gpu = benostreamdb.Device.auto_detect()
        gpu.activate()
        print(f"[load] GPU: {gpu}", flush=True)
    except Exception:
        print("[load] GPU not available, using CPU", flush=True)

    t_start = time.time()

    # ── 1. Articles (node table) ─────────────────────────────────────
    print("[load] articles ...", flush=True)
    articles = pl.read_parquet(os.path.join(args.data_dir, "articles.parquet"))
    # Pre-sort by id for sequential I/O  ← JanusGraph bulk-load pattern
    articles = articles.sort("id")
    print(f"  {len(articles):,} rows", flush=True)

    articles_uri = f"file://{os.path.abspath(args.table_dir)}/articles"
    art_table = benostreamdb.Table.create(
        articles_uri,
        {"id": "int64", "page_id": "large_utf8", "title": "large_utf8"},
    )
    _write_sorted(art_table, articles, "articles", args.batch_size)

    # ── 2. Sections (edge table) ─────────────────────────────────────
    # Vertex-centric index via CSR: article --[has_section]--> section
    print("[load] sections (edge table) ...", flush=True)
    sections = pl.read_parquet(os.path.join(args.data_dir, "sections.parquet"))
    # Pre-sort by source (article_id) so all sections of one article
    # are contiguous on disk — sequential scan per article.
    sections = sections.sort("article_id")
    print(f"  {len(sections):,} edges", flush=True)

    sections_uri = f"file://{os.path.abspath(args.table_dir)}/sections"
    sec_table = benostreamdb.Table.create_edge_table(
        sections_uri,
        {
            "source": "int64",
            "target": "int64",
            "section_title": "large_utf8",
            "level": "int32",
        },
    )
    # Edge table convention: source/target columns are auto-detected
    sec_df = sections.select([
        pl.col("article_id").alias("source"),
        pl.col("id").alias("target"),
        pl.col("section_title"),
        pl.col("level"),
    ])
    _write_sorted(sec_table, sec_df, "sections", args.batch_size)

    # ── 3. Section texts (node table) ────────────────────────────────
    print("[load] section_texts ...", flush=True)
    texts = pl.read_parquet(os.path.join(args.data_dir, "section_texts.parquet"))
    texts = texts.sort("section_id")
    print(f"  {len(texts):,} rows", flush=True)

    texts_uri = f"file://{os.path.abspath(args.table_dir)}/section_texts"
    txt_table = benostreamdb.Table.create(
        texts_uri,
        {"section_id": "int64", "text": "large_utf8"},
    )
    _write_sorted(txt_table, texts, "section_texts", args.batch_size)

    # ── 4. Section embeddings (node table) ───────────────────────────
    embeds_path = os.path.join(args.data_dir, "section_embeddings.parquet")
    emb_table = None
    if os.path.exists(embeds_path):
        print("[load] section_embeddings ...", flush=True)
        embeds = pl.read_parquet(embeds_path)
        embeds = embeds.sort("section_id")
        print(f"  {len(embeds):,} rows", flush=True)

        embeds_uri = f"file://{os.path.abspath(args.table_dir)}/section_embeds"
        emb_table = benostreamdb.Table.create(
            embeds_uri,
            {"section_id": "int64", "embedding": "list<float32>"},
        )
        _write_sorted(emb_table, embeds, "embeddings", args.batch_size)

        # ── HNSW index ───────────────────────────────────────────────
        if not args.no_index:
            print("[load] building HNSW index on embeddings ...", flush=True)
            t_idx = time.time()
            emb_table.add_index("embedding", algorithm="hnsw")
            print(f"  index built in {(time.time() - t_idx)/60:.1f}m", flush=True)
    else:
        print("[load] no embeddings file yet — skipping", flush=True)

    elapsed = time.time() - t_start
    print(f"\n[load] data loaded in {elapsed/60:.1f}m", flush=True)

    # ── 5. Preload indexes (Dgraph-style cache warming) ──────────────
    if not args.no_preload:
        print("\n[preload] warming caches ...", flush=True)
        t_pre = time.time()
        _preload_indexes(art_table, sec_table, txt_table, emb_table, articles)
        print(f"[preload] caches warm in {(time.time() - t_pre):.1f}s", flush=True)

    # ── Verify ───────────────────────────────────────────────────────
    print("\n[load] verification:", flush=True)
    for name, table in [("articles", art_table), ("sections", sec_table),
                         ("section_texts", txt_table)]:
        print(f"  {name}: {len(table):,} rows", flush=True)
    if emb_table is not None:
        print(f"  section_embeds: {len(emb_table):,} rows", flush=True)

    # Quick CSR edge traversal test
    print("\n[load] testing CSR edge traversal ...", flush=True)
    sample_article = articles["id"][0]
    neighbors = sec_table.graph_neighbors(sample_article)
    print(f"  article {sample_article} has {len(neighbors)} sections (CSR lookup)",
          flush=True)

    total = time.time() - t_start
    print(f"\n[load] all done in {total/60:.1f}m", flush=True)


def _write_sorted(table, df, name, batch_size):
    """Write a pre-sorted DataFrame to a BenoStreamDB table in batches."""
    total = len(df)
    t0 = time.time()
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        chunk = df.slice(start, end - start)
        table.write(chunk)
        pct = end * 100 / total
        elapsed = time.time() - t0
        rate = end / elapsed if elapsed > 0 else 0
        print(f"  [{name}] {end:,}/{total:,} ({pct:.0f}%)  "
              f"{rate:.0f} rows/s", flush=True)


def _preload_indexes(art_table, sec_table, txt_table, emb_table, articles):
    """Warm all caches so first user query is instant.

    Dgraph keeps posting-list indexes in RAM.  We simulate that by
    touching every index path once so moka caches are populated.
    """
    dims = 384  # all-MiniLM-L6-v2

    # ── CSR graph index ────────────────────────────────────────────
    # Touch a few random articles to pull their edge lists into cache.
    print("[preload] CSR graph index ...", flush=True)
    rng = np.random.default_rng(42)
    sample_ids = rng.choice(articles["id"].to_list(), size=min(100, len(articles)),
                            replace=False)
    for aid in sample_ids:
        try:
            _ = sec_table.graph_neighbors(int(aid))
        except Exception:
            pass  # article may have no sections
    print(f"[preload]   touched {len(sample_ids)} articles", flush=True)

    # ── HNSW vector index ──────────────────────────────────────────
    if emb_table is not None:
        print("[preload] HNSW index ...", flush=True)
        try:
            # A single vector search pulls the HNSW graph into cache
            dummy = [0.0] * dims
            _ = emb_table.vector_search("embedding", dummy, k=1)
            print("[preload]   HNSW cache warm", flush=True)
        except Exception as e:
            print(f"[preload]   HNSW warmup skipped: {e}", flush=True)

    # ── Inverted / text index ──────────────────────────────────────
    print("[preload] text index ...", flush=True)
    try:
        # A keyword search pulls the inverted index into cache
        _ = txt_table.filter("text LIKE '%the%'").to_pandas()
        print("[preload]   inverted index cache warm", flush=True)
    except Exception as e:
        print(f"[preload]   text warmup skipped: {e}", flush=True)

    # ── Manifest / metadata caches ─────────────────────────────────
    print("[preload] metadata caches ...", flush=True)
    _ = len(art_table)  # forces manifest load + version cache
    _ = len(sec_table)
    _ = len(txt_table)
    print("[preload]   manifest caches warm", flush=True)


if __name__ == "__main__":
    main()