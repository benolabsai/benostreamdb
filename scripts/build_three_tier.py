#!/usr/bin/env python3
"""End-to-end 3-tier Wikipedia Graph-RAG pipeline (single reproducible script).

Tiers:
  Tier 1 (articles)     — id, page_id, title
  Tier 2 (sections)     — id, article_id, section_title, level
  Tier 3 (section_text) — id, section_id, text   (raw wikitext slice)

A page with **no** `== Section ==` headers is treated as a **single section**
covering the whole page (`start_byte = 0`, `end_byte = -1` sentinel = to EOF).

Stages (each is idempotent and skipped when its output already exists):
  1. download  — fetch the enwiki `mediawiki_content_current` XML chunks
  2. parse     — `ingest_wikipedia --sections --full-text` (N-way parallel)
  3. merge     — concatenate the per-worker parquets
  4. tier      — build articles / sections / section_texts
  5. embed     — GPU SentenceTransformer embeddings over section text
  6. load      — load into BenoStreamDB with an edge table (CSR) + HNSW,
                 then preload the index caches

Usage:
  python scripts/build_three_tier.py                    # run all stages
  python scripts/build_three_tier.py --stages tier      # just one stage
  python scripts/build_three_tier.py --stages download,parse
  python scripts/build_three_tier.py --data-dir ~/data/benostreamdb

Everything is driven by flags/env; no manual stitching required.
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_DUMP_DATE = "2026-09-01"
DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_EMBED_DIMS = 384
DEFAULT_UA = "BenoStreamDB/1.0 (https://github.com/benostreamdb) Python-urllib/3.0"

ALL_STAGES = ["download", "parse", "merge", "tier", "embed", "load"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[3tier] {msg}", flush=True)


def _run(cmd, **kw) -> int:
    """Run a subprocess, streaming nothing (caller logs), return exit code."""
    return subprocess.call(cmd, **kw)


# ---------------------------------------------------------------------------
# Stage 1: download
# ---------------------------------------------------------------------------

def _download_file(url: str, filepath: str, retries: int = 10) -> bool:
    name = os.path.basename(filepath)
    if os.path.exists(filepath):
        log(f"download: skip {name} (present)")
        return True

    part = filepath + ".part"
    pos = os.path.getsize(part) if os.path.exists(part) else 0

    for attempt in range(1, retries + 1):
        try:
            headers = {"User-Agent": DEFAULT_UA}
            if pos > 0:
                headers["Range"] = f"bytes={pos}-"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as response:
                resumed = response.status == 206 and pos > 0
                if not resumed:
                    pos = 0
                remaining = int(response.getheader("Content-Length") or 0)
                total = pos + remaining
                mode = "ab" if resumed else "wb"
                with open(part, mode) as out:
                    while True:
                        buf = response.read(1 << 20)
                        if not buf:
                            break
                        out.write(buf)
                        pos += len(buf)
            if total and pos < total:
                raise IOError(f"connection closed early ({pos}/{total})")
            os.replace(part, filepath)
            log(f"download: ok {name} ({pos / 1e9:.2f} GB)")
            return True
        except Exception as e:  # noqa: BLE001
            status = getattr(e, "code", None)
            log(f"download: {name} attempt {attempt} failed: {e}")
            if attempt < retries:
                time.sleep((60 if status == 429 else 5) * attempt)
    log(f"download: FAILED {name} (kept .part for resume)")
    return False


def stage_download(args) -> None:
    import re

    base_url = (
        f"https://dumps.wikimedia.org/other/mediawiki_content_current/"
        f"enwiki/{args.dump_date}/xml/bzip2/"
    )
    log(f"download: fetching index {base_url}")
    req = urllib.request.Request(base_url, headers={"User-Agent": DEFAULT_UA})
    with urllib.request.urlopen(req, timeout=60) as response:
        html = response.read().decode("utf-8")

    pattern = re.compile(rf'href="(enwiki-{args.dump_date}-p\d+p\d+\.xml\.bz2)"')
    files = list(dict.fromkeys(pattern.findall(html)))
    if not files:
        raise RuntimeError(f"no dump chunks found at {base_url}")

    os.makedirs(args.data_dir, exist_ok=True)
    present = [f for f in files if os.path.exists(os.path.join(args.data_dir, f))]
    missing = [f for f in files if f not in present]
    log(f"download: {len(files)} chunks, {len(present)} present, {len(missing)} to fetch")
    if not missing:
        return

    jobs = [(base_url + f, os.path.join(args.data_dir, f)) for f in missing]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda j: _download_file(*j), jobs))
    if not all(results):
        raise RuntimeError("download incomplete; rerun to resume")


# ---------------------------------------------------------------------------
# Stage 2: parse (N-way parallel)
# ---------------------------------------------------------------------------

def _ensure_ingest_binary(repo_root: str) -> str:
    binary = os.path.join(repo_root, "target", "release", "ingest_wikipedia")
    if not os.path.exists(binary):
        log("parse: building ingest_wikipedia (release)")
        rc = _run(["cargo", "build", "--release", "--bin", "ingest_wikipedia"], cwd=repo_root)
        if rc != 0:
            raise RuntimeError("failed to build ingest_wikipedia")
    return binary


def stage_parse(args, repo_root: str) -> None:
    binary = _ensure_ingest_binary(repo_root)

    merged = os.path.join(args.data_dir, "sections.parquet")
    if os.path.exists(merged):
        log("parse: merged outputs present; skipping")
        return

    bz2s = sorted(
        f for f in os.listdir(args.data_dir) if f.endswith(".xml.bz2")
    )
    if not bz2s:
        raise RuntimeError(f"no .xml.bz2 chunks in {args.data_dir}")

    workers = max(1, min(args.workers, len(bz2s)))
    # Distribute chunks round-robin into worker subdirs (symlinks).
    worker_dirs = []
    for w in range(workers):
        wdir = os.path.join(args.data_dir, f"worker_{w}")
        os.makedirs(wdir, exist_ok=True)
        # Clear stale outputs, keep symlinks
        for name in ("nodes.parquet", "edges.parquet", "sections.parquet"):
            p = os.path.join(wdir, name)
            if os.path.exists(p):
                os.remove(p)
        worker_dirs.append(wdir)
    for i, f in enumerate(bz2s):
        wdir = worker_dirs[i % workers]
        link = os.path.join(wdir, f)
        if not os.path.lexists(link):
            os.symlink(os.path.join(args.data_dir, f), link)

    log(f"parse: {len(bz2s)} chunks across {workers} workers")
    procs = []
    for wdir in worker_dirs:
        logf = open(os.path.join(wdir, "parse.log"), "w")
        p = subprocess.Popen(
            [binary, "-i", wdir, "-o", wdir, "--sections", "--full-text"],
            stdout=logf, stderr=subprocess.STDOUT,
        )
        procs.append((p, logf, wdir))

    rc_all = 0
    for p, logf, wdir in procs:
        rc_all |= p.wait()
        logf.close()
        if p.returncode != 0:
            tail = open(os.path.join(wdir, "parse.log")).read()[-2000:]
            log(f"parse: worker {wdir} failed (rc={p.returncode}):\n{tail}")
    if rc_all != 0:
        raise RuntimeError("one or more parse workers failed")


# ---------------------------------------------------------------------------
# Stage 3: merge per-worker parquets
# ---------------------------------------------------------------------------

def stage_merge(args, repo_root: str) -> None:
    import pyarrow.parquet as pq

    for name in ("nodes", "edges", "sections"):
        out_path = os.path.join(args.data_dir, f"{name}.parquet")
        if os.path.exists(out_path):
            log(f"merge: skip {name}.parquet (present)")
            continue
        parts = []
        w = 0
        while True:
            p = os.path.join(args.data_dir, f"worker_{w}", f"{name}.parquet")
            if not os.path.exists(p):
                break
            parts.append(p)
            w += 1
        if not parts:
            raise RuntimeError(f"merge: no worker parts for {name}")
        log(f"merge: {name} <- {len(parts)} parts")
        readers = [pq.ParquetFile(p) for p in parts]
        schema = readers[0].schema_arrow
        with pq.ParquetWriter(out_path, schema) as writer:
            for r in readers:
                for rg in range(r.metadata.num_row_groups):
                    writer.write_table(r.read_row_group(rg))
        log(f"merge: {name}.parquet {os.path.getsize(out_path) / 1e9:.1f} GB")


# ---------------------------------------------------------------------------
# Stage 4: build the 3 tiers
# ---------------------------------------------------------------------------

def stage_tier(args, repo_root: str) -> None:
    import polars as pl

    nodes_path = os.path.join(args.data_dir, "nodes.parquet")
    sections_path_in = os.path.join(args.data_dir, "sections.parquet")
    if not os.path.exists(nodes_path) or not os.path.exists(sections_path_in):
        raise RuntimeError("tier: nodes.parquet / sections.parquet missing (run parse)")

    os.makedirs(args.out_dir, exist_ok=True)
    articles_path = os.path.join(args.out_dir, "articles.parquet")
    sections_path = os.path.join(args.out_dir, "sections.parquet")
    texts_path = os.path.join(args.out_dir, "section_texts.parquet")

    # Resumability: if the final tier exists, the whole tier stage is done.
    if os.path.exists(texts_path):
        log("tier: all three tiers present; skipping")
        _report_tiers(args)
        return

    # ── Tier 1: articles ──────────────────────────────────────────────
    if not os.path.exists(articles_path):
        log("tier: building articles")
        nodes = pl.read_parquet(nodes_path, columns=["id", "title"])
        articles = nodes.with_row_index("article_idx").rename(
            {"id": "page_id", "article_idx": "id"}
        ).select(["id", "page_id", "title"])
        articles.write_parquet(articles_path)
        log(f"tier: articles {len(articles):,}")
        del nodes
    else:
        articles = pl.read_parquet(articles_path)
        log(f"tier: articles present ({len(articles):,})")

    # ── Tier 2: sections (deterministic; always rebuilt if text missing) ─
    # Rebuilding sec_all here (rather than trusting a stale sections.parquet)
    # guarantees section ids match the section_texts we are about to write,
    # and makes the stage independently resumable.
    log("tier: loading sections")
    sec = pl.read_parquet(
        sections_path_in,
        columns=["page_id", "section_title", "level", "start_byte", "end_byte"],
    )
    aid_df = articles.select([pl.col("page_id"), pl.col("id").alias("article_id")])
    sec = sec.join(aid_df, on="page_id", how="inner")

    # Header-less pages become one synthetic whole-page section.
    log("tier: synthesising sections for header-less pages")
    pages_with_sections = sec.select("page_id").unique()
    headerless = (
        articles.lazy()
        .select(["page_id", "id"])
        .join(pages_with_sections.lazy(), on="page_id", how="anti")
        .with_columns([
            pl.lit("", dtype=pl.Utf8).alias("section_title"),
            pl.lit(0, dtype=pl.Int32).alias("level"),
            pl.lit(0, dtype=pl.Int64).alias("start_byte"),
            pl.lit(-1, dtype=pl.Int64).alias("end_byte"),
            pl.col("id").alias("article_id"),
        ])
        .select(["page_id", "section_title", "level", "start_byte", "end_byte",
                 "article_id"])
        .collect()
    )
    log(f"tier: header-less pages {len(headerless):,}")

    # Deterministic ordering so ids are stable across reruns: real sections
    # sorted by (page_id, start_byte), then synthetic sorted by page_id.
    cols = ["page_id", "section_title", "level", "start_byte", "end_byte", "article_id"]
    sec = sec.select(cols).sort(["page_id", "start_byte"])
    headerless = headerless.select(cols).sort("page_id")
    sec_all = pl.concat([sec, headerless]).with_row_index("id")
    sec_all.select(["id", "article_id", "section_title", "level"]).write_parquet(
        sections_path
    )
    log(f"tier: sections {len(sec_all):,}")

    # ── Tier 3: section_text (manual chunking to strictly bound memory) ─
    log("tier: chunked section_text -> parquet")
    import pyarrow.parquet as pq
    
    sec_eager = sec_all.select(["id", "page_id", "start_byte", "end_byte"])
    del sec_all  # immediately free full dataframe
    
    full_len = pl.col("text").str.len_bytes()
    eff_end = pl.when(pl.col("end_byte") < 0).then(full_len).otherwise(pl.col("end_byte"))
    eff_len = (eff_end - pl.col("start_byte")).clip(lower_bound=0)
    snippet = pl.col("text").str.slice(pl.col("start_byte"), eff_len)
    
    pq_file = pq.ParquetFile(nodes_path)
    writer = None
    text_id_offset = 0
    
    for rg in range(pq_file.num_row_groups):
        nodes_chunk = pl.from_arrow(pq_file.read_row_group(rg))
        
        chunk_res = (
            nodes_chunk.lazy()
            .select(["id", "summary"])
            .rename({"id": "page_id", "summary": "text"})
            .join(sec_eager.lazy(), on="page_id", how="inner")
            .with_columns(snippet.alias("snippet"))
            .select([pl.col("id").alias("section_id"), pl.col("snippet").alias("text")])
            .collect()
        )
        del nodes_chunk
        
        if len(chunk_res) > 0:
            chunk_res = chunk_res.with_row_index("text_id")
            chunk_res = chunk_res.with_columns(pl.col("text_id") + text_id_offset)
            chunk_res = chunk_res.select(["text_id", "section_id", "text"])
            
            text_id_offset += len(chunk_res)
            
            arrow_batch = chunk_res.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(texts_path, arrow_batch.schema)
            writer.write_table(arrow_batch)
            del arrow_batch
        del chunk_res
        log(f"tier: wrote chunk {rg+1}/{pq_file.num_row_groups} (offset {text_id_offset:,})")
        
    if writer is not None:
        writer.close()
        
    log(f"tier: section_texts {os.path.getsize(texts_path) / 1e9:.2f} GB")

    _report_tiers(args)


def _report_tiers(args) -> None:
    import pyarrow.parquet as pq

    for name in ["articles.parquet", "sections.parquet", "section_texts.parquet", "section_embeddings.parquet"]:
        p = os.path.join(args.out_dir, name)
        if os.path.exists(p):
            log(f"tier:   {name:25s} {os.path.getsize(p) / 1e6:8.1f} MB")
    parts_dir = os.path.join(args.out_dir, "section_embeddings_parts")
    if os.path.isdir(parts_dir):
        parts = sorted(glob.glob(os.path.join(parts_dir, "part_*.parquet")))
        if parts:
            total_sz = sum(os.path.getsize(p) for p in parts)
            total_rows = sum(pq.ParquetFile(p).metadata.num_rows for p in parts)
            log(f"tier:   {'section_embeddings_parts':25s} {total_sz / 1e6:8.1f} MB ({len(parts)} parts, {total_rows:,} rows)")


# ---------------------------------------------------------------------------
# Stage 5: embed (GPU)
# ---------------------------------------------------------------------------

def stage_embed(args, repo_root: str) -> None:
    import numpy as np
    import polars as pl
    import pyarrow.parquet as pq

    texts_path = os.path.join(args.out_dir, "section_texts.parquet")
    out_path = os.path.join(args.out_dir, "section_embeddings.parquet")
    parts_dir = os.path.join(args.out_dir, "section_embeddings_parts")
    os.makedirs(parts_dir, exist_ok=True)

    if not os.path.exists(texts_path):
        raise RuntimeError("embed: section_texts.parquet missing (run tier)")

    pq_file = pq.ParquetFile(texts_path)
    total = pq_file.metadata.num_rows
    log(f"embed: {total:,} sections total (chunked streaming)")

    # If a legacy monolithic embeddings file exists, adopt it as part_0000
    legacy_part0 = os.path.join(parts_dir, "part_0000.parquet")
    if os.path.exists(out_path) and not os.path.exists(legacy_part0):
        log(f"embed: adopting existing {out_path} as part_0000.parquet")
        os.rename(out_path, legacy_part0)

    # Discover existing valid part files
    existing_parts = sorted(glob.glob(os.path.join(parts_dir, "part_*.parquet")))
    total_embedded = 0
    valid_parts = []
    for p in existing_parts:
        try:
            pf = pq.ParquetFile(p)
            n = pf.metadata.num_rows
            if n > 0:
                total_embedded += n
                valid_parts.append((p, n))
        except Exception as e:
            log(f"embed: warning - removing corrupt/incomplete part file {p}: {e}")
            os.remove(p)

    if total_embedded >= total:
        log(f"embed: all {total_embedded:,}/{total:,} sections already complete across {len(valid_parts)} parts; skipping")
        return

    if total_embedded > 0:
        log(f"embed: RESUMING from row {total_embedded:,}/{total:,} ({(total_embedded * 100.0 / total):.2f}% complete across {len(valid_parts)} parts)")

    from sentence_transformers import SentenceTransformer
    log(f"embed: loading {args.embed_model} on {args.device}")
    model = SentenceTransformer(args.embed_model, device=args.device)

    MAX_ROWS_PER_PART = 5_000_000  # ~12.5 GB per part file
    part_idx = len(valid_parts)
    rows_in_current_part = 0
    writer = None
    done = total_embedded
    t0 = time.time()

    try:
        # Push the resume offset down to the Parquet reader so already-embedded
        # rows are skipped *without* reading their (large) `text` column. The
        # parts are contiguous row-order slices of section_texts, so `text_id`
        # is the row index and `text_id >= total_embedded` selects exactly the
        # remaining rows. The old code iterated every batch and discarded the
        # skipped ones in Python, which read the whole ~70 GB text column on
        # every resume and made a near-complete run look like a hang.
        import pyarrow.dataset as ds

        dataset = ds.dataset(texts_path, format="parquet")
        scanner = dataset.scanner(
            columns=["text_id", "section_id", "text"],
            filter=ds.field("text_id") >= total_embedded,
            batch_size=args.embed_batch,
        )
        for batch_record in scanner.to_batches():
            df_chunk = pl.from_arrow(batch_record)
            texts = df_chunk["text"].to_list()
            ids = df_chunk["text_id"].to_list()
            sids = df_chunk["section_id"].to_list()

            emb = model.encode(
                texts, batch_size=args.embed_batch, show_progress_bar=False,
                convert_to_numpy=True, normalize_embeddings=True,
            )

            out_df = pl.DataFrame({
                "text_id": ids,
                "section_id": sids,
                "embedding": emb.tolist(),
            })

            arrow_batch = out_df.to_arrow()
            if writer is None:
                part_path = os.path.join(parts_dir, f"part_{part_idx:04d}.parquet")
                writer = pq.ParquetWriter(part_path, arrow_batch.schema)
                rows_in_current_part = 0

            writer.write_table(arrow_batch)
            rows_in_current_part += len(texts)
            done += len(texts)

            if rows_in_current_part >= MAX_ROWS_PER_PART:
                writer.close()
                part_path = os.path.join(parts_dir, f"part_{part_idx:04d}.parquet")
                log(f"embed: finalized part_{part_idx:04d}.parquet ({rows_in_current_part:,} rows, {os.path.getsize(part_path) / 1e9:.2f} GB)")
                writer = None
                part_idx += 1

            if done % (args.embed_batch * 10) == 0 or done == total:
                elapsed = max(time.time() - t0, 1e-9)
                rate = (done - total_embedded) / elapsed
                eta = (total - done) / rate if rate > 0 else 0
                log(f"embed: {done:,}/{total:,} ({done * 100 / total:.1f}%) "
                    f"{rate:.0f}/s ETA {eta / 3600:.1f}h")

    finally:
        if writer is not None:
            writer.close()
            part_path = os.path.join(parts_dir, f"part_{part_idx:04d}.parquet")
            log(f"embed: finalized part_{part_idx:04d}.parquet ({rows_in_current_part:,} rows, {os.path.getsize(part_path) / 1e9:.2f} GB)")

    log(f"embed: finished stage_embed ({done:,}/{total:,} sections embedded)")


# ---------------------------------------------------------------------------
# Stage 6: load into BenoStreamDB
# ---------------------------------------------------------------------------

def stage_load(args, repo_root: str) -> None:
    import polars as pl
    import pyarrow as pa
    import pyarrow.parquet as pq

    sys.path.insert(0, os.path.join(repo_root, "python"))
    import benostreamdb as bsdb

    os.makedirs(args.table_dir, exist_ok=True)
    try:
        gpu = bsdb.Device.auto_detect()
        gpu.activate()
        log(f"load: GPU {gpu}")
    except Exception:  # noqa: BLE001
        log("load: GPU unavailable, CPU")

    articles = pl.read_parquet(os.path.join(args.out_dir, "articles.parquet")).sort("id")
    sections = pl.read_parquet(os.path.join(args.out_dir, "sections.parquet")).sort("article_id")

    articles_uri = f"file://{os.path.abspath(args.table_dir)}/articles"
    art = bsdb.Table.create(articles_uri, pa.schema([
        ("id", pa.int64()), ("page_id", pa.large_string()), ("title", pa.large_string()),
    ]))
    _write_batched(art, articles, "articles", args.load_batch)

    # Edge table: article --[has_section]--> section (vertex-centric CSR).
    sections_uri = f"file://{os.path.abspath(args.table_dir)}/sections"
    sec = bsdb.Table.create_edge_table(sections_uri, pa.schema([
        ("source", pa.int64()), ("target", pa.int64()),
        ("section_title", pa.large_string()), ("level", pa.int32()),
    ]))
    sec_df = sections.select([
        pl.col("article_id").alias("source"),
        pl.col("id").alias("target"),
        pl.col("section_title"), pl.col("level"),
    ])
    _write_batched(sec, sec_df, "sections", args.load_batch)

    # Stream section_texts without full RAM materialization
    texts_path = os.path.join(args.out_dir, "section_texts.parquet")
    if os.path.exists(texts_path):
        texts_uri = f"file://{os.path.abspath(args.table_dir)}/section_texts"
        txt = bsdb.Table.create(texts_uri, pa.schema([
            ("section_id", pa.int64()), ("text", pa.large_string()),
        ]))
        pq_texts = pq.ParquetFile(texts_path)
        total_txt = pq_texts.metadata.num_rows
        log(f"load: streaming section_texts ({total_txt:,} rows)")
        t0 = time.time()
        done = 0
        for batch in pq_texts.iter_batches(batch_size=args.load_batch, columns=["section_id", "text"]):
            df_batch = pl.from_arrow(batch)
            txt.write(df_batch)
            done += len(df_batch)
            if done % (args.load_batch * 4) == 0 or done == total_txt:
                rate = done / max(time.time() - t0, 1e-9)
                log(f"load: [section_texts] {done:,}/{total_txt:,} ({done * 100 / total_txt:.0f}%) {rate:.0f}/s")

    # Stream section_embeddings from parts or single file
    parts_dir = os.path.join(args.out_dir, "section_embeddings_parts")
    emb_files = sorted(glob.glob(os.path.join(parts_dir, "part_*.parquet"))) if os.path.isdir(parts_dir) else []
    emb_path = os.path.join(args.out_dir, "section_embeddings.parquet")
    if not emb_files and os.path.exists(emb_path):
        emb_files = [emb_path]

    if emb_files:
        emb_uri = f"file://{os.path.abspath(args.table_dir)}/section_embeds"
        embt = bsdb.Table.create(emb_uri, pa.schema([
            ("section_id", pa.int64()), ("embedding", pa.list_(pa.float32())),
        ]))
        total_emb = sum(pq.ParquetFile(f).metadata.num_rows for f in emb_files)
        log(f"load: streaming {len(emb_files)} embedding file(s) ({total_emb:,} rows)")
        t0 = time.time()
        done = 0
        for f in emb_files:
            pq_f = pq.ParquetFile(f)
            for batch in pq_f.iter_batches(batch_size=args.load_batch, columns=["section_id", "embedding"]):
                df_batch = pl.from_arrow(batch)
                embt.write(df_batch)
                done += len(df_batch)
                if done % (args.load_batch * 4) == 0 or done == total_emb:
                    rate = done / max(time.time() - t0, 1e-9)
                    log(f"load: [section_embeds] {done:,}/{total_emb:,} ({done * 100 / total_emb:.0f}%) {rate:.0f}/s")

        log("load: building HNSW index")
        embt.add_index("embedding", algorithm="hnsw")

    # Warm the caches (Dgraph-style in-memory residency + disk spillover).
    log("load: preloading index caches")
    for name, tbl in [("articles", art), ("sections", sec)]:
        try:
            stats = tbl.preload_indexes()
            log(f"load: preload {name}: {stats}")
        except Exception as e:  # noqa: BLE001
            log(f"load: preload {name} skipped: {e}")


def _write_batched(table, df, name: str, batch: int) -> None:
    total = len(df)
    t0 = time.time()
    for start in range(0, total, batch):
        end = min(start + batch, total)
        table.write(df.slice(start, end - start))
        rate = end / max(time.time() - t0, 1e-9)
        log(f"load: [{name}] {end:,}/{total:,} ({end * 100 / total:.0f}%) {rate:.0f}/s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    default_data_dir = os.environ.get("BENOSTREAM_DATA", os.path.expanduser("~/data/benostreamdb"))

    ap = argparse.ArgumentParser(description="3-tier Wikipedia Graph-RAG pipeline")
    ap.add_argument("--stages", default=",".join(ALL_STAGES),
                    help=f"comma-separated subset of {ALL_STAGES}")
    ap.add_argument("--data-dir", default=default_data_dir,
                    help="HDD: dumps + parsed nodes/edges/sections")
    ap.add_argument("--out-dir", default=None,
                    help="tier parquet output (default: <data-dir>/three_tier)")
    ap.add_argument("--table-dir", default=None,
                    help="BenoStreamDB table directory (default: <repo>/data/three_tier_db)")
    ap.add_argument("--workers", type=int, default=4, help="parallel parse/download workers")
    ap.add_argument("--dump-date", default=DEFAULT_DUMP_DATE)
    ap.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    ap.add_argument("--device", default="cuda", help="cuda / cpu / mps")
    ap.add_argument("--embed-batch", type=int, default=2048)
    ap.add_argument("--load-batch", type=int, default=250_000)
    args = ap.parse_args()

    args.data_dir = os.path.abspath(args.data_dir)
    if args.out_dir is None:
        args.out_dir = os.environ.get(
            "BENOSTREAM_THREE_TIER_OUT",
            os.path.join(args.data_dir, "three_tier"),
        )
    args.out_dir = os.path.abspath(args.out_dir)

    if args.table_dir is None:
        args.table_dir = os.environ.get(
            "BENOSTREAM_TABLE_DIR",
            os.path.join(repo_root, "data", "three_tier_db"),
        )
    args.table_dir = os.path.abspath(args.table_dir)

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    for s in stages:
        if s not in ALL_STAGES:
            ap.error(f"unknown stage '{s}' (choose from {ALL_STAGES})")

    log(f"repo={repo_root}")
    log(f"data={args.data_dir}  out={args.out_dir}  tables={args.table_dir}")
    log(f"stages={stages}")

    t0 = time.time()
    for s in stages:
        log(f"── stage: {s} ──")
        t = time.time()
        if s == "download":
            stage_download(args)
        elif s == "parse":
            stage_parse(args, repo_root)
        elif s == "merge":
            stage_merge(args, repo_root)
        elif s == "tier":
            stage_tier(args, repo_root)
        elif s == "embed":
            stage_embed(args, repo_root)
        elif s == "load":
            stage_load(args, repo_root)
        log(f"── stage {s} done in {(time.time() - t) / 60:.1f}m ──")

    log(f"all stages done in {(time.time() - t0) / 60:.1f}m")


if __name__ == "__main__":
    main()