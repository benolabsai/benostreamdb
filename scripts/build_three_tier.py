#!/usr/bin/env python3
"""Build the 3-tier Wikipedia dataset from parsed parquets (streaming).

Tier 1 (articles)     — id, title, page_id          (metadata only)
Tier 2 (sections)     — id, article_id, section_title, level
Tier 3 (section_text) — id, section_id, text         (raw wikitext slice)

Inputs (on the HDD, produced by the parse stage with --sections --full-text):
  * nodes.parquet     — id (str), title (str), summary (str = FULL wikitext)
  * sections.parquet  — page_id, page_title, section_title, level, start_byte, end_byte

Uses Polars streaming (scan_parquet + collect(streaming=True)) to avoid
loading 80+ GB of full wikitext into RAM.  Also filters the full-text
scan to only the page_ids that actually appear in sections, which further
cuts the working set.

Usage:
  python scripts/build_three_tier.py --nodes <nodes.parquet> \\
      --sections <sections.parquet> --out-dir data/three_tier
"""

import argparse
import os
import sys

import polars as pl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--sections", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=500_000,
                    help="Rows per output batch when streaming (default 500k)")
    ap.add_argument("--disk-cache", action="store_true",
                    help="Spill large intermediates to disk via Polars env")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.disk_cache:
        os.environ["POLARS_FORCE_OOC"] = "1"

    # ── Tier 1: articles ──────────────────────────────────────────────
    # id (curid, str), title (str) — only these two narrow columns,
    # so an eager read is fine even for 66M rows.
    print("[3tier] building articles ...", flush=True)
    nodes = pl.read_parquet(args.nodes, columns=["id", "title"])
    articles = nodes.with_row_index("article_idx").rename(
        {"id": "page_id", "article_idx": "id"}
    )
    articles = articles.select(["id", "page_id", "title"])
    articles.write_parquet(os.path.join(args.out_dir, "articles.parquet"))
    del nodes  # free the id+title frame
    print(f"[3tier] articles: {len(articles):,}", flush=True)

    # page_id (str) -> article id (int64)
    page_id_to_aid = dict(
        zip(articles["page_id"].to_list(), articles["id"].to_list())
    )
    del articles  # free before loading sections

    # ── Tier 2 + 3: sections and their text ──────────────────────────
    print("[3tier] loading sections ...", flush=True)
    sec = pl.read_parquet(
        args.sections,
        columns=["page_id", "section_title", "level", "start_byte", "end_byte"],
    )
    print(f"[3tier] sections raw: {len(sec):,}", flush=True)

    # Map page_id → article_id; drop sections whose page_id is unknown
    aid_df = pl.DataFrame({
        "page_id": list(page_id_to_aid.keys()),
        "article_id": list(page_id_to_aid.values()),
    })
    sec = sec.join(aid_df, on="page_id", how="inner")
    sec = sec.with_row_index("id")
    # id, article_id, section_title, level, page_id, start_byte, end_byte
    sec = sec.select([
        "id", "article_id", "section_title", "level",
        "page_id", "start_byte", "end_byte",
    ])
    print(f"[3tier] sections after join: {len(sec):,}", flush=True)

    # Collect the set of page_ids that actually appear in sections.
    # This is the key optimisation: we will only load full wikitext
    # for *these* pages, not all 66M.
    needed = set(sec["page_id"].to_list())
    print(f"[3tier] unique pages with sections: {len(needed):,}", flush=True)

    # Write Tier 2 (sections without text)
    sections_out = sec.select(["id", "article_id", "section_title", "level"])
    sections_out.write_parquet(os.path.join(args.out_dir, "sections.parquet"))
    print(f"[3tier] sections written: {len(sections_out):,}", flush=True)

    # ── Tier 3: section_text ─────────────────────────────────────────
    print("[3tier] building section_text (streaming) ...", flush=True)

    # Stream the full-text column, filtering to only needed page_ids.
    full_text = (
        pl.scan_parquet(args.nodes)
        .select(["id", "summary"])
        .rename({"id": "page_id", "summary": "text"})
        .filter(pl.col("page_id").is_in(list(needed)))
    )

    # Join sections (with byte ranges) to the filtered full text.
    # Use lazy join + streaming collect so we never materialise the
    # full 80 GB column in RAM.
    sec_lazy = sec.lazy()
    joined = sec_lazy.join(full_text, on="page_id", how="inner")

    # Slice section text: text[start_byte:end_byte].
    # Polars str.slice() uses byte offsets, which matches our parquet values.
    # Guard against invalid ranges (end_byte <= start_byte) by clamping
    # end_byte to text length and ensuring non-negative length.
    safe_end = (
        pl.when(pl.col("end_byte") > pl.col("start_byte"))
        .then(pl.col("end_byte"))
        .otherwise(pl.col("text").str.len_bytes())
    )
    safe_len = (
        pl.when(safe_end > pl.col("start_byte"))
        .then(safe_end - pl.col("start_byte"))
        .otherwise(pl.lit(0, dtype=pl.UInt64))
    )
    sliced = joined.with_columns(
        pl.col("text").str.slice(pl.col("start_byte"), safe_len).alias("snippet")
    )

    # Select final columns for Tier 3
    result = sliced.select([
        pl.col("id").alias("section_id"),
        pl.col("snippet").alias("text"),
    ]).with_row_index("text_id")

    # Collect in streaming mode — processes in batches, avoids OOM
    sec_text = result.collect(engine="streaming")
    print(f"[3tier] section_text rows: {len(sec_text):,}", flush=True)

    sec_text.write_parquet(os.path.join(args.out_dir, "section_texts.parquet"))
    print("[3tier] section_texts written.", flush=True)

    # ── Summary ──────────────────────────────────────────────────────
    total_sec = len(sections_out)
    total_txt = len(sec_text)
    print(f"\n[3tier] done.")
    print(f"  articles      : {total_sec:,} sections across unique pages")
    print(f"  section_texts : {total_txt:,} rows")
    for name in ["articles.parquet", "sections.parquet", "section_texts.parquet"]:
        p = os.path.join(args.out_dir, name)
        if os.path.exists(p):
            sz = os.path.getsize(p) / 1e6
            print(f"  {name:25s} {sz:8.1f} MB")


if __name__ == "__main__":
    main()