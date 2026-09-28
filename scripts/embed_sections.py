#!/usr/bin/env python3
"""Embed section texts using SentenceTransformer on GPU.

Reads data/three_tier/section_texts.parquet, embeds the 'text' column,
writes data/three_tier/section_embeddings.parquet with columns:
  text_id, section_id, embedding (list<float32>)

Usage:
  python scripts/embed_sections.py --batch-size 256
"""

import argparse
import os
import sys
import time

import numpy as np
import polars as pl
from sentence_transformers import SentenceTransformer

MODEL = os.environ.get("BSDB_DEMO_EMBED_MODEL", "all-MiniLM-L6-v2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/three_tier/section_texts.parquet")
    ap.add_argument("--output", default="data/three_tier/section_embeddings.parquet")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0,
                    help="Max sections to embed (0 = all)")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    print(f"[embed] loading model {args.model} on cuda ...", flush=True)
    model = SentenceTransformer(args.model, device="cuda")
    dims = model.get_sentence_embedding_dimension()
    print(f"[embed] embedding dimension: {dims}", flush=True)

    print(f"[embed] reading {args.input} ...", flush=True)
    df = pl.read_parquet(args.input, columns=["text_id", "section_id", "text"])
    total = len(df)
    if args.limit > 0:
        df = df.head(args.limit)
        total = len(df)
    print(f"[embed] {total:,} sections to embed", flush=True)

    texts = df["text"].to_list()
    ids = df["text_id"].to_list()
    sids = df["section_id"].to_list()
    del df

    batch_size = args.batch_size
    n_batches = (total + batch_size - 1) // batch_size

    all_embeds = []
    t0 = time.time()
    for i in range(0, total, batch_size):
        batch_texts = texts[i:i + batch_size]
        embeddings = model.encode(
            batch_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        all_embeds.append(embeddings)

        done = min(i + batch_size, total)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        pct = done * 100 / total
        eta = (total - done) / rate if rate > 0 else 0
        print(f"[embed] {done:,}/{total:,} ({pct:.1f}%)  "
              f"{rate:.0f} texts/s  ETA {eta/60:.0f}m",
              flush=True)

    elapsed = time.time() - t0
    print(f"[embed] done in {elapsed/60:.1f}m  ({total/elapsed:.0f} texts/s)",
          flush=True)

    # Concatenate all embeddings
    all_embeds = np.concatenate(all_embeds, axis=0)
    print(f"[embed] embedding matrix: {all_embeds.shape}", flush=True)

    # Build output dataframe
    out = pl.DataFrame({
        "text_id": ids,
        "section_id": sids,
        "embedding": [row.tolist() for row in all_embeds],
    })
    out.write_parquet(args.output)
    sz = os.path.getsize(args.output) / 1e6
    print(f"[embed] wrote {args.output} ({sz:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()