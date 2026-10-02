#!/usr/bin/env python3
"""Reproduce the native crash in the HNSW/TQ8 index builder.

The prepare_demo load segfaulted while building the wiki_nodes `hnsw_tq8`
index (see logs/pipeline_r13.log). The wiki_nodes embedding shards were
deleted after load, but the three-tier section embeddings (384-d, same
dimension as the 384-d demo run) are still on disk and exercise the exact
same builder.

Usage:
  RUST_BACKTRACE=full BSDB_INDEX_BUILD_CONCURRENCY=1 \
    .venv/bin/python -u scripts/repro_hnsw_crash.py --rows 2000000 --quant tq8
"""
import argparse
import os
import shutil
import sys
import time

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, os.path.abspath("python"))
import benostreamdb as bsdb  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=2_000_000)
    ap.add_argument("--quant", default="tq8", choices=["tq8", "tq4", "none"])
    ap.add_argument("--table", default="data/repro_hnsw_db")
    ap.add_argument(
        "--part",
        default=os.path.expanduser(
            "~/data/benostreamdb/three_tier/section_embeddings_parts/part_0000.parquet"
        ),
    )
    ap.add_argument("--batch", type=int, default=100_000)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    if not os.path.exists(args.part):
        print(f"ERROR: source part not found: {args.part}", flush=True)
        return 2

    uri = "file://" + os.path.abspath(args.table)
    shutil.rmtree(args.table, ignore_errors=True)
    print(
        f"repro: table={uri} rows={args.rows:,} quant={args.quant} device={args.device}",
        flush=True,
    )

    schema = pa.schema(
        [("section_id", pa.int64()), ("embedding", pa.list_(pa.float32()))]
    )
    t = bsdb.Table.create(uri, schema)
    t.set_default_device(args.device)

    pf = pq.ParquetFile(args.part)
    done = 0
    t0 = time.time()
    for batch in pf.iter_batches(
        batch_size=args.batch, columns=["section_id", "embedding"]
    ):
        if done >= args.rows:
            break
        take = min(batch.num_rows, args.rows - done)
        b = batch.slice(0, take)
        emb32 = pc.cast(b.column("embedding"), pa.list_(pa.float32()))
        out = pa.table({"section_id": b.column("section_id"), "embedding": emb32})
        t.write(out)
        done += take
        print(
            f"repro: wrote {done:,}/{args.rows:,} ({time.time() - t0:.0f}s)",
            flush=True,
        )
    t.commit()
    print(f"repro: wrote {done:,} rows in {time.time() - t0:.0f}s", flush=True)

    alg = {"type": f"hnsw_{args.quant}" if args.quant != "none" else "hnsw",
           "device": args.device}
    print(f"repro: add_index embedding {alg}", flush=True)
    t.add_index("embedding", alg)
    print("repro: waiting for background index build...", flush=True)
    t.wait_for_background_tasks()
    print("repro: DONE — index built without crash", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
