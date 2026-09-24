#!/usr/bin/env python3
"""Compact the demo nodes table and rebuild its indexes.

Compaction previously wrote merged segments with a bare SegmentConfig, silently
dropping vector/inverted indexes (queries then flat-scanned 98 GB). The engine
now carries the table's index config into the compactor; this script applies the
config to the opened table and rewrites the 73 small segments into ~25 large
indexed ones.

Usage: python scripts/demo_compact.py [min_file_size_bytes]
"""
import os
import sys
import time

os.environ.setdefault("BENOSTREAM_CACHE_GB", "40")
import benostreamdb as bsdb

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Tables live on the SSD in the repo's original location.
NODES = f"file://{os.path.join(REPO, 'data', 'wiki_graph_db', 'nodes')}"
MIN = int(sys.argv[1]) if len(sys.argv) > 1 else 2_000_000_000  # 2 GB -> 4 GB targets

t = bsdb.Table(NODES)
# Index config is not persisted across opens, so re-apply it before compaction
# (otherwise the compactor has nothing to rebuild).
t.add_index("embedding", "hnsw_tq8")
t.add_index("title", "inverted")
print(f"[compact] min_file_size_bytes={MIN:,} (target {MIN*2:,})", flush=True)
t0 = time.time()
t.rewrite_data_files(MIN)
print(f"[compact] done in {time.time()-t0:.0f}s", flush=True)