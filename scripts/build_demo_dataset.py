#!/usr/bin/env python3
"""Consume the full Wikipedia dump parquets and prune them into a compact,
interactive demo dataset for the Streamlit web UI.

All expensive graph work runs INSIDE HyperStreamDB so the build doubles as a
scale test of the engine:

  1. Stream data/nodes.parquet to identify redirect stubs (no real content)
     and build a title -> curid map.
  2. Stream data/edges.parquet, resolving mixed curid/title endpoints to
     int64 curids (the CSR graph index and graph UDFs require integer ids),
     dropping self-loops and edges touching redirects.
  3. Load the edges into a temporary HyperStreamDB table and run
     connected_components() (pointer jumping + edge contraction).
  4. Rank nodes with degree_centrality(), then extract a dense interactive
     subgraph with the subgraph() UDF (multi-hop induced subgraph from the
     highest-degree hubs of the largest component).
  5. Filter nodes to the kept ids and embed summaries with
     sentence-transformers (BAAI/bge-large-en-v1.5, 1024d) for semantic
     entity resolution in the UI.
  6. Write data/demo_nodes.parquet and data/demo_edges.parquet (int64 ids).

Usage:
  python scripts/build_demo_dataset.py [--max-nodes 50000] [--no-embed]
"""

import argparse
import os
import shutil
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# Dumps (full wiki parquets) live on the 14 TB HDD by default; override with
# --dumps-dir or HYPERSTREAM_DATA.
DEFAULT_DUMPS = os.environ.get(
    "HYPERSTREAM_DATA",
    os.path.join(os.path.expanduser("~"), "data", "hyperstreamdb"),
)


def log(msg: str) -> None:
    print(f"[build_demo] {msg}", flush=True)


def _mp_context():
    """Prefer fork (workers inherit big globals via COW — matters on Python
    >=3.14 where the default became forkserver/spawn, which would not see
    _TITLE_MAP set after pool creation). Fall back to the platform default
    where fork is unavailable (Windows; macOS 3.12+ warns but works)."""
    import multiprocessing as mp

    try:
        return mp.get_context("fork")
    except (ValueError, AttributeError):
        return mp.get_context()


def _scan_node_row_group(args):
    """Worker: one node row group -> (redirect ids, title->curid dict)."""
    nodes_path, rg = args
    tbl = pq.ParquetFile(nodes_path).read_row_group(rg, columns=["id", "title", "summary"])
    ids = pc.cast(tbl.column("id"), pa.int64()).to_numpy()
    titles = tbl.column("title").to_pylist()
    summ = tbl.column("summary").to_pylist()
    mask = np.fromiter(
        (s is not None and s.lstrip().startswith("#REDIRECT") for s in summ),
        dtype=bool, count=len(summ),
    )
    # Exact titles only; underscored wiki-links are normalized at lookup time
    # (vectorized) instead of duplicating 14M dict entries.
    return ids[mask], dict(zip(titles, ids.tolist()))


def scan_nodes(nodes_path: str, workers: int) -> tuple:
    """Parallel pass over nodes.parquet: returns (redirect_curids, title_map)."""
    import multiprocessing as mp

    n_rg = pq.ParquetFile(nodes_path).metadata.num_row_groups
    with _mp_context().Pool(workers) as pool:
        results = pool.map(_scan_node_row_group, [(nodes_path, rg) for rg in range(n_rg)])
    bad = []
    m = {}
    for b, part in results:
        bad.append(b)
        m.update(part)
    redirects = np.concatenate(bad) if bad else np.array([], dtype=np.int64)
    log(f"  node scan complete: {len(redirects):,} redirects, {len(m):,} title keys")
    return redirects, m


_TITLE_MAP: dict = {}
_DROP: np.ndarray = np.array([], dtype=np.int64)


def _init_edge_worker() -> None:
    """Forked workers inherit _TITLE_MAP/_DROP via copy-on-write; nothing to do."""


def resolve_endpoints(s: pd.Series) -> np.ndarray:
    """Resolve mixed curid/title endpoint strings to int64 curids (NaN if unknown)."""
    # Wiki links arrive as "Machine_learning"; normalize underscores to spaces
    # (vectorized) so the title map needs only canonical titles as keys.
    norm = s.str.replace("_", " ", regex=False)
    num = pd.to_numeric(s, errors="coerce")
    mapped = norm.map(_TITLE_MAP)
    # Fallback: capitalize first letter (wiki links often lowercase the target).
    rem = mapped.isna() & num.isna()
    if rem.any():
        fixed = norm[rem].str.capitalize().map(_TITLE_MAP)
        mapped = mapped.fillna(fixed)
    out = num.fillna(mapped)
    arr = out.to_numpy(dtype="float64", na_value=np.nan)
    # Guard: NaN/inf would become i64::MIN on astype and poison the graph.
    return np.where(np.isfinite(arr), arr, np.nan)


def _resolve_edge_row_group(args):
    """Worker: resolve one edge row group to int64 arrays (or None if empty)."""
    edges_path, rg = args
    batch = pq.ParquetFile(edges_path).read_row_group(rg, columns=["source", "target"])
    src = resolve_endpoints(batch.column("source").to_pandas())
    dst = resolve_endpoints(batch.column("target").to_pandas())
    known = ~np.isnan(src) & ~np.isnan(dst)
    unresolved = int((~known).sum())
    src_i = src[known].astype(np.int64)
    dst_i = dst[known].astype(np.int64)
    valid = (src_i > 0) & (dst_i > 0) & (src_i != dst_i)
    if len(_DROP):
        valid &= ~(_isin_valid(src_i, _DROP) | _isin_valid(dst_i, _DROP))
    return src_i[valid], dst_i[valid], unresolved


def _isin_valid(vals: np.ndarray, sorted_ref: np.ndarray) -> np.ndarray:
    return np.isin(vals, sorted_ref)


def stream_int64_edges(edges_path: str, out_path: str, drop: np.ndarray,
                       title_map: dict, workers: int) -> tuple:
    """Resolve endpoints to int64 curids in parallel, drop self-loops / redirects.

    Returns (edges_written, edges_unresolved).
    """
    import multiprocessing as mp

    n_rg = pq.ParquetFile(edges_path).metadata.num_row_groups
    drop_sorted = np.sort(drop.astype(np.int64)) if len(drop) else np.array([], dtype=np.int64)
    # Set module globals BEFORE forking so all workers share them copy-on-write
    # (pickling a 14M-entry dict per worker would dominate runtime).
    global _TITLE_MAP, _DROP
    _TITLE_MAP = title_map
    _DROP = drop_sorted
    total = 0
    unresolved = 0
    schema = pa.schema([("source", pa.int64()), ("target", pa.int64())])
    writer = pq.ParquetWriter(out_path, schema)
    try:
        # Fork context (see _mp_context): Python 3.14 defaults to forkserver,
        # whose server process (spawned during the earlier scan_nodes pool)
        # would not see _TITLE_MAP/_DROP set afterwards. Fork children inherit
        # the parent's current globals via copy-on-write.
        with _mp_context().Pool(workers) as pool:
            for i, (src_i, dst_i, unres) in enumerate(
                    pool.imap(_resolve_edge_row_group,
                              [(edges_path, rg) for rg in range(n_rg)], chunksize=4)):
                unresolved += unres
                if len(src_i):
                    writer.write_table(pa.Table.from_arrays(
                        [pa.array(src_i, type=pa.int64()), pa.array(dst_i, type=pa.int64())],
                        schema=schema,
                    ))
                    total += len(src_i)
                if (i + 1) % 32 == 0:
                    log(f"  edges: resolved {i + 1}/{n_rg} row groups "
                        f"({total:,} kept, {unresolved:,} unresolved)")
    finally:
        writer.close()
    return total, unresolved


def load_into_hdb(tmp_uri: str, edges_int: str):
    """Load the int64 edge table into a temporary HyperStreamDB instance."""
    import hyperstreamdb

    tmp_dir = tmp_uri.removeprefix("file://")
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    table = hyperstreamdb.Table(tmp_uri)
    pf = pq.ParquetFile(edges_int)
    t0 = time.time()
    for rg in range(pf.metadata.num_row_groups):
        table.write(pf.read_row_group(rg))
        if (rg + 1) % 32 == 0:
            log(f"  hdb ingest: {rg + 1}/{pf.metadata.num_row_groups} row groups")
    table.commit()
    table.wait_for_background_tasks()
    log(f"  hdb ingest + commit in {time.time() - t0:.1f}s")
    return table


def largest_component(table) -> np.ndarray:
    """Node ids of the largest weakly connected component (computed in hdb)."""
    log("  running connected_components() in HyperStreamDB...")
    t0 = time.time()
    cc = table.connected_components()
    log(f"  connected_components() finished in {time.time() - t0:.1f}s")
    cc_df = cc.to_pandas() if hasattr(cc, "to_pandas") else pd.DataFrame(cc)

    counts = cc_df["component"].value_counts()
    biggest = counts.index[0]
    log(f"  {len(counts):,} components; largest has {counts.iloc[0]:,} nodes")
    return cc_df.loc[cc_df["component"] == biggest, "node"].to_numpy().astype("int64")


def hub_subgraph(table, cc_nodes: np.ndarray, max_nodes: int):
    """Extract the demo subgraph INSIDE HyperStreamDB: rank nodes by degree
    (degree_centrality), then expand a multi-hop induced subgraph (subgraph
    UDF) from the highest-degree hubs of the largest component.

    Returns (kept_nodes: np.ndarray[int64], edges_df: pd.DataFrame).
    """
    deg = table.degree_centrality()
    deg_df = deg.to_pandas() if hasattr(deg, "to_pandas") else pd.DataFrame(deg)
    cc_set = set(cc_nodes.tolist())
    deg_df = deg_df[deg_df["node"].isin(cc_set)]
    deg_df = deg_df.sort_values("degree", ascending=False)
    # Guard against NULL/NaN or non-positive node ids poisoning seed literals.
    hubs = [
        int(h) for h in deg_df["node"].head(20).tolist()
        if pd.notna(h) and int(h) > 0
    ]
    log(f"  top hubs by degree: {hubs[:5]}")

    nodes = np.array([], dtype=np.int64)
    res_df = pd.DataFrame(columns=["source", "target"])
    for hops in (1, 2, 3):
        t0 = time.time()
        res = table.subgraph(seeds=hubs, hops=hops, directed=False)
        res_df = res.to_pandas() if hasattr(res, "to_pandas") else pd.DataFrame(res)
        nodes = np.union1d(res_df["source"].unique(), res_df["target"].unique())
        log(f"  subgraph(hops={hops}) in {time.time() - t0:.1f}s -> "
            f"{len(nodes):,} nodes / {len(res_df):,} edges")
        if not max_nodes or len(nodes) >= max_nodes or hops == 3:
            break

    # Final size trim only (data manipulation, not graph traversal):
    # keep the highest-degree nodes inside the extracted subgraph.
    if max_nodes and len(nodes) > max_nodes:
        from collections import Counter

        cnt = Counter()
        for u, v in zip(res_df["source"], res_df["target"]):
            cnt[u] += 1
            cnt[v] += 1
        keep = np.array(sorted(n for n, _ in cnt.most_common(max_nodes)), dtype=np.int64)
        km = res_df["source"].isin(keep) & res_df["target"].isin(keep)
        res_df = res_df[km]
        nodes = keep

    return nodes.astype("int64"), res_df[["source", "target"]].reset_index(drop=True)


def embed_nodes(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    from sentence_transformers import SentenceTransformer

    log(f"  embedding {len(df):,} summaries with {model_name} (CPU)...")
    model = SentenceTransformer(model_name)
    texts = (df["title"].astype(str) + ". " + df["summary"].astype(str).str.slice(0, 512)).tolist()
    vecs = model.encode(texts, batch_size=256, show_progress_bar=True, convert_to_numpy=True)
    df = df.copy()
    df["embedding"] = [v.astype(np.float32).tolist() for v in vecs]
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dumps-dir", default=DEFAULT_DUMPS,
                        help="directory holding the full wiki parquets "
                             "(default: $HOME/data/hyperstreamdb)")
    parser.add_argument("--nodes", default=None,
                        help="default: <dumps-dir>/nodes.parquet")
    parser.add_argument("--edges", default=None,
                        help="default: <dumps-dir>/edges.parquet")
    parser.add_argument("--out_nodes", default=None,
                        help="default: <dumps-dir>/demo_nodes.parquet")
    parser.add_argument("--out_edges", default=None,
                        help="default: <dumps-dir>/demo_edges.parquet")
    parser.add_argument("--max-nodes", type=int, default=50_000,
                        help="Cap the demo to N nodes (0 = no cap)")
    parser.add_argument("--embed-model", default="BAAI/bge-large-en-v1.5",
                        help="SentenceTransformer model (1024-d default; "
                             "'all-MiniLM-L6-v2' is a fast 384-d fallback)")
    parser.add_argument("--no-embed", action="store_true",
                        help="Skip sentence-transformers embeddings "
                             "(UI falls back to fuzzy title match)")
    parser.add_argument("--workdir", default="/tmp/hdb_demo_build")
    parser.add_argument("--tmp-uri", default="file:///tmp/hdb_demo_cc")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 8,
                        help="Parallel processes for the CPU-bound resolve phase")
    args = parser.parse_args()

    dumps = os.path.abspath(os.path.expanduser(args.dumps_dir))
    args.nodes = args.nodes or os.path.join(dumps, "nodes.parquet")
    args.edges = args.edges or os.path.join(dumps, "edges.parquet")
    args.out_nodes = args.out_nodes or os.path.join(dumps, "demo_nodes.parquet")
    args.out_edges = args.out_edges or os.path.join(dumps, "demo_edges.parquet")

    os.makedirs(args.workdir, exist_ok=True)
    edges_int = os.path.join(args.workdir, "edges_int64.parquet")

    log(f"step 1/5: scanning nodes (redirect stubs + title map) on {args.workers} workers")
    bad, title_map = scan_nodes(args.nodes, args.workers)

    log(f"step 2/5: resolving edge endpoints to int64 curids from {args.edges}")
    n, unresolved = stream_int64_edges(args.edges, edges_int, bad, title_map, args.workers)
    log(f"  {n:,} clean int64 edges ({unresolved:,} endpoints unresolved and dropped)")

    log("step 3/5: connected components in HyperStreamDB")
    table = load_into_hdb(args.tmp_uri, edges_int)
    cc_nodes = largest_component(table)

    log("step 4/5: hub degree + induced subgraph extraction in HyperStreamDB")
    if args.max_nodes and len(cc_nodes) > args.max_nodes:
        keep, edges_demo = hub_subgraph(table, cc_nodes, args.max_nodes)
    else:
        keep = cc_nodes
        edges_all = pd.read_parquet(edges_int)
        mask = (np.isin(edges_all["source"].to_numpy(), keep)
                & np.isin(edges_all["target"].to_numpy(), keep))
        edges_demo = edges_all[mask]
    log(f"  demo graph: {len(keep):,} nodes / {len(edges_demo):,} edges")

    log("step 5/5: writing demo parquets")
    keep_sorted = np.sort(keep)
    pf = pq.ParquetFile(args.nodes)
    parts = []
    for rg in range(pf.metadata.num_row_groups):
        tbl = pf.read_row_group(rg, columns=["id", "title", "summary"])
        ids = pc.cast(tbl.column("id"), pa.int64()).to_numpy()
        sel = np.isin(ids, keep_sorted)
        parts.append(tbl.filter(sel))
    nodes_tbl = pa.concat_tables(parts)
    nodes_df = nodes_tbl.to_pandas()
    nodes_df["id"] = nodes_df["id"].astype("int64")

    if not args.no_embed:
        nodes_df = embed_nodes(nodes_df, args.embed_model)

    edges_out = pa.Table.from_pandas(
        edges_demo.astype("int64"),
        schema=pa.schema([("source", pa.int64()), ("target", pa.int64())]),
        preserve_index=False,
    )
    pq.write_table(pa.Table.from_pandas(nodes_df, preserve_index=False), args.out_nodes)
    pq.write_table(edges_out, args.out_edges)
    log(f"wrote {args.out_nodes} ({len(nodes_df):,} rows) and {args.out_edges} ({len(edges_demo):,} rows)")

    shutil.rmtree(args.workdir, ignore_errors=True)
    shutil.rmtree(args.tmp_uri.removeprefix("file://"), ignore_errors=True)
    log("done.")


if __name__ == "__main__":
    main()
