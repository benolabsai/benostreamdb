#!/usr/bin/env python3
"""Graph Benchmark Runner: Evaluates graph algorithms across engines.

Supported Engines:
- benostreamdb (Native CSR & Graph RAG graph engine)
- networkx (In-memory Python reference baseline)
- neo4j (Graph Database with GDS plugin)
- cugraph (GPU-accelerated graph analytics)

Algorithms:
- pagerank (damping=0.85, iterations=30)
- connected_components (weakly connected components)
- shortest_path (breadth-first search / shortest path)

Usage:
    python benchmarks/graph/run.py --nodes 20000 --edges 100000 --algorithms pagerank,connected_components,shortest_path
    python benchmarks/graph/run.py --graph-edges data/ldbc_sample.txt --engines benostreamdb,networkx --out results/graph_ldbc.md
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional

import numpy as np

CACHE_DIR = os.path.join(os.path.dirname(__file__), "data")


# Public, citable edge-list datasets (SNAP). Files are gzip-compressed with a
# `#`-prefixed comment header; we strip the header and cache a plain `.tsv`.
GRAPH_DATASETS = {
    "snap-web-google": "https://snap.stanford.edu/data/web-Google.txt.gz",
    "snap-roadnet-ca": "https://snap.stanford.edu/data/roadNet-CA.txt.gz",
    "snap-com-livejournal": "https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz",
}


def ensure_named_dataset(name: str, max_edges: int = 0) -> str:
    """Download + decompress a named public dataset into a cached `.tsv`.

    `max_edges` caps the number of edges kept (0 = all), so a large SNAP graph
    can still be run through the single-threaded NetworkX baseline.
    """
    import gzip
    import urllib.request

    url = GRAPH_DATASETS.get(name)
    if not url:
        raise SystemExit(f"unknown dataset '{name}' (have: {sorted(GRAPH_DATASETS)})")
    os.makedirs(CACHE_DIR, exist_ok=True)
    suffix = f"_{max_edges}" if max_edges else ""
    out_path = os.path.join(CACHE_DIR, f"{name}{suffix}.tsv")
    if os.path.exists(out_path):
        return out_path

    gz_path = os.path.join(CACHE_DIR, f"{name}.txt.gz")
    if not os.path.exists(gz_path):
        print(f"downloading {url} ...", flush=True)
        urllib.request.urlretrieve(url, gz_path)

    print(f"extracting {gz_path} -> {out_path} ...", flush=True)
    kept = 0
    with gzip.open(gz_path, "rt", encoding="utf-8", errors="ignore") as src, \
            open(out_path, "w", encoding="utf-8") as dst:
        for line in src:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 2:
                dst.write(f"{parts[0]}\t{parts[1]}\n")
                kept += 1
                if max_edges and kept >= max_edges:
                    break
    print(f"wrote {out_path} ({kept} edges)", flush=True)
    return out_path


def ensure_graph_dataset(
    edges_file: Optional[str], nodes: int, edges_target: int, seed: int = 42
) -> str:
    """Returns path to an edge list file (source, target per line)."""
    if edges_file and os.path.exists(edges_file):
        return edges_file

    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"synth_graph_n{nodes}_e{edges_target}_s{seed}.txt")
    if os.path.exists(cache_path):
        return cache_path

    print(f"Generating synthetic scale-free graph (nodes={nodes}, target_edges={edges_target})...", flush=True)
    import networkx as nx

    # Power-law / scale-free distribution typical of social / LDBC networks
    m_attach = max(1, edges_target // nodes)
    g = nx.barabasi_albert_graph(nodes, m_attach, seed=seed)
    edge_list = np.array(list(g.edges()), dtype=np.int64)

    np.savetxt(cache_path, edge_list, fmt="%d")
    print(f"Saved generated graph to {cache_path} ({len(edge_list)} edges)", flush=True)
    return cache_path


def run_benostreamdb(edges: np.ndarray, algorithm: str, source: int, target: int, damping: float, iterations: int) -> Dict[str, Any]:
    import benostreamdb as bsdb
    import pyarrow as pa

    tmpdir = tempfile.mkdtemp(prefix="bsdb_graph_")
    try:
        t_build_0 = time.perf_counter()
        schema = pa.schema([
            ("source", pa.uint64()),
            ("target", pa.uint64()),
        ])
        table = bsdb.Table.create(tmpdir, schema)
        # Configure Forward and Reverse CSR Graph indexes
        table.add_index(
            "source", {"type": "graph", "src_column": "source", "dst_column": "target"}
        )
        table.add_index(
            "target", {"type": "graph", "src_column": "target", "dst_column": "source"}
        )
        arr_src = pa.array(edges[:, 0].astype(np.uint64), type=pa.uint64())
        arr_dst = pa.array(edges[:, 1].astype(np.uint64), type=pa.uint64())
        batch = pa.Table.from_arrays([arr_src, arr_dst], names=["source", "target"])
        table.insert(batch)
        table.commit()
        table.wait_for_background_tasks()
        build_s = time.perf_counter() - t_build_0

        index_bytes = sum(
            os.path.getsize(os.path.join(r, f))
            for r, _, files in os.walk(tmpdir)
            for f in files
            if ".puffin" in f
            or ".graph" in f
            or ".csr" in f
            or ".bin" in f
            or ".idx" in f
        )

        t0 = time.perf_counter()
        if algorithm == "pagerank":
            res = table.pagerank(damping=damping, iterations=iterations).to_pandas()
            n_out = len(res)
        elif algorithm == "connected_components":
            res = table.connected_components().to_pandas()
            n_out = int(res["component"].nunique()) if "component" in res.columns else len(res)
        elif algorithm == "shortest_path":
            res = table.shortest_path(source, target)
            n_out = len(res)
        elif algorithm == "subgraph":
            res = table.subgraph([source], hops=2).to_pandas()
            n_out = len(res)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        elapsed = time.perf_counter() - t0

        return {
            "engine": "benostreamdb",
            "available": True,
            "build_s": build_s,
            "index_kb": round(index_bytes / 1024, 1),
            "seconds": elapsed,
            "result_size": n_out,
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_networkx(edges: np.ndarray, algorithm: str, source: int, target: int, damping: float, iterations: int) -> Dict[str, Any]:
    if not importlib.util.find_spec("networkx"):
        return {"engine": "networkx", "available": False, "error": "networkx not installed"}
    import networkx as nx

    t_build_0 = time.perf_counter()
    create_using = nx.DiGraph if algorithm == "pagerank" else nx.Graph
    g = nx.from_edgelist(((int(a), int(b)) for a, b in edges), create_using=create_using)
    build_s = time.perf_counter() - t_build_0

    t0 = time.perf_counter()
    if algorithm == "pagerank":
        res = nx.pagerank(g, alpha=damping, max_iter=iterations)
        n_out = len(res)
    elif algorithm == "connected_components":
        res = list(nx.connected_components(g))
        n_out = len(res)
    elif algorithm == "shortest_path":
        try:
            res = nx.shortest_path(g, source=source, target=target)
            n_out = len(res)
        except Exception:
            n_out = 0
    elif algorithm == "subgraph":
        res = nx.ego_graph(g, n=source, radius=2)
        n_out = res.number_of_edges()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    elapsed = time.perf_counter() - t0

    return {
        "engine": "networkx",
        "available": True,
        "build_s": build_s,
        "seconds": elapsed,
        "result_size": n_out,
    }


def run_cugraph(edges: np.ndarray, algorithm: str, source: int, target: int, damping: float, iterations: int) -> Dict[str, Any]:
    if not importlib.util.find_spec("cugraph") or not importlib.util.find_spec("cudf"):
        return {"engine": "cugraph", "available": False, "error": "cugraph/cudf not installed"}
    import cugraph
    import cudf

    t_build_0 = time.perf_counter()
    gdf = cudf.DataFrame({"src": edges[:, 0], "dst": edges[:, 1]})
    G = cugraph.Graph(directed=True)
    G.from_cudf_edgelist(gdf, source="src", destination="dst")
    build_s = time.perf_counter() - t_build_0

    t0 = time.perf_counter()
    if algorithm == "pagerank":
        res = cugraph.pagerank(G, alpha=damping, max_iter=iterations)
        n_out = len(res)
    elif algorithm == "connected_components":
        res = cugraph.weakly_connected_components(G)
        n_out = len(res)
    elif algorithm == "shortest_path":
        res = cugraph.bfs(G, start=source)
        res_target = res[res["vertex"] == target]
        n_out = len(res_target)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    elapsed = time.perf_counter() - t0

    return {
        "engine": "cugraph",
        "available": True,
        "build_s": build_s,
        "seconds": elapsed,
        "result_size": n_out,
    }


def run_neo4j(
    edges: np.ndarray, algorithm: str, source: int, target: int, damping: float, iterations: int
) -> Dict[str, Any]:
    """Neo4j with the Graph Data Science plugin (Tier-1 graph reference)."""
    if not importlib.util.find_spec("neo4j"):
        return {"engine": "neo4j", "available": False, "error": "neo4j not installed"}
    from neo4j import GraphDatabase

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "benchpass")
    procs = {
        "pagerank": "gds.pageRank.stream",
        "connected_components": "gds.wcc.stream",
        "shortest_path": "gds.shortestPath.dijkstra.stream",
    }
    proc = procs.get(algorithm)
    if proc is None:
        return {"engine": "neo4j", "available": False, "error": f"unknown algorithm {algorithm}"}

    driver = GraphDatabase.driver(uri, auth=(user, password))
    # Load + project are setup, not query time, so time them separately.
    t_load = time.perf_counter()
    rows = [(int(a), int(b)) for a, b in edges]
    try:
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            for i in range(0, len(rows), 50_000):
                session.run(
                    "UNWIND $rows AS r MERGE (a:Node {id: r[0]}) "
                    "MERGE (b:Node {id: r[1]}) MERGE (a)-[:E]->(b)",
                    rows=rows[i : i + 50_000],
                )
            session.run("CALL gds.graph.drop('bench', false)").consume()
            session.run("CALL gds.graph.project('bench', 'Node', 'E')").consume()
        build_s = time.perf_counter() - t_load

        t0 = time.perf_counter()
        with driver.session() as session:
            if algorithm == "shortest_path":
                res = session.run(
                    f"CALL {proc}('bench', {{sourceNode: $src, targetNode: $tgt}})",
                    src=int(source),
                    tgt=int(target),
                )
            else:
                res = session.run(f"CALL {proc}('bench')")
            out_rows = list(res)
        return {
            "engine": "neo4j",
            "available": True,
            "build_s": build_s,
            "seconds": time.perf_counter() - t0,
            "result_size": len(out_rows),
        }
    except Exception as exc:  # server unreachable / GDS missing
        return {"engine": "neo4j", "available": False, "error": str(exc)[:200]}
    finally:
        driver.close()


def main():
    parser = argparse.ArgumentParser(description="Graph Algorithm Benchmark Runner")
    parser.add_argument("--graph-edges", default=None, help="Path to edgelist file (txt/tsv)")
    parser.add_argument("--dataset", default=None, help="Named public dataset (e.g. snap-web-google, snap-roadnet-ca, snap-com-livejournal)")
    parser.add_argument("--max-edges", type=int, default=0, help="Cap edges for a named dataset (0 = all)")
    parser.add_argument("--nodes", type=int, default=20000, help="Number of nodes for synthetic graph")
    parser.add_argument("--edges", type=int, default=100000, help="Number of edges for synthetic graph")
    parser.add_argument("--engines", default="benostreamdb,networkx", help="Comma-separated engines to run")
    parser.add_argument("--algorithms", default="pagerank,connected_components,shortest_path", help="Comma-separated algorithms")
    parser.add_argument("--source", type=int, default=0, help="Source node for shortest path")
    parser.add_argument("--target", type=int, default=10, help="Target node for shortest path")
    parser.add_argument("--damping", type=float, default=0.85, help="PageRank damping factor")
    parser.add_argument("--iterations", type=int, default=30, help="PageRank iterations")
    parser.add_argument("--out", default=None, help="Output markdown path")
    args = parser.parse_args()

    if args.dataset:
        edges_path = ensure_named_dataset(args.dataset, args.max_edges)
    else:
        edges_path = ensure_graph_dataset(args.graph_edges, args.nodes, args.edges)
    edges = np.loadtxt(edges_path, dtype=np.int64, ndmin=2)
    num_nodes = len(np.unique(edges))
    num_edges = len(edges)

    print(f"Graph loaded: {num_nodes} nodes, {num_edges} edges from {edges_path}", flush=True)

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    algorithms = [a.strip() for a in args.algorithms.split(",") if a.strip()]

    runners = {
        "benostreamdb": run_benostreamdb,
        "networkx": run_networkx,
        "neo4j": run_neo4j,
        "cugraph": run_cugraph,
    }

    results = []
    for algo in algorithms:
        for engine in engines:
            runner = runners.get(engine)
            if not runner:
                print(f"Skipping unknown engine: {engine}")
                continue
            print(f"Running {engine} on {algo}...", flush=True)
            res = runner(edges, algo, args.source, args.target, args.damping, args.iterations)
            res["algorithm"] = algo
            res["nodes"] = num_nodes
            res["edges"] = num_edges
            results.append(res)
            if res.get("available"):
                print(f"  -> {engine} {algo}: {res['seconds'] * 1000:.2f} ms (build: {res.get('build_s', 0):.3f}s, size: {res.get('result_size', 0)})", flush=True)
            else:
                print(f"  -> {engine} {algo}: NOT AVAILABLE ({res.get('error')})", flush=True)

    # Format Markdown Report
    lines = [
        "# Graph Benchmark Results",
        "",
        f"- **Graph Nodes**: {num_nodes:,}",
        f"- **Graph Edges**: {num_edges:,}",
        f"- **Source Dataset**: `{os.path.basename(edges_path)}`",
        "",
        "| Algorithm | Engine | Status | Build (s) | Execution Latency (ms) | Result Size | Speedup vs NetworkX |",
        "|---|---|---|---|---|---|---|",
    ]

    # Calculate speedup relative to networkx
    nx_times = {
        r["algorithm"]: r["seconds"]
        for r in results
        if r["engine"] == "networkx" and r.get("available")
    }

    for r in results:
        status = "✅ Pass" if r.get("available") else "❌ Unavailable"
        build_str = f"{r.get('build_s', 0):.3f}s" if r.get("available") else "-"
        lat_str = f"{r['seconds'] * 1000:.2f} ms" if r.get("available") else "-"
        size_str = str(r.get("result_size", "-")) if r.get("available") else "-"

        speedup_str = "-"
        if r.get("available") and r["algorithm"] in nx_times:
            nx_t = nx_times[r["algorithm"]]
            if r["seconds"] > 0:
                speedup = nx_t / r["seconds"]
                speedup_str = f"{speedup:.2f}x" if r["engine"] != "networkx" else "1.00x (baseline)"

        lines.append(
            f"| {r['algorithm']} | {r['engine']} | {status} | {build_str} | {lat_str} | {size_str} | {speedup_str} |"
        )

    report_md = "\n".join(lines) + "\n"
    print("\n" + report_md)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report_md)
        print(f"Wrote graph benchmark report to {args.out}", flush=True)


if __name__ == "__main__":
    main()
