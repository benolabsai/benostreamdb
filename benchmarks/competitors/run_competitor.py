#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Competitor runner for the BenoStreamDB benchmark plan.

Runs a competitor under the same resource envelope and parameter set as the
BenoStreamDB harness (`benchmarks/ann_benchmarks/run.py`) and emits the same
metrics, so results are directly comparable.

Workload families:

* ``vector`` — ANN over an ANN-Benchmarks HDF5 dataset.
  Engines: ``faiss``, ``hnswlib`` (embedded), ``pgvector``, ``lancedb``,
  ``opensearch`` (server-backed).
* ``graph``  — a graph algorithm over an edge list.
  Engines: ``networkx`` (reference), ``neo4j`` (GDS).
* ``sql``    — an analytical query. Engines: ``duckdb``.

Adapters degrade gracefully: if the client library is not installed, or a
server is unreachable, the runner reports it rather than crashing.

Usage:
    python benchmarks/competitors/run_competitor.py --engine faiss \\
        --dataset sift-128-euclidean --cores 8 --ram-gb 16 \\
        --m 16 --ef-construction 200 --ef-search 200 --queries 500

    python benchmarks/competitors/run_competitor.py --engine pgvector \\
        --dsn "postgresql://user:pass@localhost/bench" --dataset glove-100-angular

    python benchmarks/competitors/run_competitor.py --engine opensearch \\
        --host http://localhost:9200 --dataset sift-128-euclidean

    python benchmarks/competitors/run_competitor.py --engine networkx \\
        --graph-edges edges.txt --algorithm pagerank

    python benchmarks/competitors/run_competitor.py --engine duckdb \\
        --sql "SELECT count(*) FROM read_parquet('hits.parquet')"
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import os
import platform
import shutil
import tempfile
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

ANN_BASE = "http://ann-benchmarks.com"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

DATASET_METRIC = {
    "sift-128-euclidean": "l2",
    "gist-960-euclidean": "l2",
    "fashion-mnist-784-euclidean": "l2",
    "mnist-784-euclidean": "l2",
    "glove-100-angular": "cosine",
    "glove-200-angular": "cosine",
    "nytimes-256-angular": "cosine",
    "lastfm-64-dot": "inner_product",
}


# --------------------------------------------------------------------------- #
# Resource envelope
# --------------------------------------------------------------------------- #
# Engines whose runtime reserves a large *virtual* address space (CUDA/RAPIDS
# managed memory, wgpu). `RLIMIT_AS` limits virtual address space, not resident
# memory, so applying the RAM envelope to these engines makes the `dlopen` of
# their libraries fail with a misleading "cannot open shared object file"
# (the `mmap` returns ENOMEM). The container cgroup
# (`deploy.resources.limits.memory`) already enforces the physical RAM envelope,
# so `RLIMIT_AS` is both redundant and wrong for GPU engines.
_GPU_ENGINES = {"cugraph"}


def apply_envelope(
    cores: int, ram_gb: float, engine: str = "", device: str = "cpu"
) -> None:
    """Best-effort process limits so the competitor matches BenoStreamDB.

    Threads: set the common BLAS/OMP env vars (must be set before numpy/faiss
    import to take effect; the runner sets them as early as possible). RAM: use
    ``RLIMIT_AS`` when ``ram_gb`` is given, so an over-budget run fails loudly
    instead of silently using more memory than BenoStreamDB was given — except
    for GPU engines, where virtual address space far exceeds physical RAM (see
    ``_GPU_ENGINES``).
    """
    if cores > 0:
        for var in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ[var] = str(cores)
    if ram_gb > 0 and engine not in _GPU_ENGINES and device != "gpu":
        try:
            import resource

            limit = int(ram_gb * 1024**3)
            _, hard = resource.getrlimit(resource.RLIMIT_AS)
            resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
        except Exception as exc:  # pragma: no cover - platform dependent
            print(f"warning: could not set RLIMIT_AS: {exc}", flush=True)


# --------------------------------------------------------------------------- #
# Vector adapters
# --------------------------------------------------------------------------- #
@dataclass
class VectorAdapter:
    """Build a vector index and answer top-k queries.

    ``search`` receives ``ef_search`` and ``metric`` explicitly (not via hidden
    attribute state) so every adapter is driven identically.
    """

    name: str
    module: str
    server_side: bool = field(default=False)
    _impl: Any = field(default=None, repr=False)

    def available(self) -> bool:
        try:
            importlib.import_module(self.module)
            return True
        except Exception:
            return False

    def build(self, train: np.ndarray, params: dict, ctx: dict):  # -> handle
        raise NotImplementedError

    def search(
        self, handle, queries: np.ndarray, k: int, ef_search: int, metric: str
    ) -> np.ndarray:
        raise NotImplementedError

    def index_bytes(self, handle) -> int:
        return 0


def _vec_literal(v: np.ndarray) -> str:
    """pgvector text literal: ``[a,b,c]``."""
    return "[" + ",".join(repr(float(x)) for x in v) + "]"


def _dir_size(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for fn in files:
            total += os.path.getsize(os.path.join(root, fn))
    return total


class FaissAdapter(VectorAdapter):
    def __init__(self) -> None:
        super().__init__(name="faiss", module="faiss")

    def build(self, train, params, ctx):
        import faiss

        dim = train.shape[1]
        metric = params.get("metric", "l2")
        faiss_metric = (
            faiss.METRIC_INNER_PRODUCT
            if metric in ("inner_product", "cosine")
            else faiss.METRIC_L2
        )
        index = faiss.IndexHNSWFlat(dim, params["m"], faiss_metric)
        index.hnsw.efConstruction = params["ef_construction"]
        data = np.ascontiguousarray(train)
        if metric == "cosine":
            faiss.normalize_L2(data)
        index.add(data)
        if ctx.get("device") == "gpu":
            # Requires faiss-gpu; falls back to the CPU index when unavailable.
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
            except Exception as exc:
                print(f"faiss: GPU unavailable, using CPU index ({exc})", flush=True)
        return index

    def search(self, handle, queries, k, ef_search, metric):
        import faiss

        handle.hnsw.efSearch = ef_search
        q = np.ascontiguousarray(queries)
        if metric == "cosine":
            faiss.normalize_L2(q)
        _, ids = handle.search(q, k)
        return ids


class HnswlibAdapter(VectorAdapter):
    def __init__(self) -> None:
        super().__init__(name="hnswlib", module="hnswlib")

    def build(self, train, params, ctx):
        import hnswlib

        space = "cosine" if params.get("metric") == "cosine" else "l2"
        index = hnswlib.Index(space=space, dim=train.shape[1])
        index.init_index(
            max_elements=train.shape[0],
            ef_construction=params["ef_construction"],
            M=params["m"],
        )
        index.add_items(train, np.arange(train.shape[0]))
        return index

    def search(self, handle, queries, k, ef_search, metric):
        handle.set_ef(ef_search)
        labels, _ = handle.knn_query(queries, k=k)
        return labels


class PgvectorAdapter(VectorAdapter):
    def __init__(self) -> None:
        super().__init__(name="pgvector", module="psycopg", server_side=True)

    def build(self, train, params, ctx):
        import psycopg

        dsn = ctx.get("dsn") or os.environ.get("PGVECTOR_DSN")
        if not dsn:
            raise RuntimeError("pgvector requires --dsn or PGVECTOR_DSN")
        conn = psycopg.connect(dsn, autocommit=True)
        cur = conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute("DROP TABLE IF EXISTS bsdb_bench")
        dim = train.shape[1]
        cur.execute(
            f"CREATE TABLE bsdb_bench (id int primary key, embedding vector({dim}))"
        )
        metric = params.get("metric", "l2")
        opclass = (
            "vector_ip_ops"
            if metric == "inner_product"
            else ("vector_cosine_ops" if metric == "cosine" else "vector_l2_ops")
        )
        with cur.copy("COPY bsdb_bench (id, embedding) FROM STDIN") as copy:
            for i, row in enumerate(train):
                copy.write_row((int(i), _vec_literal(row)))
        # DDL cannot take bind parameters, so inline the (validated) integers.
        cur.execute(
            f"CREATE INDEX ON bsdb_bench USING hnsw (embedding {opclass}) "
            f"WITH (m = {int(params['m'])}, ef_construction = {int(params['ef_construction'])})"
        )
        return conn

    def search(self, handle, queries, k, ef_search, metric):
        op = {
            "l2": "<=>",
            "cosine": "<=>",
            "inner_product": "<#>",
        }.get(metric, "<=>")
        cur = handle.cursor()
        cur.execute(f"SET hnsw.ef_search = {int(ef_search)}")
        out = []
        for q in queries:
            cur.execute(
                f"SELECT id FROM bsdb_bench ORDER BY embedding {op} %s::vector LIMIT %s",
                (_vec_literal(q), int(k)),
            )
            out.append([r[0] for r in cur.fetchall()])
        return np.array(out)

    def index_bytes(self, handle):
        cur = handle.cursor()
        cur.execute("SELECT pg_total_relation_size('bsdb_bench')")
        return int(cur.fetchone()[0])


class LanceDbAdapter(VectorAdapter):
    """LanceDB vector index.

    ``index_type`` selects the ANN family, so both of LanceDB's relevant
    algorithms can be reported side by side:

    * ``ivf_pq``  — LanceDB's **default** disk-ANN index (IVF + product
      quantization). This is what ``create_index`` builds with no config, so it
      is the representative "what a LanceDB user gets" baseline.
    * ``hnsw_sq`` — HNSW with scalar quantization, the closest LanceDB has to
      the plain-float HNSW the other engines use (LanceDB has no plain-float
      HNSW). Kept so the comparison can also be HNSW-vs-HNSW.
    """

    def __init__(self, index_type: str = "ivf_pq") -> None:
        name = "lancedb" if index_type == "ivf_pq" else "lancedb_hnsw"
        super().__init__(name=name, module="lancedb")
        self.index_type = index_type

    def build(self, train, params, ctx):
        import lancedb
        import pyarrow as pa

        uri = ctx.get("path") or tempfile.mkdtemp(prefix="lancedb_bench_")
        db = lancedb.connect(uri)
        tbl = pa.table(
            {
                "id": pa.array(np.arange(len(train), dtype=np.int64)),
                "vector": pa.array(
                    [row.tolist() for row in train], type=pa.list_(pa.float32())
                ),
            }
        )
        t = db.create_table("bench", data=tbl, mode="overwrite")
        metric = "cosine" if params.get("metric") == "cosine" else "l2"
        # New unified API: the first positional arg is the vector column name.
        if self.index_type == "hnsw_sq":
            from lancedb.index import HnswSq

            t.create_index(
                "vector",
                config=HnswSq(
                    distance_type=metric,
                    m=int(params["m"]),
                    ef_construction=int(params["ef_construction"]),
                ),
            )
        else:
            # IVF_PQ with LanceDB's defaults (num_partitions / num_sub_vectors
            # auto-selected) — the disk-ANN index LanceDB builds by default.
            from lancedb.index import IvfPq

            t.create_index("vector", config=IvfPq(distance_type=metric))
        return (t, uri)

    def search(self, handle, queries, k, ef_search, metric):
        t, _uri = handle
        out = []
        for q in queries:
            # Project only `id`: the other engines return ids only, so fetching
            # the vector payload here would make LanceDB do strictly more work.
            rows = t.search(q.tolist()).select(["id"]).limit(int(k)).to_list()
            out.append([int(r["id"]) for r in rows])
        return np.array(out)

    def index_bytes(self, handle):
        _t, uri = handle
        return _dir_size(uri)


def _opensearch_client(host: str):
    """Return (client, helpers) for the OpenSearch server at `host`.

    The compose stack runs OpenSearch, so we use the opensearch-py client; the
    elasticsearch 8.x client hard-rejects non-Elasticsearch servers with
    "UnsupportedProductError".
    """
    from opensearchpy import OpenSearch, helpers

    return OpenSearch(host), helpers


class OpenSearchAdapter(VectorAdapter):
    """OpenSearch (dense_vector + HNSW kNN)."""

    def __init__(self) -> None:
        super().__init__(name="opensearch", module="opensearchpy", server_side=True)

    def build(self, train, params, ctx):
        host = ctx.get("host") or os.environ.get("ES_URL") or "http://localhost:9200"
        es, helpers = _opensearch_client(host)
        dim = train.shape[1]
        metric = params.get("metric", "l2")
        # OpenSearch kNN space types differ from Elasticsearch's similarity names.
        space_type = {
            "l2": "l2",
            "cosine": "cosinesimil",
            "inner_product": "innerproduct",
        }.get(metric, "l2")
        es.indices.delete(index="bsdb-bench", ignore_unavailable=True)
        es.indices.create(
            index="bsdb-bench",
            body={
                "settings": {
                    "index": {
                        "knn": True,
                        "knn.algo_param.ef_search": int(params.get("ef_search", 100)),
                    }
                },
                "mappings": {
                    "properties": {
                        "vector": {
                            "type": "knn_vector",
                            "dimension": dim,
                            "method": {
                                "name": "hnsw",
                                "space_type": space_type,
                                "engine": "lucene",
                                "parameters": {
                                    "m": int(params["m"]),
                                    "ef_construction": int(params["ef_construction"]),
                                },
                            },
                        }
                    }
                },
            },
        )
        def _row_source(i, row):
            vec = row.tolist()
            # OpenSearch rejects an all-zero vector for cosinesimil ("zero vector
            # is not supported"); nudge it to a tiny vector. A zero vector is
            # orthogonal to everything, so this preserves the similarity result.
            if space_type == "cosinesimil" and not any(vec):
                vec = list(vec)
                vec[0] = 1e-6
            return {"_index": "bsdb-bench", "_id": int(i), "_source": {"vector": vec}}

        actions = (_row_source(i, row) for i, row in enumerate(train))
        helpers.bulk(es, actions)
        es.indices.refresh(index="bsdb-bench")
        return es

    def search(self, handle, queries, k, ef_search, metric):
        es = handle
        out = []
        for q in queries:
            body = {
                "size": int(k),
                "query": {"knn": {"vector": {"vector": q.tolist(), "k": int(k)}}},
                "_source": False,
            }
            res = es.search(index="bsdb-bench", body=body)
            out.append([int(h["_id"]) for h in res["hits"]["hits"]])
        return np.array(out)

    def index_bytes(self, handle):
        # Server-side size is not meaningful per index; report the count instead.
        try:
            return int(handle.count(index="bsdb-bench")["count"])
        except Exception:
            return 0


def vector_adapters() -> dict[str, VectorAdapter]:
    return {
        "faiss": FaissAdapter(),
        "hnswlib": HnswlibAdapter(),
        "pgvector": PgvectorAdapter(),
        # LanceDB is reported twice: its default disk-ANN index (IVF_PQ) and the
        # scalar-quantized HNSW variant, so the comparison covers both families.
        "lancedb": LanceDbAdapter("ivf_pq"),
        "lancedb_hnsw": LanceDbAdapter("hnsw_sq"),
        "opensearch": OpenSearchAdapter(),
    }


# --------------------------------------------------------------------------- #
# Vector workload
# --------------------------------------------------------------------------- #
def download(dataset: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{dataset}.hdf5")
    if os.path.exists(path):
        return path
    url = f"{ANN_BASE}/{dataset}.hdf5"
    print(f"downloading {url} ...", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(path, "wb") as out:
        shutil.copyfileobj(resp, out)
    return path


def load_dataset(path: str):
    import h5py

    with h5py.File(path, "r") as f:
        train = np.array(f["train"], dtype=np.float32)
        test = np.array(f["test"], dtype=np.float32)
        neighbors = np.array(f["neighbors"], dtype=np.int64)
    return train, test, neighbors


def run_vector(adapter: VectorAdapter, args) -> dict:
    metric = args.metric or DATASET_METRIC.get(args.dataset, "l2")
    train, test, neighbors = load_dataset(download(args.dataset))
    if args.limit and args.limit < len(train):
        train = train[: args.limit]
        neighbors = None  # ground truth invalid for a subset
    n, dim = train.shape

    params = {
        "m": args.m,
        "ef_construction": args.ef_construction,
        "ef_search": args.ef_search,
        "metric": metric,
    }
    ctx = {
        "dsn": args.dsn,
        "host": args.host,
        "path": args.path,
        "device": args.device,
    }

    t0 = time.time()
    handle = adapter.build(train, params, ctx)
    build_s = time.time() - t0

    if neighbors is None:
        q = min(args.queries, len(test))
        truth = []
        for i in range(q):
            d = np.linalg.norm(train - test[i], axis=1)
            truth.append(np.argsort(d)[: args.k])
        neighbors = np.array(truth)

    q = min(args.queries, len(test))
    k = args.k

    latencies = []
    hits = 0.0
    for i in range(q):
        t1 = time.perf_counter()
        ids = adapter.search(handle, test[i : i + 1], k, args.ef_search, metric)
        latencies.append(time.perf_counter() - t1)
        got = set(int(x) for x in ids[0].tolist())
        truth_i = set(int(x) for x in neighbors[i][:k].tolist())
        hits += len(got & truth_i) / k

    recall = hits / q
    total = sum(latencies) or 1e-9
    return {
        "engine": adapter.name,
        "dataset": args.dataset,
        "workload": "vector_ann",
        "device": args.device,
        "n": int(n),
        "dim": int(dim),
        "index": params,
        "queries": q,
        "k": k,
        "build_s": round(build_s, 3),
        "index_mb": round(adapter.index_bytes(handle) / 1e6, 2),
        "recall_at_k": round(recall, 4),
        "qps": round(q / total, 1),
        "p50_ms": round(float(np.percentile(latencies, 50) * 1000), 3),
        "p99_ms": round(float(np.percentile(latencies, 99) * 1000), 3),
    }


# --------------------------------------------------------------------------- #
# Graph workload
# --------------------------------------------------------------------------- #
def _graph_networkx(args, edges) -> dict:
    import networkx as nx

    g = nx.from_edgelist((int(a), int(b)) for a, b in edges)
    t0 = time.time()
    if args.algorithm == "pagerank":
        result = nx.pagerank(g)
        n_out = len(result)
    elif args.algorithm == "connected_components":
        result = list(nx.connected_components(g))
        n_out = len(result)
    elif args.algorithm == "shortest_path":
        result = nx.single_source_shortest_path_length(g, args.source)
        n_out = len(result)
    else:
        raise RuntimeError(f"unknown algorithm {args.algorithm}")
    return {
        "engine": "networkx",
        "workload": "graph",
        "available": True,
        "algorithm": args.algorithm,
        "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(),
        "seconds": round(time.time() - t0, 3),
        "result_size": n_out,
    }


def _graph_neo4j(args, edges) -> dict:
    from neo4j import GraphDatabase

    uri = args.uri or os.environ.get("NEO4J_URI") or "bolt://localhost:7687"
    user = args.user or os.environ.get("NEO4J_USER") or "neo4j"
    password = args.password or os.environ.get("NEO4J_PASSWORD") or "neo4j"
    driver = GraphDatabase.driver(uri, auth=(user, password))

    procs = {
        "pagerank": "gds.pageRank.stream",
        "connected_components": "gds.wcc.stream",
        "shortest_path": "gds.shortestPath.dijkstra.stream",
    }
    proc = procs.get(args.algorithm)
    if proc is None:
        raise RuntimeError(f"unknown algorithm {args.algorithm}")

    # Load + project are setup, not query time. Time them separately so the
    # reported `seconds` is the algorithm only (comparable to the other
    # engines, which also exclude ingestion).
    t_load = time.time()
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        session.run("DROP CONSTRAINT node_id IF EXISTS")
        session.run("CREATE CONSTRAINT node_id FOR (n:Node) REQUIRE n.id IS UNIQUE")
        edges_list = [(int(a), int(b)) for a, b in edges]
        batch_size = 10000
        for i in range(0, len(edges_list), batch_size):
            session.run(
                "UNWIND $rows AS r MERGE (a:Node {id: r[0]}) "
                "MERGE (b:Node {id: r[1]}) MERGE (a)-[:E]->(b)",
                rows=edges_list[i:i + batch_size],
            )
        session.run("CALL gds.graph.drop('bench', false)").consume()
        session.run("CALL gds.graph.project('bench', 'Node', 'E')").consume()
    load_s = time.time() - t_load

    t0 = time.time()
    with driver.session() as session:
        if args.algorithm == "shortest_path":
            res = session.run(
                f"CALL {proc}('bench', {{sourceNode: $src}})",
                src=int(args.source),
            )
        else:
            res = session.run(f"CALL {proc}('bench')")
        rows = list(res)
    return {
        "engine": "neo4j",
        "workload": "graph",
        "available": True,
        "algorithm": args.algorithm,
        "edges": len(edges),
        "load_s": round(load_s, 3),
        "seconds": round(time.time() - t0, 3),
        "result_size": len(rows),
    }


def _graph_benostreamdb(args, edges) -> dict:
    import benostreamdb as bsdb
    import pyarrow as pa
    import tempfile
    import shutil

    tmpdir = tempfile.mkdtemp(prefix="bsdb_graph_bench_")
    try:
        t_build_0 = time.time()
        schema = pa.schema([
            ("source", pa.uint64()),
            ("target", pa.uint64()),
        ])
        table = bsdb.Table.create(tmpdir, schema)
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
        build_s = round(time.time() - t_build_0, 3)

        damping = float(getattr(args, "damping", 0.85))
        iterations = int(getattr(args, "iterations", 30))

        t0 = time.time()
        if args.algorithm == "pagerank":
            res = table.pagerank(damping=damping, iterations=iterations).to_pandas()
            n_out = len(res)
        elif args.algorithm == "connected_components":
            res = table.connected_components().to_pandas()
            n_out = int(res["component"].nunique()) if "component" in res.columns else len(res)
        elif args.algorithm == "shortest_path":
            target = getattr(args, "target", None)
            if target is not None:
                res = table.shortest_path(int(args.source), int(target))
                n_out = len(res)
            else:
                res = table.graph_neighbors(int(args.source), hops=int(getattr(args, "hops", 2)))
                n_out = len(res)
        else:
            raise RuntimeError(f"unknown algorithm {args.algorithm}")
        elapsed = round(time.time() - t0, 3)
        return {
            "engine": "benostreamdb",
            "workload": "graph",
            "available": True,
            "algorithm": args.algorithm,
            "edges": len(edges),
            "build_seconds": build_s,
            "seconds": elapsed,
            "result_size": n_out,
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _graph_cugraph(args, edges) -> dict:
    if not importlib.util.find_spec("cugraph") or not importlib.util.find_spec("cudf"):
        return {
            "engine": "cugraph",
            "workload": "graph",
            "available": False,
            "error": "cugraph and cudf required (pip/conda install cugraph)",
        }
    # Configure RMM *before* importing cudf/cugraph. Managed (unified) memory lets
    # the device pool spill to host RAM instead of OOMing — important on a
    # workstation GPU where the desktop compositor (Wayland) holds VRAM.
    try:
        import rmm

        rmm.reinitialize(managed_memory=True)
    except Exception:
        pass
    import cugraph
    import cudf

    t_build_0 = time.time()
    gdf = cudf.DataFrame({
        "src": edges[:, 0],
        "dst": edges[:, 1],
    })
    G = cugraph.Graph(directed=True)
    # renumber=True maps vertex IDs to a contiguous [0, N-1] block. Without it a
    # sparse/high ID space makes cuGraph allocate an array sized to the max ID,
    # which OOMs immediately.
    G.from_cudf_edgelist(gdf, source="src", destination="dst", renumber=True)
    build_s = round(time.time() - t_build_0, 3)

    damping = float(getattr(args, "damping", 0.85))
    iterations = int(getattr(args, "iterations", 30))

    t0 = time.time()
    if args.algorithm == "pagerank":
        res = cugraph.pagerank(G, alpha=damping, max_iter=iterations)
        n_out = len(res)
    elif args.algorithm == "connected_components":
        res = cugraph.weakly_connected_components(G)
        n_out = len(res)
    elif args.algorithm == "shortest_path":
        target = getattr(args, "target", None)
        res = cugraph.bfs(G, start=int(args.source))
        if target is not None:
            res_target = res[res["vertex"] == int(target)]
            n_out = len(res_target)
        else:
            n_out = len(res)
    else:
        raise RuntimeError(f"unknown algorithm {args.algorithm}")
    return {
        "engine": "cugraph",
        "workload": "graph",
        "available": True,
        "algorithm": args.algorithm,
        "edges": len(edges),
        "build_seconds": build_s,
        "seconds": round(time.time() - t0, 3),
        "result_size": n_out,
    }


def run_graph(args) -> dict:
    """Dispatch a graph engine, tagging the record with the dataset label."""
    res = _graph_dispatch(args)
    if isinstance(res, dict):
        res.setdefault("dataset", getattr(args, "dataset", None) or "graph")
    return res


def _graph_dispatch(args) -> dict:
    if not args.graph_edges:
        return {
            "workload": "graph",
            "available": False,
            "error": "graph engines require --graph-edges",
        }
    edges = np.loadtxt(args.graph_edges, dtype=np.int64, ndmin=2)
    engine = args.engine
    if engine in ("benostreamdb", "bsdb"):
        return _graph_benostreamdb(args, edges)
    if engine == "cugraph":
        return _graph_cugraph(args, edges)
    if engine == "neo4j":
        if not importlib.util.find_spec("neo4j"):
            return {
                "engine": "neo4j",
                "workload": "graph",
                "available": False,
                "error": "pip install neo4j (and run Neo4j with the GDS plugin)",
            }
        return _graph_neo4j(args, edges)
    if engine == "networkx":
        if not importlib.util.find_spec("networkx"):
            return {"engine": "networkx", "workload": "graph", "available": False}
        return _graph_networkx(args, edges)
    return {
        "engine": engine,
        "workload": "graph",
        "available": False,
        "error": f"unknown graph engine '{engine}'",
    }


# --------------------------------------------------------------------------- #
# Lexical workload
# --------------------------------------------------------------------------- #
def _lexical_tantivy(args, corpus, queries, qrels) -> dict:
    import tantivy
    import tempfile

    schema_builder = tantivy.SchemaBuilder()
    schema_builder.add_text_field("id", stored=True)
    schema_builder.add_text_field("text", stored=True, tokenizer_name="en_stem")
    schema = schema_builder.build()

    t0 = time.time()
    with tempfile.TemporaryDirectory() as index_dir:
        index = tantivy.Index(schema, path=index_dir)
        writer = index.writer(heap_size=1024_000_000)

        for doc_id, text in corpus:
            writer.add_document(tantivy.Document(
                id=[doc_id],
                text=[text],
            ))
        writer.commit()
        build_s = time.time() - t0

        index.reload()
        searcher = index.searcher()

        hits = 0.0
        q = min(args.queries, len(queries))
        latencies = []

        import re
        for q_id, q_text in queries[:q]:
            # Sanitize to avoid tantivy syntax errors
            safe_q = re.sub(r'[^\w\s]', ' ', q_text)
            query = index.parse_query(safe_q, ["text"])

            t1 = time.perf_counter()
            results = searcher.search(query, args.k).hits
            latencies.append(time.perf_counter() - t1)

            got = {searcher.doc(doc_address)["id"][0] for score, doc_address in results}
            truth = qrels.get(q_id, set())
            if truth:
                hits += len(got & truth) / min(args.k, len(truth))

        recall = hits / q
        total = sum(latencies) or 1e-9

        return {
            "engine": "tantivy",
            "workload": "lexical",
            "available": True,
            "queries": q,
            "build_s": round(build_s, 3),
            "recall_at_k": round(recall, 4),
            "qps": round(q / total, 1),
            "p50_ms": round(float(np.percentile(latencies, 50) * 1000), 3),
            "p99_ms": round(float(np.percentile(latencies, 99) * 1000), 3),
        }

def _lexical_opensearch(args, corpus, queries, qrels) -> dict:
    from opensearchpy import OpenSearch, helpers
    host = args.host or os.environ.get("ES_URL") or "http://localhost:9200"
    es = OpenSearch(host)

    # 1. Build index
    es.indices.delete(index="bsdb-lexical", ignore_unavailable=True)
    es.indices.create(
        index="bsdb-lexical",
        body={
            "mappings": {
                "properties": {
                    "text": {"type": "text", "analyzer": "english"}
                }
            }
        },
    )
    
    t0 = time.time()
    actions = (
        {
            "_index": "bsdb-lexical",
            "_id": doc_id,
            "_source": {"text": text},
        }
        for doc_id, text in corpus
    )
    helpers.bulk(es, actions)
    es.indices.refresh(index="bsdb-lexical")
    build_s = time.time() - t0

    # 2. Search
    hits = 0.0
    q = min(args.queries, len(queries))
    latencies = []
    
    import re
    for q_id, q_text in queries[:q]:
        safe_q = re.sub(r'[^\w\s]', ' ', q_text)
        
        t1 = time.perf_counter()
        res = es.search(
            index="bsdb-lexical",
            body={
                "query": {"match": {"text": safe_q}},
                "size": args.k,
                "_source": False,
            }
        )
        latencies.append(time.perf_counter() - t1)
        
        got = {str(h["_id"]) for h in res["hits"]["hits"]}
        truth = qrels.get(q_id, set())
        if truth:
            hits += len(got & truth) / min(args.k, len(truth))

    recall = hits / q
    total = sum(latencies) or 1e-9

    return {
        "engine": args.engine,
        "workload": "lexical",
        "available": True,
        "queries": q,
        "build_s": round(build_s, 3),
        "recall_at_k": round(recall, 4),
        "qps": round(q / total, 1),
        "p50_ms": round(float(np.percentile(latencies, 50) * 1000), 3),
        "p99_ms": round(float(np.percentile(latencies, 99) * 1000), 3),
    }

def run_lexical(args) -> dict:
    if not args.beir_corpus:
        return {
            "workload": "lexical",
            "available": False,
            "error": "lexical engines require --beir-corpus",
        }

    engine = args.engine
    if engine in ("tantivy", "opensearch"):
        if engine == "tantivy" and not importlib.util.find_spec("tantivy"):
            return {
                "engine": "tantivy",
                "workload": "lexical",
                "available": False,
                "error": "pip install tantivy",
            }
        elif engine == "opensearch" and not importlib.util.find_spec("opensearchpy"):
            return {
                "engine": engine,
                "workload": "lexical",
                "available": False,
                "error": "pip install opensearch-py",
            }

        corpus = []
        with open(args.beir_corpus, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                corpus.append((d["_id"], d.get("title", "") + " " + d.get("text", "")))

        qrels = {}
        with open(args.beir_qrels, "r", encoding="utf-8") as f:
            next(f) # skip header
            for line in f:
                q_id, doc_id, score = line.strip().split('\t')
                if int(score) > 0:
                    qrels.setdefault(q_id, set()).add(doc_id)

        queries = []
        with open(args.beir_queries, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if str(d["_id"]) in qrels:
                    queries.append((str(d["_id"]), d["text"]))

        if engine == "tantivy":
            return _lexical_tantivy(args, corpus, queries, qrels)
        else:
            return _lexical_opensearch(args, corpus, queries, qrels)

    return {
        "engine": engine,
        "workload": "lexical",
        "available": False,
        "error": f"unsupported lexical engine {engine}"
    }

# --------------------------------------------------------------------------- #
# SQL workload
# --------------------------------------------------------------------------- #
def _sql_duckdb(args) -> dict:
    if not importlib.util.find_spec("duckdb"):
        return {
            "engine": "duckdb",
            "workload": "sql",
            "available": False,
            "error": "pip install duckdb",
        }
    import duckdb

    sql = args.sql
    if not sql:
        if not args.parquet:
            return {
                "engine": "duckdb",
                "workload": "sql",
                "available": False,
                "error": "provide --sql or --parquet",
            }
        sql = (
            "SELECT count(*) AS n, avg(WatchID) AS avg_id "
            f"FROM read_parquet('{args.parquet}')"
        )
    con = duckdb.connect()
    if args.parquet and "from t" in sql.lower() and "read_parquet" not in sql.lower():
        con.execute(f"CREATE VIEW t AS SELECT * FROM read_parquet('{args.parquet}')")
    t0 = time.time()
    result = con.execute(sql).fetchall()
    return {
        "engine": "duckdb",
        "workload": "sql",
        "available": True,
        "sql": sql,
        "seconds": round(time.time() - t0, 3),
        "rows": len(result),
    }


def _sql_datafusion(args) -> dict:
    if not importlib.util.find_spec("datafusion"):
        return {
            "engine": "datafusion",
            "workload": "sql",
            "available": False,
            "error": "pip install datafusion",
        }
    import datafusion

    ctx = datafusion.SessionContext()
    sql = args.sql
    if not sql:
        if not args.parquet:
            return {
                "engine": "datafusion",
                "workload": "sql",
                "available": False,
                "error": "provide --sql or --parquet",
            }
        ctx.register_parquet("t", args.parquet)
        sql = "SELECT count(*) AS n FROM t"
    elif args.parquet:
        ctx.register_parquet("t", args.parquet)

    t0 = time.time()
    df = ctx.sql(sql)
    batches = df.collect()
    total_rows = sum(b.num_rows for b in batches)
    return {
        "engine": "datafusion",
        "workload": "sql",
        "available": True,
        "sql": sql,
        "seconds": round(time.time() - t0, 3),
        "rows": total_rows,
    }


def _sql_clickhouse(args) -> dict:
    if not importlib.util.find_spec("clickhouse_connect"):
        return {
            "engine": "clickhouse",
            "workload": "sql",
            "available": False,
            "error": "pip install clickhouse_connect",
        }
    import clickhouse_connect

    host = getattr(args, "host", None) or os.environ.get("CLICKHOUSE_HOST", "localhost")
    port = int(getattr(args, "port", None) or os.environ.get("CLICKHOUSE_PORT", "8123"))
    user = getattr(args, "user", None) or os.environ.get("CLICKHOUSE_USER", "default")
    password = getattr(args, "password", None) or os.environ.get("CLICKHOUSE_PASSWORD", "")
    try:
        client = clickhouse_connect.get_client(host=host, port=port, username=user, password=password)
    except Exception as e:
        return {
            "engine": "clickhouse",
            "workload": "sql",
            "available": False,
            "error": f"connection error: {e}",
        }

    sql = args.sql
    if args.parquet:
        # Materialise the Parquet into a table named `t` so the shared SQL
        # (which references `t`) works across every SQL engine. The schema is
        # taken from the Parquet itself (ClickBench, TPC-H, NYC TLC, ... all
        # differ). Load via the client (not `file()`) so ClickHouse needs no
        # data volume — its entrypoint chowns any mounted dir.
        import pyarrow as pa
        import pyarrow.parquet as pq

        def _ch_type(t, nullable: bool) -> str:
            if pa.types.is_int64(t):
                base = "Int64"
            elif pa.types.is_int32(t):
                base = "Int32"
            elif pa.types.is_int16(t):
                base = "Int16"
            elif pa.types.is_float64(t):
                base = "Float64"
            elif pa.types.is_float32(t):
                base = "Float32"
            elif pa.types.is_boolean(t):
                base = "UInt8"
            elif pa.types.is_date32(t):
                base = "Date32"
            elif pa.types.is_date64(t):
                base = "Date"
            elif pa.types.is_timestamp(t):
                base = "DateTime64(6)"
            elif pa.types.is_decimal(t):
                base = f"Decimal({t.precision},{t.scale})"
            else:
                base = "String"
            # A nullable Parquet column (e.g. NYC TLC's `passenger_count`) must
            # map to a Nullable ClickHouse type, or the client cannot build the
            # native array for the NULL values.
            return f"Nullable({base})" if nullable else base

        tbl = pq.read_table(args.parquet)
        cols = tbl.schema.names
        col_defs = ", ".join(
            f"`{c}` {_ch_type(tbl.schema.field(c).type, tbl.schema.field(c).nullable)}"
            for c in cols
        )
        client.command("DROP TABLE IF EXISTS t")
        client.command(f"CREATE TABLE t ({col_defs}) ENGINE = MergeTree ORDER BY tuple()")
        client.insert(
            "t",
            list(zip(*[tbl.column(c).to_pylist() for c in cols])),
            column_names=cols,
        )
    if not sql:
        if not args.parquet:
            return {
                "engine": "clickhouse",
                "workload": "sql",
                "available": False,
                "error": "provide --sql or --parquet",
            }
        sql = "SELECT count(*) AS n FROM t"

    t0 = time.time()
    res = client.query(sql)
    return {
        "engine": "clickhouse",
        "workload": "sql",
        "available": True,
        "sql": sql,
        "seconds": round(time.time() - t0, 3),
        "rows": len(res.result_rows),
    }


def _sql_trino(args) -> dict:
    if not importlib.util.find_spec("trino"):
        return {
            "engine": "trino",
            "workload": "sql",
            "available": False,
            "error": "pip install trino",
        }
    from trino.dbapi import connect

    host = getattr(args, "host", None) or os.environ.get("TRINO_HOST", "localhost")
    port = int(getattr(args, "port", None) or os.environ.get("TRINO_PORT", "8080"))
    user = getattr(args, "user", None) or os.environ.get("TRINO_USER", "bench")
    catalog = getattr(args, "catalog", None) or os.environ.get("TRINO_CATALOG", "memory")
    schema = getattr(args, "schema", None) or os.environ.get("TRINO_SCHEMA", "default")
    # A Parquet dataset is registered as an external Hive table (the memory
    # connector cannot bulk-load large datasets), so query through `hive`.
    if args.parquet:
        catalog = "hive"

    try:
        conn = connect(host=host, port=port, user=user, catalog=catalog, schema=schema)
    except Exception as e:
        return {
            "engine": "trino",
            "workload": "sql",
            "available": False,
            "error": f"connection error: {e}",
        }

    cur = conn.cursor()
    ingest_s = None
    if args.parquet:
        # Register the shared Parquet as an external Hive table `t` (the memory
        # connector cannot bulk-load large datasets — it OOMs). The schema is
        # taken from the Parquet itself (ClickBench / TPC-H / NYC TLC differ).
        # The runner and Trino mount the same data dir at the same path, so the
        # `file://` location resolves inside Trino.
        import pyarrow as pa
        import pyarrow.parquet as pq

        def _trino_type(t) -> str:
            if pa.types.is_boolean(t):
                return "boolean"
            if pa.types.is_int64(t):
                return "bigint"
            if pa.types.is_int32(t):
                return "integer"
            if pa.types.is_int16(t):
                return "smallint"
            if pa.types.is_float64(t):
                return "double"
            if pa.types.is_float32(t):
                return "real"
            if pa.types.is_date(t):
                return "date"
            if pa.types.is_timestamp(t):
                return "timestamp"
            if pa.types.is_decimal(t):
                return f"decimal({t.precision},{t.scale})"
            return "varchar"

        t_ingest = time.time()
        tbl = pq.read_table(args.parquet)
        cols = tbl.schema.names
        col_defs = ", ".join(
            f'"{c}" {_trino_type(tbl.schema.field(c).type)}' for c in cols
        )
        location = "file://" + os.path.dirname(args.parquet)
        cur.execute("CREATE SCHEMA IF NOT EXISTS hive.default")
        cur.execute("DROP TABLE IF EXISTS hive.default.t")
        cur.execute(
            f"CREATE TABLE hive.default.t ({col_defs}) "
            f"WITH (format = 'PARQUET', external_location = '{location}')"
        )
        ingest_s = round(time.time() - t_ingest, 3)

    sql = args.sql
    if not sql:
        if not args.parquet:
            return {
                "engine": "trino",
                "workload": "sql",
                "available": False,
                "error": "provide --sql or --parquet",
            }
        sql = "SELECT count(*) AS n FROM t"

    t0 = time.time()
    cur.execute(sql)
    rows = cur.fetchall()
    return {
        "engine": "trino",
        "workload": "sql",
        "available": True,
        "sql": sql,
        "ingest_seconds": ingest_s,
        "seconds": round(time.time() - t0, 3),
        "rows": len(rows),
    }


def _sql_benostreamdb(args) -> dict:
    import benostreamdb as bsdb
    import pyarrow.parquet as pq
    import tempfile
    import shutil

    tmpdir = tempfile.mkdtemp(prefix="bsdb_sql_bench_")
    try:
        t_ingest_0 = time.time()
        table = bsdb.Table(tmpdir)
        if args.parquet:
            pa_table = pq.read_table(args.parquet)
            table.write(pa_table)
            table.commit()
            table.wait_for_background_tasks()
        ingest_s = round(time.time() - t_ingest_0, 3)

        sql = args.sql or "SELECT count(*) FROM t"
        t0 = time.time()
        res = table.sql(sql)
        elapsed = round(time.time() - t0, 3)
        return {
            "engine": "benostreamdb",
            "workload": "sql",
            "available": True,
            "sql": sql,
            "ingest_seconds": ingest_s,
            "seconds": elapsed,
            "rows": len(res),
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_sql(args) -> dict:
    engine = args.engine
    if engine == "duckdb":
        return _sql_duckdb(args)
    if engine == "datafusion":
        return _sql_datafusion(args)
    if engine == "clickhouse":
        return _sql_clickhouse(args)
    if engine == "trino":
        return _sql_trino(args)
    if engine in ("benostreamdb", "bsdb"):
        return _sql_benostreamdb(args)
    return {
        "engine": engine,
        "workload": "sql",
        "available": False,
        "error": f"unknown sql engine '{engine}'",
    }


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor()


def _total_ram_gb() -> Optional[float]:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page = os.sysconf("SC_PAGE_SIZE")
        return round(pages * page / 1024**3, 1)
    except Exception:
        return None


def _gpu_info() -> list[str]:
    """Visible NVIDIA GPUs (empty in a CPU-only run)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    except Exception:
        pass
    return []


def emit(record: dict, out: Optional[str]) -> None:
    """Attach the hardware profile so every result is self-describing.

    ``cores``/``ram_gb`` are the *cgroup-visible* limits when running in Docker
    (`os.cpu_count()` and `SC_PHYS_PAGES` reflect the container's limits), so
    the recorded envelope matches what the engine actually saw.
    """
    record.setdefault("env", {})
    record["env"].setdefault("cpu_model", _cpu_model())
    record["env"].setdefault("cores", os.cpu_count())
    record["env"].setdefault("ram_gb", _total_ram_gb())
    record["env"].setdefault("os", platform.platform())
    record["env"].setdefault("python", platform.python_version())
    record["env"].setdefault("containerized", os.path.exists("/.dockerenv"))
    record["env"].setdefault("gpus", _gpu_info())
    print(json.dumps(record, indent=2), flush=True)
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        with open(out.replace(".json", ".md"), "w", encoding="utf-8") as f:
            f.write(f"# Competitor: {record.get('engine')} / {record.get('workload')}\n\n")
            f.write("| Metric | Value |\n|---|---|\n")
            for key, val in record.items():
                if isinstance(val, dict):
                    val = ", ".join(f"{k}={v}" for k, v in val.items())
                f.write(f"| {key} | {val} |\n")
        print(f"wrote {out}", flush=True)


def fail(engine: str, exc: Exception) -> dict:
    return {
        "engine": engine,
        "available": False,
        "error": f"{type(exc).__name__}: {exc}",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--dataset", default="sift-128-euclidean")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--queries", type=int, default=500)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--ef-construction", type=int, default=200)
    ap.add_argument("--ef-search", type=int, default=200)
    ap.add_argument("--metric", default=None)
    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--cores", type=int, default=0)
    ap.add_argument("--ram-gb", type=float, default=0.0)
    # Server / path connection options.
    ap.add_argument("--dsn", default=None, help="pgvector connection string")
    ap.add_argument("--host", default=None, help="OpenSearch URL")
    ap.add_argument("--path", default=None, help="LanceDB directory")
    ap.add_argument("--uri", default=None, help="Neo4j bolt URI")
    ap.add_argument("--user", default=None)
    ap.add_argument("--password", default=None)
    # Graph / SQL / Lexical options.
    ap.add_argument("--workload", default=None, choices=["vector", "graph", "sql", "lexical"])
    ap.add_argument("--graph-edges", default=None)
    ap.add_argument("--beir-corpus", default=None)
    ap.add_argument("--beir-queries", default=None)
    ap.add_argument("--beir-qrels", default=None)
    ap.add_argument("--algorithm", default="pagerank")
    ap.add_argument("--source", type=int, default=0)
    ap.add_argument("--target", type=int, default=None)
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--damping", type=float, default=0.85)
    ap.add_argument("--iterations", type=int, default=30)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--sql", default=None)
    ap.add_argument("--parquet", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    apply_envelope(args.cores, args.ram_gb, args.engine, args.device)

    try:
        if args.workload == "sql" or args.engine in ("duckdb", "datafusion", "clickhouse", "trino") or (args.engine in ("benostreamdb", "bsdb") and (args.sql or (args.parquet and not args.dataset))):
            emit(run_sql(args), args.out)
            return
        if args.workload == "graph" or args.engine in ("networkx", "neo4j", "cugraph") or (args.engine in ("benostreamdb", "bsdb") and args.graph_edges):
            emit(run_graph(args), args.out)
            return
        if args.workload == "lexical" or args.engine == "tantivy" or (args.engine == "opensearch" and args.beir_corpus):
            emit(run_lexical(args), args.out)
            return

        adapters = vector_adapters()
        adapter = adapters.get(args.engine)
        if adapter is None:
            raise SystemExit(f"unknown engine '{args.engine}' (have: {sorted(adapters)})")
        if not adapter.available():
            print(
                json.dumps(
                    {
                        "engine": args.engine,
                        "available": False,
                        "error": f"'{adapter.module}' is not installed — "
                        f"`pip install {adapter.module}`",
                    },
                    indent=2,
                )
            )
            return
        emit(run_vector(adapter, args), args.out)
    except Exception as exc:  # server/setup failures: report, do not crash
        emit(fail(args.engine, exc), args.out)


if __name__ == "__main__":
    main()
