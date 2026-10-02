#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Competitor runner for the BenoStreamDB benchmark plan.

Runs a competitor under the same resource envelope and parameter set as the
BenoStreamDB harness (`benchmarks/ann_benchmarks/run.py`) and emits the same
metrics, so results are directly comparable.

Workload families:

* ``vector`` — ANN over an ANN-Benchmarks HDF5 dataset.
  Engines: ``faiss``, ``hnswlib`` (embedded), ``pgvector``, ``lancedb``,
  ``elasticsearch`` (server-backed).
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

    python benchmarks/competitors/run_competitor.py --engine elasticsearch \\
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
def apply_envelope(cores: int, ram_gb: float) -> None:
    """Best-effort process limits so the competitor matches BenoStreamDB.

    Threads: set the common BLAS/OMP env vars (must be set before numpy/faiss
    import to take effect; the runner sets them as early as possible). RAM: use
    ``RLIMIT_AS`` when ``ram_gb`` is given, so an over-budget run fails loudly
    instead of silently using more memory than BenoStreamDB was given.
    """
    if cores > 0:
        for var in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ[var] = str(cores)
    if ram_gb > 0:
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
        cur.execute(
            f"CREATE INDEX ON bsdb_bench USING hnsw (embedding {opclass}) "
            f"WITH (m = %s, ef_construction = %s)",
            (int(params["m"]), int(params["ef_construction"])),
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
    def __init__(self) -> None:
        super().__init__(name="lancedb", module="lancedb")

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
        dim = train.shape[1]
        # IVF_PQ partitions/sub-vectors scale with the data; keep them sane.
        t.create_index(
            metric=metric,
            num_partitions=min(256, max(1, len(train) // 256)),
            num_sub_vectors=min(16, max(1, dim // 8)),
        )
        return (t, uri)

    def search(self, handle, queries, k, ef_search, metric):
        t, _uri = handle
        out = []
        for q in queries:
            rows = t.search(q.tolist()).limit(int(k)).to_list()
            out.append([int(r["id"]) for r in rows])
        return np.array(out)

    def index_bytes(self, handle):
        _t, uri = handle
        return _dir_size(uri)


class ElasticsearchAdapter(VectorAdapter):
    """Elasticsearch or OpenSearch (dense_vector + HNSW kNN)."""

    def __init__(self) -> None:
        super().__init__(name="elasticsearch", module="elasticsearch", server_side=True)

    def build(self, train, params, ctx):
        from elasticsearch import Elasticsearch, helpers

        host = ctx.get("host") or os.environ.get("ES_URL") or "http://localhost:9200"
        es = Elasticsearch(host)
        dim = train.shape[1]
        metric = params.get("metric", "l2")
        similarity = {"l2": "l2_norm", "cosine": "cosine", "inner_product": "dot_product"}.get(
            metric, "l2_norm"
        )
        es.indices.delete(index="bsdb-bench", ignore_unavailable=True)
        es.indices.create(
            index="bsdb-bench",
            mappings={
                "properties": {
                    "vector": {
                        "type": "dense_vector",
                        "dims": dim,
                        "index": True,
                        "similarity": similarity,
                        "index_options": {
                            "type": "hnsw",
                            "m": int(params["m"]),
                            "ef_construction": int(params["ef_construction"]),
                        },
                    }
                }
            },
        )
        actions = (
            {
                "_index": "bsdb-bench",
                "_id": int(i),
                "_source": {"vector": row.tolist()},
            }
            for i, row in enumerate(train)
        )
        helpers.bulk(es, actions)
        es.indices.refresh(index="bsdb-bench")
        return es

    def search(self, handle, queries, k, ef_search, metric):
        es = handle
        out = []
        for q in queries:
            body = {
                "knn": {
                    "field": "vector",
                    "query_vector": q.tolist(),
                    "k": int(k),
                    "num_candidates": max(int(k) * 10, int(ef_search)),
                },
                "size": int(k),
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
        "lancedb": LanceDbAdapter(),
        "elasticsearch": ElasticsearchAdapter(),
        "opensearch": ElasticsearchAdapter(),
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

    params = {"m": args.m, "ef_construction": args.ef_construction, "metric": metric}
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

    t0 = time.time()
    with driver.session() as session:
        session.run(
            "MATCH (n) DETACH DELETE n"
        )
        session.run(
            "UNWIND $rows AS r MERGE (a:Node {id: r[0]}) "
            "MERGE (b:Node {id: r[1]}) MERGE (a)-[:E]->(b)",
            rows=[(int(a), int(b)) for a, b in edges],
        )
        session.run("CALL gds.graph.drop('bench', false)").consume()
        session.run(
            "CALL gds.graph.project('bench', 'Node', 'E')"
        ).consume()
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
        "seconds": round(time.time() - t0, 3),
        "result_size": len(rows),
    }


def run_graph(args) -> dict:
    if not args.graph_edges:
        return {
            "workload": "graph",
            "available": False,
            "error": "graph engines require --graph-edges",
        }
    edges = np.loadtxt(args.graph_edges, dtype=np.int64, ndmin=2)
    engine = args.engine
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
# SQL workload
# --------------------------------------------------------------------------- #
def run_sql(args) -> dict:
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
    ap.add_argument("--host", default=None, help="Elasticsearch/OpenSearch URL")
    ap.add_argument("--path", default=None, help="LanceDB directory")
    ap.add_argument("--uri", default=None, help="Neo4j bolt URI")
    ap.add_argument("--user", default=None)
    ap.add_argument("--password", default=None)
    # Graph / SQL options.
    ap.add_argument("--graph-edges", default=None)
    ap.add_argument("--algorithm", default="pagerank")
    ap.add_argument("--source", type=int, default=0)
    ap.add_argument("--sql", default=None)
    ap.add_argument("--parquet", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    apply_envelope(args.cores, args.ram_gb)

    try:
        if args.engine == "duckdb":
            emit(run_sql(args), args.out)
            return
        if args.engine in ("networkx", "neo4j"):
            emit(run_graph(args), args.out)
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
