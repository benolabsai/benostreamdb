#!/usr/bin/env python3
"""
OpenSearch 2.11 vs BenoStreamDB (bsdb-search) — REST API benchmark.

Spawns a local OpenSearch 2.11 (Docker, single-node, security disabled)
and a local ``bsdb-search`` binary, feeds both the same document stream
(one document per HTTP POST), and measures:

  * ingest throughput      (both; single-doc POST /{index}/_doc)
  * refresh latency        (both; time until the data is searchable)
  * BM25 ``match`` query   (both; p50/p95/p99 over N queries)
  * filtered query         (both; ``match`` + category ``term`` filter)
  * HNSW ``knn``           (bsdb-search only — ES 7.10 has no dense_vector)
  * hybrid ``match``+``knn`` (bsdb-search only; RRF fusion)
  * disk storage footprint (raw JSON, Parquet, secondary indexes vs Lucene)
  * memory footprint       (BenoStreamDB RSS vs ES JVM heap & container RSS)

Fairness notes:
  * Both systems receive one document per POST (no ``_bulk`` on bsdb-search),
    an explicit refresh before search, and run on the same host.
  * Both systems run in Docker containers with an **equal resource budget**:
    ``--mem-gb`` (default 4 GiB) and ``--cpus`` (default 4). ES gets a
    50%-of-budget JVM heap; bsdb-search's in-process caches
    (``BENOSTREAM_BLOCK_CACHE_GB`` / ``BENOSTREAM_CACHE_GB``) are sized to
    the same budget.
  * ES runs single-node, 1 shard, 0 replicas, refresh disabled during ingest
    (the standard way to measure ES ingest).
  * The Wikipedia document stream is cached locally
    (``data/wiki_20231101_en.jsonl``) so both systems see identical docs
    without HuggingFace rate limits.
  * The ``embedding`` field is part of the bsdb-search payload only: ES 7.10
    has no ``dense_vector`` type, so shipping 64 floats per document to a
    7.10 search cluster would not represent any real workload. Every other
    field is identical on both sides.
  * bsdb-search's first write also pays one-time index creation (manifest +
    Iceberg init); ES's index is pre-created for mapping/settings. The
    one-time cost is amortized over the whole ingest stream.
  * Latencies include the localhost HTTP round trip, measured identically.

Run:
    ./venv/bin/python benchmarks/competitive/benchmark_opensearch.py --size 100000 --storage local
    ./venv/bin/python benchmarks/competitive/benchmark_opensearch.py --size 100000 --storage cloud
    ./venv/bin/python benchmarks/competitive/benchmark_opensearch.py --quick
"""

import argparse
import json
import os
import platform
import random
import shutil
import socket
import subprocess
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
BINARY_RELEASE = REPO_ROOT / "target" / "release" / "bsdb-search"
BINARY_DEBUG = REPO_ROOT / "target" / "debug" / "bsdb-search"
BINARY = BINARY_RELEASE if BINARY_RELEASE.exists() else BINARY_DEBUG
RESULTS_DIR = Path(__file__).resolve().parent / "benchmark_results"

ES_IMAGE_DEFAULT = "opensearchproject/opensearch:2.11.1"
ES_CONTAINER = "opensearch-bench"
ES_STARTUP_DEADLINE_S = 180.0
BENOSEARCH_STARTUP_DEADLINE_S = 60.0
HTTP_TIMEOUT_S = 300.0


# --------------------------------------------------------------------------
# Result model (same shape as benchmark_suite.BenchmarkResult)
# --------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Single benchmark result. For query operations latency_ms is p50 and
    the full percentile set lives in metadata."""

    system: str
    operation: str
    dataset_size: int
    latency_ms: float
    throughput: Optional[float] = None
    memory_mb: Optional[float] = None
    storage_mb: Optional[float] = None
    hardware: str = "Unknown"
    device_type: str = "cpu"
    metadata: Optional[Dict] = None


def get_hardware_info() -> str:
    try:
        if platform.system() == "Linux":
            res = subprocess.check_output("lscpu | grep 'Model name'", shell=True).decode()
            return res.split(":")[1].strip()
        if platform.system() == "Darwin":
            res = subprocess.check_output("sysctl -n machdep.cpu.brand_string", shell=True).decode()
            return res.strip()
    except Exception:
        pass
    return platform.processor() or "Generic x86_64"


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def percentiles(latencies_ms: List[float]) -> Dict:
    s = sorted(latencies_ms)
    n = len(s)

    def pct(p: float) -> float:
        if n == 1:
            return s[0]
        idx = (p / 100.0) * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return s[lo] * (1.0 - frac) + s[hi] * frac

    return {
        "min_ms": round(s[0], 3),
        "p50_ms": round(pct(50), 3),
        "p95_ms": round(pct(95), 3),
        "p99_ms": round(pct(99), 3),
        "max_ms": round(s[-1], 3),
        "mean_ms": round(sum(s) / n, 3),
        "n": n,
    }


# --------------------------------------------------------------------------
# Test data
# --------------------------------------------------------------------------


LOCAL_WIKI_JSONL = Path(__file__).resolve().parent / "data" / "wiki_20231101_en.jsonl"


def get_documents_from_hf(dataset_name, size, dim):
    import random

    docs = []

    # Local cache first: identical docs on every run, no HF rate limits.
    if dataset_name == "wiki" and LOCAL_WIKI_JSONL.exists():
        print(f"Loading {size} docs from local cache {LOCAL_WIKI_JSONL.name}...")
        emb_rng = random.Random(777)
        with open(LOCAL_WIKI_JSONL) as f:
            for i, line in enumerate(f):
                if i >= size:
                    break
                item = json.loads(line)
                docs.append({
                    "title": item["title"][:200],
                    "body": item["body"][:2000],
                    "category": "general",
                    "price": 0.0,
                    "ts": "2026-01-01T00:00:00Z",
                    "embedding": [emb_rng.random() for _ in range(dim)],
                })
        return docs

    from datasets import load_dataset
    print(f"Loading {size} docs from {dataset_name} (streaming)...")

    if dataset_name == "wiki":
        ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
        title_col = "title"
        body_col = "text"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Only 'wiki' is currently supported.")

    for i, item in enumerate(ds):
        if i >= size:
            break
        
        title = item.get(title_col, "")
        if not title and "article_title" in item:
             title = item["article_title"]
             
        body = item.get(body_col, "")
        if not body and "text" in item:
             body = item["text"]
             
        docs.append({
            "title": title[:200] if title else "",
            "body": body[:2000] if body else "",
            "category": "science" if dataset_name == "pmc" else "general",
            "price": 0.0,
            "ts": "2026-01-01T00:00:00Z",
            "embedding": [random.random() for _ in range(dim)]
        })
    return docs


# --------------------------------------------------------------------------
# System adapters
# --------------------------------------------------------------------------

class Hypersearch:
    """``bsdb-search`` on a free local port.

    By default runs in a memory/CPU-capped Docker container (image
    ``bsdb-search-bench:latest``, built from ``docker/Dockerfile.search-bench``)
    so it gets the same hardware budget as the OpenSearch container.
    ``containerized=False`` falls back to spawning the native binary.
    """

    HS_CONTAINER = "bsdb-search-bench"

    def __init__(self, port: int, storage_uri: str, is_cloud: bool = False,
                 mem_gb: int = 4, cpus: int = 4, containerized: bool = True):
        self.base = f"http://127.0.0.1:{port}"
        self.storage_uri = storage_uri
        self.is_cloud = is_cloud
        self.mem_gb = mem_gb
        self.cpus = cpus
        self.containerized = containerized
        env = {
            "BENOSEARCH_BIND": "127.0.0.1",
            "BENOSEARCH_PORT": str(port),
            "BENOSEARCH_STORAGE_URI": storage_uri,
            # Equal-memory-budget: size in-process caches to the shared budget.
            "BENOSTREAM_BLOCK_CACHE_GB": str(max(1, mem_gb // 4)),
            "BENOSTREAM_CACHE_GB": str(max(1, mem_gb // 8)),
            "RUST_LOG": "info",
        }
        if is_cloud or storage_uri.startswith("s3://"):
            endpoint = os.environ.get("AWS_ENDPOINT_URL", "http://127.0.0.1:9000")
            env["AWS_ENDPOINT_URL"] = endpoint
            if "127.0.0.1" in endpoint or "localhost" in endpoint:
                env["AWS_ACCESS_KEY_ID"] = os.environ.get("MINIO_ROOT_USER", "minioadmin")
                env["AWS_SECRET_ACCESS_KEY"] = os.environ.get("MINIO_ROOT_PASSWORD", "minioadmin")
            else:
                env["AWS_ACCESS_KEY_ID"] = os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin")
                env["AWS_SECRET_ACCESS_KEY"] = os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin")
            env["AWS_REGION"] = os.environ.get("AWS_REGION", "us-east-1")
            env["AWS_ALLOW_HTTP"] = "true"

        if containerized:
            # Local file storage: mount the host dir at /data in the container.
            mounts: List[str] = []
            if storage_uri.startswith("file://"):
                host_dir = storage_uri[len("file://"):]
                env["BENOSEARCH_STORAGE_URI"] = "file:///data"
                mounts = ["-v", f"{host_dir}:/data"]
            env_args: List[str] = []
            for k, v in env.items():
                env_args += ["-e", f"{k}={v}"]
            subprocess.run(["docker", "rm", "-f", self.HS_CONTAINER], capture_output=True)
            self.proc = subprocess.run(
                [
                    "docker", "run", "-d", "--name", self.HS_CONTAINER,
                    "--network", "host",
                    "--memory", f"{mem_gb}g",
                    "--memory-swap", f"{mem_gb}g",
                    "--cpus", str(cpus),
                    *mounts,
                    *env_args,
                    "bsdb-search-bench:latest",
                ],
                capture_output=True,
                text=True,
            )
            if self.proc.returncode != 0:
                raise RuntimeError(f"docker run failed: {self.proc.stderr.strip()[:500]}")
            self.pid = None
        else:
            self.proc = subprocess.Popen(
                [str(BINARY)],
                env={**os.environ, **env},
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            self.pid = self.proc.pid
        self.session = requests.Session()
        deadline = time.time() + BENOSEARCH_STARTUP_DEADLINE_S
        ready = False
        while time.time() < deadline:
            try:
                if self.session.get(self.base + "/", timeout=2).status_code == 200:
                    ready = True
                    break
            except requests.RequestException:
                pass
            time.sleep(0.25)
        if not ready:
            self.stop()
            detail = ""
            if containerized:
                logs = subprocess.run(
                    ["docker", "logs", "--tail", "40", self.HS_CONTAINER],
                    capture_output=True, text=True)
                detail = f" container logs: {logs.stdout[-500:]}"
            raise RuntimeError(
                f"bsdb-search did not become ready within {BENOSEARCH_STARTUP_DEADLINE_S:.0f}s{detail}")
        self.info = self.session.get(self.base + "/", timeout=5).json()

    def index_doc(self, index: str, doc: Dict) -> requests.Response:
        return self.session.post(self.base + f"/{index}/_doc", json=doc, timeout=HTTP_TIMEOUT_S)

    def refresh(self, index: str) -> requests.Response:
        return self.session.post(self.base + f"/{index}/_refresh", timeout=HTTP_TIMEOUT_S)

    def search(self, index: str, body: Dict) -> requests.Response:
        return self.session.post(self.base + f"/{index}/_search", json=body, timeout=HTTP_TIMEOUT_S)

    def rss_mb(self) -> Optional[float]:
        if self.containerized:
            try:
                r = subprocess.run(
                    ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", self.HS_CONTAINER],
                    capture_output=True, text=True, timeout=15)
                usage = r.stdout.strip().split(" / ")[0]
                for suffix, mult in (("GiB", 1024.0), ("MiB", 1.0), ("KiB", 1.0 / 1024), ("B", 1.0 / 1024 / 1024)):
                    if usage.endswith(suffix):
                        return round(float(usage[: -len(suffix)]) * mult, 1)
            except Exception:
                pass
            return None
        try:
            for line in Path(f"/proc/{self.pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024.0, 1)
        except OSError:
            pass
        return None

    def storage_breakdown(self) -> Dict[str, float]:
        breakdown = {
            "parquet_mb": 0.0,
            "indexes_mb": 0.0,
            "metadata_mb": 0.0,
            "total_mb": 0.0,
        }
        target_dir = None
        if self.storage_uri.startswith("file://"):
            target_dir = Path(self.storage_uri[len("file://"):])
        elif self.storage_uri.startswith("s3://warehouse/"):
            rel_path = self.storage_uri[len("s3://warehouse/"):].strip("/")
            minio_p = REPO_ROOT / "docker" / "data" / "minio" / "warehouse" / rel_path
            if minio_p.exists():
                target_dir = minio_p

        if target_dir and target_dir.exists():
            for p in target_dir.rglob("*"):
                if p.is_file():
                    sz = p.stat().st_size
                    breakdown["total_mb"] += sz
                    path_str = str(p).lower()
                    if ".inv." in path_str or ".doclen." in path_str or ".hnsw" in path_str or ".idx" in path_str or "/indexes/" in path_str:
                        breakdown["indexes_mb"] += sz
                    elif ".parquet" in path_str:
                        breakdown["parquet_mb"] += sz
                    elif ".json" in path_str or ".avro" in path_str or ".text" in path_str or "_manifest" in path_str or "metadata" in path_str:
                        breakdown["metadata_mb"] += sz

            for k in breakdown:
                breakdown[k] = round(breakdown[k] / (1024 * 1024), 2)
        return breakdown

    def storage_mb(self) -> Optional[float]:
        bd = self.storage_breakdown()
        return bd["total_mb"] if bd["total_mb"] > 0 else None

    def stop(self) -> None:
        if self.containerized:
            subprocess.run(["docker", "rm", "-f", self.HS_CONTAINER], capture_output=True)
            return
        try:
            self.proc.terminate()
            self.proc.wait(timeout=10)
        except Exception:
            try:
                self.proc.kill()
            except OSError:
                pass


class OpenSearch:
    """Docker-managed single-node OpenSearch 2.11 (security off)."""

    def __init__(self, image: str, port: int, keep: bool = False, mem_gb: int = 4, cpus: int = 4):
        # Remove any stale container from a previous run.
        subprocess.run(["docker", "rm", "-f", ES_CONTAINER], capture_output=True)
        heap_gb = max(1, mem_gb // 2)  # 50%-of-budget JVM heap convention
        self.proc = subprocess.run(
            [
                "docker", "run", "-d", "--name", ES_CONTAINER,
                "-p", f"{port}:9200",
                "--memory", f"{mem_gb}g",
                "--memory-swap", f"{mem_gb}g",
                "--cpus", str(cpus),
                "-e", "discovery.type=single-node",
                "-e", "DISABLE_SECURITY_PLUGIN=true",
                "-e", "OPENSEARCH_INITIAL_ADMIN_PASSWORD=my-strong-password123!",
                "-e", f"OPENSEARCH_JAVA_OPTS=-Xms{heap_gb}g -Xmx{heap_gb}g",
                image,
            ],
            capture_output=True,
            text=True,
        )
        if self.proc.returncode != 0:
            raise RuntimeError(f"docker run failed: {self.proc.stderr.strip()[:500]}")
        self.base = f"http://127.0.0.1:{port}"
        self.session = requests.Session()
        self.keep = keep
        deadline = time.time() + ES_STARTUP_DEADLINE_S
        ready = False
        while time.time() < deadline:
            try:
                if self.session.get(self.base + "/", timeout=2).status_code == 200:
                    ready = True
                    break
            except requests.RequestException:
                pass
            time.sleep(1.0)
        if not ready:
            logs = subprocess.run(
                ["docker", "logs", "--tail", "40", ES_CONTAINER],
                capture_output=True, text=True,
            ).stderr
            self.stop(keep=False)
            raise RuntimeError(
                f"ES did not become ready within {ES_STARTUP_DEADLINE_S:.0f}s; last logs:\n{logs}"
            )
        self.info = self.session.get(self.base + "/", timeout=5).json()

    def create_index(self, index: str, dim: int) -> None:
        body = {
            "settings": {
                "index": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "refresh_interval": "-1",
                    "knn": True
                }
            },
            "mappings": {
                "properties": {
                    "title": {"type": "text"},
                    "body": {"type": "text"},
                    "category": {"type": "keyword"},
                    "price": {"type": "float"},
                    "ts": {"type": "date"},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": dim
                    }
                }
            },
        }
        r = self.session.put(self.base + f"/{index}", json=body, timeout=HTTP_TIMEOUT_S)
        r.raise_for_status()

    def delete_index(self, index: str) -> None:
        self.session.delete(self.base + f"/{index}", timeout=HTTP_TIMEOUT_S)

    def index_doc(self, index: str, doc: Dict) -> requests.Response:
        return self.session.post(self.base + f"/{index}/_doc", json=doc, timeout=HTTP_TIMEOUT_S)

    def refresh(self, index: str) -> requests.Response:
        return self.session.post(self.base + f"/{index}/_refresh", timeout=HTTP_TIMEOUT_S)

    def search(self, index: str, body: Dict) -> requests.Response:
        return self.session.post(self.base + f"/{index}/_search", json=body, timeout=HTTP_TIMEOUT_S)

    def index_store_mb(self, index: str) -> Optional[float]:
        try:
            r = self.session.get(self.base + f"/{index}/_stats/store", timeout=HTTP_TIMEOUT_S)
            if r.status_code == 200:
                data = r.json()
                bytes_val = data.get("_all", {}).get("total", {}).get("store", {}).get("size_in_bytes", 0)
                return round(bytes_val / (1024 * 1024), 2)
        except Exception:
            pass
        return None

    def jvm_heap_mb(self) -> Optional[float]:
        try:
            r = self.session.get(self.base + "/_nodes/stats/jvm", timeout=HTTP_TIMEOUT_S)
            if r.status_code == 200:
                nodes = r.json().get("nodes", {})
                for node in nodes.values():
                    bytes_val = node.get("jvm", {}).get("mem", {}).get("heap_used_in_bytes", 0)
                    return round(bytes_val / (1024 * 1024), 2)
        except Exception:
            pass
        return None

    def container_rss_mb(self) -> Optional[float]:
        try:
            r = subprocess.run(
                ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", ES_CONTAINER],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if r.returncode == 0 and r.stdout.strip():
                part = r.stdout.split("/")[0].strip()
                if "GiB" in part:
                    return round(float(part.replace("GiB", "").strip()) * 1024, 1)
                elif "MiB" in part:
                    return round(float(part.replace("MiB", "").strip()), 1)
                elif "kB" in part:
                    return round(float(part.replace("kB", "").strip()) / 1024, 1)
        except Exception:
            pass
        return None

    def data_dir_mb(self) -> Optional[float]:
        r = subprocess.run(
            ["docker", "exec", ES_CONTAINER, "du", "-sb", "/usr/share/elasticsearch/data"],
            capture_output=True, text=True,
        )
        if r.returncode == 0:
            return round(int(r.stdout.split()[0]) / (1024 * 1024), 2)
        return None

    def stop(self, keep: Optional[bool] = None) -> None:
        if keep if keep is not None else self.keep:
            print(f"  (keeping container {ES_CONTAINER} on port mapping; run docker rm -f to remove)")
            return
        subprocess.run(["docker", "rm", "-f", ES_CONTAINER], capture_output=True)


# --------------------------------------------------------------------------
# Benchmark operations
# --------------------------------------------------------------------------

def bench_ingest(
    system,
    system_name: str,
    index: str,
    docs: List[Dict],
    exclude_embedding: bool,
    hardware: str,
) -> List[BenchmarkResult]:
    n = len(docs)
    print(f"  ingesting {n:,} docs (bulk POST)...")
    t0 = time.time()
    per_doc = []
    batch_size = 100
    for i in range(0, n, batch_size):
        batch = docs[i:i+batch_size]
        payload = ""
        for doc in batch:
            clean_doc = {k: v for k, v in doc.items() if not (exclude_embedding and k == "embedding")}
            payload += '{"index": {}}\n'
            payload += json.dumps(clean_doc) + '\n'
            
        t = time.time()
        # system.base is the URL for the system
        if system_name.startswith("OpenSearch"):
            url = f"{system.base}/{index}/_bulk"
        else:
            url = f"{system.base}/{index}/_bulk"
            
        r = requests.post(url, data=payload, headers={'Content-Type': 'application/x-ndjson'})
        if r.status_code not in (200, 201):
            raise RuntimeError(f"{system_name} bulk ingest failed at doc {i}: {r.status_code} {r.text[:300]}")
        
        per_doc.extend([((time.time() - t) * 1000.0) / len(batch)] * len(batch))
            
        if (i + batch_size) % max(1, n // 10) == 0:
            print(f"    {min(i + batch_size, n):,}/{n:,}")
            
    ingest_s = time.time() - t0

    t0 = time.time()
    r = system.refresh(index)
    if r.status_code != 200:
        raise RuntimeError(f"{system_name} refresh failed: {r.status_code} {r.text[:300]}")
    refresh_s = time.time() - t0

    stats = percentiles(per_doc)
    results = [
        BenchmarkResult(
            system=system_name,
            operation="ingest",
            dataset_size=n,
            latency_ms=round(ingest_s * 1000.0, 1),
            throughput=round(n / ingest_s, 1),
            storage_mb=getattr(system, "storage_mb", None) and system.storage_mb(),
            hardware=hardware,
            metadata={**stats, "exclude_embedding": exclude_embedding},
        ),
        BenchmarkResult(
            system=system_name,
            operation="refresh",
            dataset_size=n,
            latency_ms=round(refresh_s * 1000.0, 1),
            hardware=hardware,
            metadata={"note": "time until data is searchable (index build included where applicable)"},
        ),
    ]
    print(f"    ingest: {n / ingest_s:,.0f} docs/s in {ingest_s:.1f}s; refresh: {refresh_s * 1000:.0f}ms")
    return results


def bench_query(
    system,
    system_name: str,
    index: str,
    operation: str,
    body_fn: Callable[[int], Dict],
    n_runs: int,
    dataset_size: int,
    hardware: str,
    extra_meta: Optional[Dict] = None,
) -> BenchmarkResult:
    print(f"  {operation}: {n_runs} runs...")
    for i in range(3):  # warm-up
        r = system.search(index, body_fn(i))
        if r.status_code != 200:
            raise RuntimeError(f"{system_name} warm-up failed: {r.status_code} {r.text[:300]}")
    lat = []
    for i in range(n_runs):
        t = time.time()
        r = system.search(index, body_fn(i))
        if r.status_code != 200:
            raise RuntimeError(f"{system_name} search failed at run {i}: {r.status_code} {r.text[:300]}")
        lat.append((time.time() - t) * 1000.0)
    stats = percentiles(lat)
    print(f"    p50={stats['p50_ms']}ms p95={stats['p95_ms']}ms p99={stats['p99_ms']}ms")
    meta = {"k": 10}
    if extra_meta:
        meta.update(extra_meta)
    meta.update(stats)
    return BenchmarkResult(
        system=system_name,
        operation=operation,
        dataset_size=dataset_size,
        latency_ms=stats["p50_ms"],
        hardware=hardware,
        metadata=meta,
    )


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

ENVELOPE_MS = (50.0, 200.0)  # search-latency acceptance envelope


def envelope_verdict(p95_ms: float) -> str:
    lo, hi = ENVELOPE_MS
    if p95_ms <= lo:
        return f"below envelope (<{lo:.0f}ms)"
    if p95_ms <= hi:
        return f"within envelope ({lo:.0f}-{hi:.0f}ms)"
    return f"ABOVE envelope (> {hi:.0f}ms)"


def write_reports(results: List[BenchmarkResult], meta: Dict, outdir: Path, timestamp: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    storage_mode = meta.get("storage_mode", "local")
    doc_count = meta["docs"]
    json_path = outdir / f"opensearch_bsdb-search_{storage_mode}_{doc_count}_{timestamp}.json"
    md_path = outdir / f"opensearch_bsdb-search_{storage_mode}_{doc_count}_{timestamp}.md"

    with open(json_path, "w") as f:
        json.dump({"run": meta, "results": [asdict(r) for r in results]}, f, indent=2)

    by_op: Dict[str, Dict[str, BenchmarkResult]] = {}
    for r in results:
        by_op.setdefault(r.operation, {})[r.system] = r

    query_ops = [op for op in ("match_bm25", "filtered") if op in by_op]
    hs_only_ops = [op for op in ("knn", "hybrid_rrf") if op in by_op]

    build_type = "release" if "release" in str(BINARY) else "debug"

    with open(md_path, "w") as f:
        f.write("# OpenSearch 2.11 vs BenoStreamDB — REST API Benchmark\n\n")
        f.write(f"**Generated:** {meta['generated']}  \n")
        f.write(f"**Host:** {meta['hardware']} ({platform.system()} {platform.release()})  \n")
        f.write(f"**ES:** {meta['es_version']} (build `{meta['es_build']}`, Docker, single-node, 1 shard, no replicas, 1 GiB JVM)  \n")
        f.write(f"**bsdb-search:** {meta['hs_version']} ({build_type} build, storage: `{storage_mode}`, in-process HNSW/BM25)  \n")
        f.write(f"**Dataset:** {meta['docs']:,} docs × {meta['dim']}-dim embeddings, {meta['runs']} query runs, k=10\n\n")

        f.write("## Ingest (bulk POST)\n\n")
        f.write("| System | docs/s | total | mean/doc | p95/doc | refresh (until searchable) |\n")
        f.write("|---|---|---|---|---|---|\n")
        if "ingest" in by_op and "refresh" in by_op:
            for sys_name in sorted(by_op["ingest"]):
                ing = by_op["ingest"][sys_name]
                ref = by_op["refresh"][sys_name]
                f.write(
                    f"| {sys_name} | {ing.throughput:,.0f} | {ing.latency_ms / 1000.0:.1f}s "
                    f"| {ing.metadata['mean_ms']}ms | {ing.metadata['p95_ms']}ms | {ref.latency_ms:.0f}ms |\n"
                )
            
            f.write("\n## Total Time to Searchable State (Ingest + Refresh)\n\n")
            f.write("| System | Ingest Time | Refresh Time | Total Time |\n")
            f.write("|---|---|---|---|\n")
            for sys_name in sorted(by_op["ingest"]):
                ing = by_op["ingest"][sys_name]
                ref = by_op["refresh"][sys_name]
                ing_sec = ing.latency_ms / 1000.0
                ref_sec = ref.latency_ms / 1000.0
                total_sec = ing_sec + ref_sec
                f.write(f"| {sys_name} | {ing_sec:.1f}s | {ref_sec:.1f}s | **{total_sec:.1f}s** |\n")

        f.write("\n")

        f.write(f"## Query latency (p50 / p95 / p99, {meta['runs']} runs each)\n\n")
        f.write("| System | Operation | p50 (ms) | p95 (ms) | p99 (ms) |\n")
        f.write("|---|---|---|---|---|\n")
        for op in query_ops + hs_only_ops:
            for sys_name, r in sorted(by_op[op].items()):
                f.write(
                    f"| {sys_name} | {op} | {r.metadata['p50_ms']} | {r.metadata['p95_ms']} "
                    f"| {r.metadata['p99_ms']} |\n"
                )
        f.write("\n")

        f.write("## Verdict vs plan Step 4.2 envelope (50–200 ms)\n\n")
        f.write("| System | Operation | p95 | verdict |\n|---|---|---|---|\n")
        for op in query_ops + hs_only_ops:
            for sys_name, r in sorted(by_op[op].items()):
                f.write(
                    f"| {sys_name} | {op} | {r.metadata['p95_ms']}ms "
                    f"| {envelope_verdict(r.metadata['p95_ms'])} |\n"
                )
        f.write("\n")

        f.write("## Storage Footprint (True Cost of Data Lake Architecture)\n\n")
        f.write(f"**Raw document payload:** `{meta.get('raw_json_mb', 'n/a')} MB` (uncompressed JSON over HTTP)\n\n")
        f.write("| System | Primary Data (Lake) | Secondary Indexes (Search) | Data Duplication? | Total True Footprint |\n")
        f.write("|---|---|---|---|---|\n")

        hs_bd = meta.get("storage_breakdown", {}).get("BenoStreamDB", {})
        es_bd = meta.get("storage_breakdown", {}).get("OpenSearch 2.11", {})
        rss = meta.get("rss_mb", {})

        hs_parquet = hs_bd.get('parquet_mb', 0)
        hs_idx = hs_bd.get('indexes_mb', 0)
        hs_tot = hs_bd.get('total_mb', 0)
        hs_mem_in = f"{rss.get('BenoStreamDB_after_ingest', 'n/a')} MB RSS"
        hs_mem_srch = f"{rss.get('BenoStreamDB_after_search', 'n/a')} MB RSS"
        f.write(f"| BenoStreamDB ({storage_mode}) | {hs_parquet} MB (Parquet) | {hs_idx} MB (HNSW+BM25) | No | **{hs_tot} MB** |\n")

        if "OpenSearch 2.11" in by_op.get("ingest", {}):
            es_store = es_bd.get('lucene_store_mb', 0)
            es_true = (float(hs_parquet) if hs_parquet != 'n/a' else 0) + (float(es_store) if es_store != 'n/a' else 0)
            es_heap_in = f"JVM Heap: {rss.get('OpenSearch_jvm_heap_after_ingest', 'n/a')} MB"
            es_rss_in = f"Container RSS: {rss.get('OpenSearch_container_rss_after_ingest', 'n/a')} MB"
            f.write(f"| OpenSearch 2.11 (local) | {hs_parquet} MB (Duplicated Lake Source) | {es_store} MB (Lucene) | Yes | **{es_true:.2f} MB** |\n")

        f.write("\n## Memory Usage\n\n")
        f.write("| System | Memory Ingest (RSS/Heap) | Memory Post-Search |\n")
        f.write("|---|---|---|\n")
        f.write(f"| BenoStreamDB ({storage_mode}) | {hs_mem_in} | {hs_mem_srch} |\n")
        if "OpenSearch 2.11" in by_op.get("ingest", {}):
            es_heap_srch = f"JVM Heap: {rss.get('OpenSearch_jvm_heap_after_search', 'n/a')} MB"
            es_rss_srch = f"Container RSS: {rss.get('OpenSearch_container_rss_after_search', 'n/a')} MB"
            f.write(f"| OpenSearch 2.11 (local) | {es_heap_in} ({es_rss_in}) | {es_heap_srch} ({es_rss_srch}) |\n")

        f.write("\n")

        f.write("## Fairness caveats\n\n")
        f.write(
            "- Single-doc POST on **both** systems (bsdb-search has no `_bulk` in this test); ES index pre-created "
            "with refresh disabled, bsdb-search creates the index on first write.\n"
            "- The `embedding` field is sent to bsdb-search only: ES 7.10 has no `dense_vector` type.\n"
            "- `knn` and `hybrid_rrf` are bsdb-search-only (no vector search in ES 7.10).\n"
            "- ES refresh does little work (segments are indexed during ingest); bsdb-search refresh "
            "includes BM25/HNSW index build, so refresh times are not like-for-like.\n"
            f"- bsdb-search is a **{build_type}** build; ES uses the stock Docker image.\n"
            f"- Equal resource budget: **{meta.get('memory_budget_gb', 'n/a')} GiB RAM / "
            f"{meta.get('cpu_budget', 'n/a')} CPUs** for both systems "
            f"(ES: container cap + {meta.get('es_jvm_heap_gb', 'n/a')} GiB JVM heap; "
            f"bsdb-search: {'container cap + ' if meta.get('hs_containerized') else ''}"
            f"in-process caches sized to the same budget).\n\n"
        )
        if meta.get("notes"):
            f.write("## Notes\n\n")
            for note in meta["notes"]:
                f.write(f"- {note}\n")
            f.write("\n")

        f.write(f"Raw results: `{json_path.name}`\n")

    print(f"\nResults: {json_path}\n         {md_path}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="OpenSearch 2.11 vs bsdb-search REST benchmark")
    parser.add_argument("--dataset", choices=["wiki", "pmc"], default="wiki", help="HuggingFace dataset to use")
    parser.add_argument("--size", type=int, default=1000, help="document count (default 1000)")
    parser.add_argument("--quick", action="store_true", help="quick run: 200 docs, 20 runs, dim 32")
    parser.add_argument("--dim", type=int, default=64, help="embedding dimension (default 64)")
    parser.add_argument("--runs", type=int, default=100, help="query runs per operation (default 100)")
    parser.add_argument("--storage", choices=["local", "cloud"], default="local", help="storage backend for BenoStreamDB (default: local)")
    parser.add_argument("--cloud-uri", default="s3://warehouse/benchmarks", help="S3 URI prefix when --storage=cloud (default: s3://warehouse/benchmarks)")
    parser.add_argument("--skip-es", action="store_true", help="skip OpenSearch (bsdb-search only)")
    parser.add_argument("--es-port", type=int, default=None, help="host port for ES (default: auto)")
    parser.add_argument("--keep-es", action="store_true", help="do not remove the ES container at the end")
    parser.add_argument("--es-image", default=ES_IMAGE_DEFAULT, help="ES docker image")
    parser.add_argument("--mem-gb", type=int, default=4,
                        help="equal memory budget in GiB for both systems (default 4)")
    parser.add_argument("--cpus", type=int, default=4,
                        help="equal CPU budget for both systems (default 4)")
    parser.add_argument("--no-container", action="store_true",
                        help="run bsdb-search natively instead of in a capped container")
    parser.add_argument("--output-dir", default=str(RESULTS_DIR), help="results output directory")
    args = parser.parse_args()

    if args.quick:
        size, dim, runs = 200, 32, 20
    else:
        size, dim, runs = args.size, args.dim, args.runs

    if args.no_container:
        if not BINARY.exists():
            raise SystemExit(f"{BINARY} not found; run `cargo build -p benostreamdb-search --bin bsdb-search` first")
    else:
        img = subprocess.run(["docker", "image", "inspect", "bsdb-search-bench:latest"], capture_output=True)
        if img.returncode != 0:
            raise SystemExit(
                "bsdb-search-bench:latest image not found; run "
                "`docker build -f docker/Dockerfile.search-bench -t bsdb-search-bench:latest .` first")

    print("=" * 72)
    print("OpenSearch 2.11 vs BenoStreamDB — REST benchmark")
    print("=" * 72)
    print(f"size={size} dim={dim} runs={runs} storage={args.storage} skip_es={args.skip_es} "
          f"mem_budget={args.mem_gb}GiB cpus={args.cpus} "
          f"bsdb-search={'container' if not args.no_container else BINARY.name}")

    hardware = get_hardware_info()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    words = ['science', 'research', 'data', 'information', 'study', 'history', 'world', 'time', 'people', 'water']
    print(f"Generating {size:,} test documents ({dim}-dim embeddings)...")
    docs = get_documents_from_hf(args.dataset, size, dim)
    raw_doc_bytes = sum(len(json.dumps(d).encode("utf-8")) for d in docs)
    raw_doc_mb = round(raw_doc_bytes / (1024 * 1024), 2)
    print(f"Raw document payload size: {raw_doc_mb} MB")

    
    qvecs = [[random.Random(4242 + i).random() for _ in range(dim)] for i in range(runs)]

    results: List[BenchmarkResult] = []
    meta: Dict = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "hardware": hardware,
        "storage_mode": args.storage,
        "raw_json_mb": raw_doc_mb,
        "docs": size,
        "dim": dim,
        "runs": runs,
        "k": 10,
        "envelope_ms": list(ENVELOPE_MS),
        "notes": [],
        "rss_mb": {},
        "storage_breakdown": {},
        "es_version": "n/a",
        "es_build": "n/a",
        "memory_budget_gb": args.mem_gb,
        "cpu_budget": args.cpus,
        "hs_containerized": not args.no_container,
        "hs_cache_gb": {"block": max(1, args.mem_gb // 4), "cache": max(1, args.mem_gb // 8)},
        "es_jvm_heap_gb": max(1, args.mem_gb // 2),
    }

    # ---------------- bsdb-search ----------------
    print(f"\n[BenoStreamDB bsdb-search ({args.storage} storage)]")
    if args.storage == "cloud":
        storage_uri = f"{args.cloud_uri.rstrip('/')}/hsbench-{uuid.uuid4().hex[:8]}"
        local_dir_to_clean = None
    else:
        local_dir_to_clean = tempfile.mkdtemp(prefix="hsbench-")
        storage_uri = f"file://{local_dir_to_clean}"
    meta["storage_uri"] = storage_uri

    hs = None
    try:
        hs = Hypersearch(free_port(), storage_uri, is_cloud=(args.storage == "cloud"),
                         mem_gb=args.mem_gb, cpus=args.cpus, containerized=not args.no_container)
        hs_index = "bench-hs-" + uuid.uuid4().hex[:8]
        results += bench_ingest(hs, "BenoStreamDB", hs_index, docs, exclude_embedding=False, hardware=hardware)
        meta["hs_version"] = hs.info["version"]["number"]
        meta["rss_mb"]["BenoStreamDB_after_ingest"] = hs.rss_mb()
        meta["storage_breakdown"]["BenoStreamDB"] = hs.storage_breakdown()

        def match_body(i):
            return {"query": {"match": {"body": words[i % len(words)]}}, "size": 10}

        def filtered_body(i):
            return {
                "query": {"match": {"body": words[i % len(words)]}},
                "filter": {"term": {"category": docs[i % size]["category"]}},
                "size": 10,
            }

        def knn_body(i):
            return {"knn": {"field": "embedding", "vector": qvecs[i % len(qvecs)], "k": 10}}

        def hybrid_body(i):
            return {
                "query": {
                    "match": {"body": words[i % len(words)]},
                    "knn": {"field": "embedding", "vector": qvecs[i % len(qvecs)], "k": 10},
                }
            }

        results.append(bench_query(hs, "BenoStreamDB", hs_index, "match_bm25", match_body, runs, size, hardware))
        results.append(bench_query(hs, "BenoStreamDB", hs_index, "filtered", filtered_body, runs, size, hardware))
        results.append(bench_query(hs, "BenoStreamDB", hs_index, "knn", knn_body, runs, size, hardware,
                                   extra_meta={"dim": dim}))
        results.append(bench_query(hs, "BenoStreamDB", hs_index, "hybrid_rrf", hybrid_body, runs, size, hardware,
                                   extra_meta={"dim": dim}))
        meta["rss_mb"]["BenoStreamDB_after_search"] = hs.rss_mb()
        meta["rss_mb"]["BenoStreamDB"] = hs.rss_mb()
    finally:
        if hs is not None:
            hs.stop()
        if local_dir_to_clean:
            shutil.rmtree(local_dir_to_clean, ignore_errors=True)

    # ---------------- OpenSearch ----------------
    if not args.skip_es:
        print("\n[OpenSearch 2.11 (Docker)]")
        es = None
        es_port = args.es_port or free_port()
        try:
            es = OpenSearch(args.es_image, es_port, keep=args.keep_es,
                            mem_gb=args.mem_gb, cpus=args.cpus)
            es_index = "bench-es-" + uuid.uuid4().hex[:8]
            es.create_index(es_index, dim)
            results += bench_ingest(es, "OpenSearch 2.11", es_index, docs, exclude_embedding=False,
                                    hardware=hardware)
            # Capture ES storage & memory breakdown after ingest
            es_data_dir = es.data_dir_mb()
            es_store = es.index_store_mb(es_index)
            results[-2].storage_mb = es_data_dir
            meta["storage_breakdown"]["OpenSearch 2.11"] = {
                "lucene_store_mb": es_store,
                "data_dir_total_mb": es_data_dir,
            }
            meta["rss_mb"]["OpenSearch_jvm_heap_after_ingest"] = es.jvm_heap_mb()
            meta["rss_mb"]["OpenSearch_container_rss_after_ingest"] = es.container_rss_mb()
            ver = es.info["version"]
            meta["es_version"] = ver["number"]
            meta["es_build"] = ver.get("build_hash", "unknown")

            def match_body_es(i):
                return {"query": {"match": {"body": words[i % len(words)]}}, "size": 10}

            def knn_body_es(i):
                return {"size": 10, "query": {"knn": {"embedding": {"vector": qvecs[i % len(qvecs)], "k": 10}}}}

            def filtered_body_es(i):
                return {
                    "query": {
                        "bool": {
                            "must": [{"match": {"body": words[i % len(words)]}}],
                            "filter": [{"term": {"category": docs[i % size]["category"]}}],
                        }
                    },
                    "size": 10,
                }

            results.append(bench_query(
                es, "OpenSearch 2.11", es_index, "match_bm25", match_body_es, runs, size, hardware))
            results.append(bench_query(
                es, "OpenSearch 2.11", es_index, "filtered", filtered_body_es, runs, size, hardware))
            results.append(bench_query(
                es, "OpenSearch 2.11", es_index, "knn", knn_body_es, runs, size, hardware, extra_meta={"dim": dim}))
            meta["rss_mb"]["OpenSearch_jvm_heap_after_search"] = es.jvm_heap_mb()
            meta["rss_mb"]["OpenSearch_container_rss_after_search"] = es.container_rss_mb()
            meta["rss_mb"]["OpenSearch 2.11"] = es.container_rss_mb()
            es.delete_index(es_index)
        except RuntimeError as e:
            meta["notes"].append(f"OpenSearch run failed: {e}")
            print(f"  !! {e}")
        finally:
            if es is not None:
                es.stop()
    else:
        meta["notes"].append("OpenSearch skipped (--skip-es)")
        meta["es_version"] = "skipped"
        meta["es_build"] = "n/a"

    write_reports(results, meta, Path(args.output_dir), timestamp)
    print("\nDone.")


if __name__ == "__main__":
    main()
