#!/usr/bin/env python3
"""
ES 7.10.2 vs HyperStreamDB (hypersearch) — REST API benchmark (plan Step 4.2).

Spawns a local Elasticsearch 7.10.2 (Docker, single-node, security disabled)
and a local ``hypersearch`` binary, feeds both the same document stream
(one document per HTTP POST — hypersearch has no ``_bulk``), and measures:

  * ingest throughput      (both; single-doc POST /{index}/_doc)
  * refresh latency        (both; time until the data is searchable)
  * BM25 ``match`` query   (both; p50/p95/p99 over N queries)
  * filtered query         (both; ``match`` + category ``term`` filter)
  * HNSW ``knn``           (hypersearch only — ES 7.10 has no dense_vector)
  * hybrid ``match``+``knn`` (hypersearch only; RRF fusion)

Fairness notes:
  * Both systems receive one document per POST (no ``_bulk`` on hypersearch),
    an explicit refresh before search, and run on the same host.
  * ES runs single-node, 1 shard, 0 replicas, refresh disabled during ingest
    (the standard way to measure ES ingest), with the image's default 1 GiB
    JVM heap.
  * The ``embedding`` field is part of the hypersearch payload only: ES 7.10
    has no ``dense_vector`` type, so shipping 64 floats per document to a
    7.10 search cluster would not represent any real workload. Every other
    field is identical on both sides.
  * hypersearch's first write also pays one-time index creation (manifest +
    Iceberg init); ES's index is pre-created for mapping/settings. The
    one-time cost is amortized over the whole ingest stream.
  * Latencies include the localhost HTTP round trip, measured identically.

Run (from the repo root, after ``cargo build -p hyperstreamdb-search --bin hypersearch``):

    ./venv/bin/python benchmarks/competitive/benchmark_es710.py
    ./venv/bin/python benchmarks/competitive/benchmark_es710.py --quick
    ./venv/bin/python benchmarks/competitive/benchmark_es710.py --skip-es

Writes timestamped JSON + Markdown result pairs into
``benchmarks/competitive/benchmark_results/``.
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
BINARY = REPO_ROOT / "target" / "debug" / "hypersearch"
RESULTS_DIR = Path(__file__).resolve().parent / "benchmark_results"

ES_IMAGE_DEFAULT = "docker.elastic.co/elasticsearch/elasticsearch:7.10.2"
ES_CONTAINER = "es710-bench"
ES_STARTUP_DEADLINE_S = 180.0
HYPERSEARCH_STARTUP_DEADLINE_S = 60.0
HTTP_TIMEOUT_S = 30.0


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

def make_vocab(n: int = 512) -> List[str]:
    """Deterministic pronounceable word vocabulary (stable across runs)."""
    prefixes = ["ne", "te", "vo", "ra", "lu", "mi", "ka", "se", "bo", "fa", "gi", "hu"]
    roots = ["ra", "lo", "vi", "na", "to", "mi", "de", "su", "pa", "ri", "le", "no"]
    suffixes = ["n", "t", "s", "x"]
    vocab = [f"{p}{r}{s}" for p in prefixes for r in roots for s in suffixes]
    random.Random(1337).shuffle(vocab)
    return vocab[:n]


def generate_documents(n: int, dim: int, vocab: List[str]) -> List[Dict]:
    rng = random.Random(1337)
    cats = [f"cat-{i}" for i in range(8)]
    docs = []
    for i in range(n):
        docs.append({
            "title": " ".join(rng.choice(vocab) for _ in range(rng.randint(4, 8))),
            "body": " ".join(rng.choice(vocab) for _ in range(rng.randint(80, 120))),
            "category": rng.choice(cats),
            "price": round(rng.uniform(1.0, 1000.0), 2),
            "ts": f"2026-01-{1 + i % 28:02d}T00:00:00Z",
            "embedding": [rng.random() for _ in range(dim)],
        })
    return docs


# --------------------------------------------------------------------------
# System adapters
# --------------------------------------------------------------------------

class Hypersearch:
    """Spawned ``hypersearch`` binary on a free local port."""

    def __init__(self, port: int, storage_dir: str):
        self.base = f"http://127.0.0.1:{port}"
        env = {
            **os.environ,
            "HYPERSEARCH_BIND": "127.0.0.1",
            "HYPERSEARCH_PORT": str(port),
            "HYPERSEARCH_STORAGE_URI": f"file://{storage_dir}",
        }
        self.proc = subprocess.Popen(
            [str(BINARY)],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.session = requests.Session()
        deadline = time.time() + HYPERSEARCH_STARTUP_DEADLINE_S
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
            raise RuntimeError(f"hypersearch did not become ready within {HYPERSEARCH_STARTUP_DEADLINE_S:.0f}s")
        self.info = self.session.get(self.base + "/", timeout=5).json()
        self.storage_dir = Path(storage_dir)

    def index_doc(self, index: str, doc: Dict) -> requests.Response:
        return self.session.post(self.base + f"/{index}/_doc", json=doc, timeout=HTTP_TIMEOUT_S)

    def refresh(self, index: str) -> requests.Response:
        return self.session.post(self.base + f"/{index}/_refresh", timeout=HTTP_TIMEOUT_S)

    def search(self, index: str, body: Dict) -> requests.Response:
        return self.session.post(self.base + f"/{index}/_search", json=body, timeout=HTTP_TIMEOUT_S)

    def rss_mb(self) -> Optional[float]:
        try:
            for line in Path(f"/proc/{self.proc.pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024.0, 1)
        except OSError:
            pass
        return None

    def storage_mb(self) -> float:
        total = sum(p.stat().st_size for p in self.storage_dir.rglob("*") if p.is_file())
        return round(total / 1024 / 1024, 2)

    def stop(self) -> None:
        try:
            self.proc.terminate()
            self.proc.wait(timeout=10)
        except Exception:
            try:
                self.proc.kill()
            except OSError:
                pass


class Elasticsearch:
    """Docker-managed single-node ES 7.10.2 (security off)."""

    def __init__(self, image: str, port: int, keep: bool = False):
        # Remove any stale container from a previous run.
        subprocess.run(["docker", "rm", "-f", ES_CONTAINER], capture_output=True)
        self.proc = subprocess.run(
            [
                "docker", "run", "-d", "--name", ES_CONTAINER,
                "-p", f"{port}:9200",
                "-e", "discovery.type=single-node",
                "-e", "xpack.security.enabled=false",
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

    def create_index(self, index: str) -> None:
        body = {
            "settings": {
                "index": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "refresh_interval": "-1",
                }
            },
            "mappings": {
                "properties": {
                    "title": {"type": "text"},
                    "body": {"type": "text"},
                    "category": {"type": "keyword"},
                    "price": {"type": "float"},
                    "ts": {"type": "date"},
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

    def data_dir_mb(self) -> Optional[float]:
        r = subprocess.run(
            ["docker", "exec", ES_CONTAINER, "du", "-sb", "/usr/share/elasticsearch/data"],
            capture_output=True, text=True,
        )
        if r.returncode == 0:
            return round(int(r.stdout.split()[0]) / 1024 / 1024, 2)
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
    print(f"  ingesting {n:,} docs (single-doc POST)...")
    t0 = time.time()
    per_doc = []
    for i, doc in enumerate(docs):
        payload = {k: v for k, v in doc.items() if not (exclude_embedding and k == "embedding")}
        t = time.time()
        r = system.index_doc(index, payload)
        per_doc.append((time.time() - t) * 1000.0)
        if r.status_code not in (200, 201):
            raise RuntimeError(f"{system_name} ingest failed at doc {i}: {r.status_code} {r.text[:300]}")
        if (i + 1) % max(1, n // 10) == 0:
            print(f"    {i + 1:,}/{n:,}")
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

ENVELOPE_MS = (50.0, 200.0)  # plan Step 4.2 search-latency acceptance envelope


def envelope_verdict(p95_ms: float) -> str:
    lo, hi = ENVELOPE_MS
    if p95_ms <= lo:
        return f"below envelope (<{lo:.0f}ms)"
    if p95_ms <= hi:
        return f"within envelope ({lo:.0f}-{hi:.0f}ms)"
    return f"ABOVE envelope (> {hi:.0f}ms)"


def write_reports(results: List[BenchmarkResult], meta: Dict, outdir: Path, timestamp: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / f"es710_hypersearch_{timestamp}.json"
    md_path = outdir / f"es710_hypersearch_{timestamp}.md"

    with open(json_path, "w") as f:
        json.dump({"run": meta, "results": [asdict(r) for r in results]}, f, indent=2)

    by_op: Dict[str, Dict[str, BenchmarkResult]] = {}
    for r in results:
        by_op.setdefault(r.operation, {})[r.system] = r

    query_ops = [op for op in ("match_bm25", "filtered") if op in by_op]
    hs_only_ops = [op for op in ("knn", "hybrid_rrf") if op in by_op]

    with open(md_path, "w") as f:
        f.write("# ES 7.10.2 vs HyperStreamDB — REST API Benchmark\n\n")
        f.write(f"**Generated:** {meta['generated']}  \n")
        f.write(f"**Host:** {meta['hardware']} ({platform.system()} {platform.release()})  \n")
        f.write(f"**ES:** {meta['es_version']} (build `{meta['es_build']}`, Docker, single-node, 1 shard, no replicas, 1 GiB JVM)  \n")
        f.write(f"**hypersearch:** {meta['hs_version']} (debug build, in-process HNSW/BM25)  \n")
        f.write(f"**Dataset:** {meta['docs']:,} docs × {meta['dim']}-dim embeddings, {meta['runs']} query runs, k=10\n\n")

        f.write("## Ingest (single-doc POST; hypersearch has no `_bulk`)\n\n")
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

        f.write("## Storage after ingest\n\n")
        f.write("| System | data dir (MB) | process RSS (MB) |\n|---|---|---|\n")
        for r in results:
            if r.operation == "ingest":
                f.write(f"| {r.system} | {r.storage_mb if r.storage_mb is not None else 'n/a'} "
                        f"| {meta.get('rss_mb', {}).get(r.system, 'n/a')} |\n")
        f.write("\n")

        f.write("## Fairness caveats\n\n")
        f.write(
            "- Single-doc POST on **both** systems (hypersearch has no `_bulk`); ES index pre-created "
            "with refresh disabled, hypersearch creates the index on first write.\n"
            "- The `embedding` field is sent to hypersearch only: ES 7.10 has no `dense_vector` type.\n"
            "- `knn` and `hybrid_rrf` are hypersearch-only (no vector search in ES 7.10).\n"
            "- ES refresh does little work (segments are indexed during ingest); hypersearch refresh "
            "includes BM25/HNSW index build, so refresh times are not like-for-like.\n"
            "- hypersearch is a **debug** build; ES uses the stock Docker image.\n"
            "- Both systems run on the same host; ES JVM heap is the image default (1 GiB).\n\n"
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
    parser = argparse.ArgumentParser(description="ES 7.10.2 vs hypersearch REST benchmark")
    parser.add_argument("--size", type=int, default=1000, help="document count (default 1000)")
    parser.add_argument("--quick", action="store_true", help="quick run: 200 docs, 20 runs, dim 32")
    parser.add_argument("--dim", type=int, default=64, help="embedding dimension (default 64)")
    parser.add_argument("--runs", type=int, default=100, help="query runs per operation (default 100)")
    parser.add_argument("--skip-es", action="store_true", help="skip Elasticsearch (hypersearch only)")
    parser.add_argument("--es-port", type=int, default=None, help="host port for ES (default: auto)")
    parser.add_argument("--keep-es", action="store_true", help="do not remove the ES container at the end")
    parser.add_argument("--es-image", default=ES_IMAGE_DEFAULT, help="ES docker image")
    parser.add_argument("--output-dir", default=str(RESULTS_DIR), help="results output directory")
    args = parser.parse_args()

    if args.quick:
        size, dim, runs = 200, 32, 20
    else:
        size, dim, runs = args.size, args.dim, args.runs

    if not BINARY.exists():
        raise SystemExit(f"{BINARY} not found; run `cargo build -p hyperstreamdb-search --bin hypersearch` first")

    print("=" * 72)
    print("ES 7.10.2 vs HyperStreamDB — REST benchmark")
    print("=" * 72)
    print(f"size={size} dim={dim} runs={runs} skip_es={args.skip_es}")

    hardware = get_hardware_info()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    vocab = make_vocab(512)
    docs = generate_documents(size, dim, vocab)
    words = random.Random(2026).sample(vocab, min(runs, len(vocab)))
    qvecs = [[random.Random(4242 + i).random() for _ in range(dim)] for i in range(runs)]

    results: List[BenchmarkResult] = []
    meta: Dict = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "hardware": hardware,
        "docs": size,
        "dim": dim,
        "runs": runs,
        "k": 10,
        "envelope_ms": list(ENVELOPE_MS),
        "notes": [],
        "rss_mb": {},
        "es_version": "n/a",
        "es_build": "n/a",
    }

    # ---------------- hypersearch ----------------
    print("\n[HyperStreamDB hypersearch]")
    storage = tempfile.mkdtemp(prefix="hsbench-")
    hs = None
    try:
        hs = Hypersearch(free_port(), storage)
        hs_index = "bench-hs-" + uuid.uuid4().hex[:8]
        results += bench_ingest(hs, "HyperStreamDB", hs_index, docs, exclude_embedding=False, hardware=hardware)
        meta["hs_version"] = hs.info["version"]["number"]
        meta["rss_mb"]["HyperStreamDB"] = hs.rss_mb()

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

        results.append(bench_query(hs, "HyperStreamDB", hs_index, "match_bm25", match_body, runs, size, hardware))
        results.append(bench_query(hs, "HyperStreamDB", hs_index, "filtered", filtered_body, runs, size, hardware))
        results.append(bench_query(hs, "HyperStreamDB", hs_index, "knn", knn_body, runs, size, hardware,
                                   extra_meta={"dim": dim}))
        results.append(bench_query(hs, "HyperStreamDB", hs_index, "hybrid_rrf", hybrid_body, runs, size, hardware,
                                   extra_meta={"dim": dim}))
        meta["rss_mb"]["HyperStreamDB"] = hs.rss_mb()
    finally:
        if hs is not None:
            hs.stop()
        shutil.rmtree(storage, ignore_errors=True)

    # ---------------- Elasticsearch ----------------
    if not args.skip_es:
        print("\n[Elasticsearch 7.10.2 (Docker)]")
        es = None
        es_port = args.es_port or free_port()
        try:
            es = Elasticsearch(args.es_image, es_port, keep=args.keep_es)
            es_index = "bench-es-" + uuid.uuid4().hex[:8]
            es.create_index(es_index)
            results += bench_ingest(es, "Elasticsearch 7.10.2", es_index, docs, exclude_embedding=True,
                                    hardware=hardware)
            # ES data-directory size after ingest (bench_ingest can't reach it for us).
            results[-2].storage_mb = es.data_dir_mb()
            ver = es.info["version"]
            meta["es_version"] = ver["number"]
            meta["es_build"] = ver.get("build_hash", "unknown")

            def match_body_es(i):
                return {"query": {"match": {"body": words[i % len(words)]}}, "size": 10}

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
                es, "Elasticsearch 7.10.2", es_index, "match_bm25", match_body_es, runs, size, hardware))
            results.append(bench_query(
                es, "Elasticsearch 7.10.2", es_index, "filtered", filtered_body_es, runs, size, hardware))
            es.delete_index(es_index)
        except RuntimeError as e:
            meta["notes"].append(f"Elasticsearch run failed: {e}")
            print(f"  !! {e}")
        finally:
            if es is not None:
                es.stop()
    else:
        meta["notes"].append("Elasticsearch skipped (--skip-es)")
        meta["es_version"] = "skipped"
        meta["es_build"] = "n/a"

    write_reports(results, meta, Path(args.output_dir), timestamp)
    print("\nDone.")


if __name__ == "__main__":
    main()
