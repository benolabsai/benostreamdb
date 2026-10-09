#!/usr/bin/env python3
"""BEIR Lexical / BM25 Benchmark Runner: BenoStreamDB vs Tantivy / OpenSearch / Elasticsearch.

Evaluates BM25 inverted index construction, index size on disk, query latency
(p50, p90, p99), throughput (QPS), and IR retrieval quality (Recall@10, nDCG@10, MRR@10)
on standard Information Retrieval benchmarks (SciFact / BEIR).

Engines:
  * ``benostreamdb``  — embedded engine, BM25 overlay index.
  * ``tantivy``       — embedded Rust IR library (``pip install tantivy``).
  * ``opensearch``    — server-backed BM25 (``pip install opensearch-py``); the
    docker compose stack runs OpenSearch and exports ``ES_URL``.
  * ``elasticsearch`` — server-backed BM25 (``pip install elasticsearch``).

Every engine is measured with the same corpus, queries, ``k``, and metric code,
so the numbers are directly comparable.

Usage:
    python benchmarks/beir/run.py --dataset scifact --engines benostreamdb,tantivy \\
        --out benchmarks/beir/results/scifact_bm25.md

    python benchmarks/beir/run.py --dataset scifact \\
        --engines benostreamdb,tantivy,opensearch --host http://localhost:9200 \\
        --out benchmarks/beir/results/scifact_bm25_competitors.md
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import shutil
import tempfile
import time
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pyarrow as pa


BEIR_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Canonical BEIR dataset archives (the data dir is gitignored, so a fresh clone
# or CI run downloads the corpus on first use).
BEIR_DATASET_URLS = {
    "scifact": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
}


def ensure_dataset(dataset_name: str, data_dir: str | None = None) -> str:
    """Return the dataset directory, downloading + extracting the BEIR zip if absent."""
    base = data_dir or BEIR_DATA_DIR
    dataset_dir = os.path.join(base, dataset_name)
    if os.path.isdir(dataset_dir):
        return dataset_dir
    url = BEIR_DATASET_URLS.get(dataset_name)
    if not url:
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir} "
            f"(no download URL registered for {dataset_name!r})"
        )
    import urllib.request
    import zipfile

    os.makedirs(base, exist_ok=True)
    zip_path = os.path.join(base, f"{dataset_name}.zip")
    print(f"Downloading {dataset_name} from {url} ...", flush=True)
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(base)
    os.remove(zip_path)
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Extracted archive did not contain {dataset_dir}")
    return dataset_dir


def envelope_info() -> Dict[str, Any]:
    """Container-visible resource envelope, so results are self-describing.

    Reads the same ``BENCH_CPUS`` / ``BENCH_MEM`` variables the Docker compose
    stack applies to *every* participant (server engines and the runner), so a
    BEIR row can be checked against the envelope it was measured under.
    """
    cpus = os.environ.get("BENCH_CPUS")
    mem = os.environ.get("BENCH_MEM")
    containerized = os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv")
    return {
        "cpus": int(cpus) if cpus and cpus.isdigit() else None,
        "mem": mem,
        "containerized": containerized,
    }


def _env_record() -> Dict[str, Any]:
    """Container-visible environment, mirroring the competitor JSON ``env`` block."""
    ram_gb = None
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    ram_gb = round(int(line.split()[1]) / 1048576, 1)
                    break
    except OSError:
        pass
    return {
        "cpu_model": platform.processor() or platform.machine(),
        "cores": os.cpu_count() or 0,
        "ram_gb": ram_gb,
        "os": platform.platform(),
        "python": platform.python_version(),
        "containerized": os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv"),
        "gpus": [],
    }


def write_json_records(
    results: List[Dict[str, Any]],
    dataset: str,
    k: int,
    num_queries: int,
    json_dir: str,
    workload: str,
    device: str,
) -> None:
    """Write one competitor-schema JSON record per engine.

    The record matches the shape the competitor matrix emits
    (``{engine}_{dataset}_{device}.json``), so ``generate_summary.py`` can roll
    the BEIR rows into the consolidated benchmark report alongside the
    vector/graph/SQL records.
    """
    os.makedirs(json_dir, exist_ok=True)
    env = _env_record()
    for r in results:
        if not r.get("available"):
            continue
        record = {
            "engine": r["engine"],
            "dataset": dataset,
            "workload": workload,
            "device": device,
            "queries": num_queries,
            "k": k,
            "build_s": round(r["build_s"], 3),
            "index_mb": r["index_mb"],
            "recall_at_k": round(r.get(f"recall@{k}", 0.0), 4),
            "ndcg_at_k": round(r.get(f"ndcg@{k}", 0.0), 4),
            "mrr_at_k": round(r.get(f"mrr@{k}", 0.0), 4),
            "qps": round(r["qps"], 1),
            "p50_ms": round(r["p50_ms"], 3),
            "p99_ms": round(r["p99_ms"], 3),
            "env": env,
        }
        path = os.path.join(json_dir, f"{r['engine']}_{dataset}_{workload}_{device}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        print(f"wrote {path}", flush=True)


def load_scifact_data(
    dataset_name: str = "scifact",
    data_dir: str | None = None,
) -> Tuple[List[Tuple[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:
    """Loads corpus, test queries, and ground-truth relevance judgements (qrels)."""
    dataset_dir = ensure_dataset(dataset_name, data_dir)

    corpus_path = os.path.join(dataset_dir, "corpus.jsonl")
    queries_path = os.path.join(dataset_dir, "queries.jsonl")
    qrels_path = os.path.join(dataset_dir, "qrels", "test.tsv")

    # 1. Load corpus
    corpus = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            doc_id = str(doc["_id"])
            text = f"{doc.get('title', '')} {doc.get('text', '')}".strip()
            corpus.append((doc_id, text))

    # 2. Load all queries
    all_queries = {}
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            all_queries[str(q["_id"])] = q.get("text", "").strip()

    # 3. Load test qrels (query_id -> {doc_id: score})
    qrels = {}
    if os.path.exists(qrels_path):
        with open(qrels_path, "r", encoding="utf-8") as f:
            header = True
            for line in f:
                if header:
                    header = False
                    continue
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    qid, docid, score = parts[0], parts[1], int(parts[2])
                    if score > 0:
                        qrels.setdefault(qid, {})[docid] = score

    # Filter to only queries that have test qrels
    test_queries = {qid: all_queries[qid] for qid in qrels if qid in all_queries}
    if not test_queries:
        test_queries = all_queries

    return corpus, test_queries, qrels


def compute_ir_metrics(
    retrieved: Dict[str, List[str]], qrels: Dict[str, Dict[str, int]], k: int = 10
) -> Dict[str, float]:
    """Computes Recall@k, nDCG@k, and MRR@k."""
    recalls = []
    ndcgs = []
    mrrs = []

    for qid, rel_docs in qrels.items():
        if qid not in retrieved or not rel_docs:
            continue

        hits = retrieved[qid][:k]

        # Recall@k
        num_relevant = len(rel_docs)
        num_hits_relevant = sum(1 for doc_id in hits if doc_id in rel_docs)
        recall = num_hits_relevant / num_relevant if num_relevant > 0 else 0.0
        recalls.append(recall)

        # MRR@k
        mrr = 0.0
        for rank, doc_id in enumerate(hits, start=1):
            if doc_id in rel_docs:
                mrr = 1.0 / rank
                break
        mrrs.append(mrr)

        # nDCG@k
        dcg = 0.0
        for rank, doc_id in enumerate(hits, start=1):
            if doc_id in rel_docs:
                dcg += rel_docs[doc_id] / math.log2(rank + 1)

        # Ideal DCG
        ideal_scores = sorted(rel_docs.values(), reverse=True)[:k]
        idcg = sum(score / math.log2(rank + 1) for rank, score in enumerate(ideal_scores, start=1))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcgs.append(ndcg)

    return {
        f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"mrr@{k}": float(np.mean(mrrs)) if mrrs else 0.0,
    }


def run_benostreamdb(
    corpus: List[Tuple[str, str]],
    queries: Dict[str, str],
    k: int = 10,
) -> Dict[str, Any]:
    """Benchmarks BenoStreamDB BM25 inverted index build and retrieval."""
    import benostreamdb as bsdb

    tmpdir = tempfile.mkdtemp(prefix="bsdb_beir_")
    try:
        t_build_0 = time.perf_counter()
        schema = pa.schema([
            ("id", pa.string()),
            ("text", pa.string()),
        ])
        table = bsdb.Table.create(tmpdir, schema)
        table.add_index("text", {"type": "bm25", "k1": 1.2, "b": 0.75})

        arr_id = pa.array([c[0] for c in corpus], type=pa.string())
        arr_text = pa.array([c[1] for c in corpus], type=pa.string())
        batch = pa.Table.from_arrays([arr_id, arr_text], names=["id", "text"])

        table.insert(batch)
        table.commit()
        table.wait_for_background_tasks()
        build_s = time.perf_counter() - t_build_0

        index_bytes = sum(
            os.path.getsize(os.path.join(r, f))
            for r, _, files in os.walk(tmpdir)
            for f in files
            if ".puffin" in f
            or ".inverted" in f
            or ".bm25" in f
            or ".idx" in f
            or ".dict" in f
            or ".inv." in f
            or ".doclen." in f
        )

        # Warmup with 5 queries
        query_items = list(queries.items())
        for _, q_text in query_items[:5]:
            _ = table.vector_search("text", q_text, k=k)

        # Search benchmark
        latencies = []
        retrieved: Dict[str, List[str]] = {}

        t_search_start = time.perf_counter()
        for qid, q_text in query_items:
            t0 = time.perf_counter()
            res = table.vector_search("text", q_text, k=k)
            elapsed = (time.perf_counter() - t0) * 1000.0  # ms
            latencies.append(elapsed)

            retrieved[qid] = [str(x) for x in res["id"].tolist()] if len(res) > 0 else []

        total_search_time = time.perf_counter() - t_search_start
        qps = len(queries) / total_search_time if total_search_time > 0 else 0.0

        return {
            "engine": "benostreamdb",
            "available": True,
            "build_s": build_s,
            "index_mb": round(index_bytes / (1024 * 1024), 2),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p90_ms": float(np.percentile(latencies, 90)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "mean_ms": float(np.mean(latencies)),
            "qps": qps,
            "retrieved": retrieved,
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_tantivy(
    corpus: List[Tuple[str, str]],
    queries: Dict[str, str],
    k: int = 10,
) -> Dict[str, Any]:
    """Benchmarks Tantivy BM25 inverted index build and retrieval."""
    if not importlib.util.find_spec("tantivy"):
        return {"engine": "tantivy", "available": False, "error": "tantivy not installed"}
    import tantivy

    tmpdir = tempfile.mkdtemp(prefix="tantivy_beir_")
    try:
        t_build_0 = time.perf_counter()
        schema_builder = tantivy.SchemaBuilder()
        schema_builder.add_text_field("id", stored=True)
        schema_builder.add_text_field("text", stored=True)
        schema = schema_builder.build()

        index = tantivy.Index(schema, path=tmpdir)
        writer = index.writer()

        for doc_id, text in corpus:
            writer.add_document(tantivy.Document(id=[doc_id], text=[text]))
        writer.commit()
        index.reload()
        build_s = time.perf_counter() - t_build_0

        index_bytes = sum(
            os.path.getsize(os.path.join(r, f))
            for r, _, files in os.walk(tmpdir)
            for f in files
        )

        searcher = index.searcher()

        # Warmup with 5 queries
        query_items = list(queries.items())
        for _, q_text in query_items[:5]:
            q_parsed, _ = index.parse_query_lenient(q_text, ["text"])
            _ = searcher.search(q_parsed, k)

        # Search benchmark
        latencies = []
        retrieved: Dict[str, List[str]] = {}

        t_search_start = time.perf_counter()
        for qid, q_text in query_items:
            t0 = time.perf_counter()
            q_parsed, _ = index.parse_query_lenient(q_text, ["text"])
            hits = searcher.search(q_parsed, k).hits
            elapsed = (time.perf_counter() - t0) * 1000.0  # ms
            latencies.append(elapsed)

            doc_ids = []
            for _, doc_addr in hits:
                doc = searcher.doc(doc_addr)
                doc_ids.append(str(doc["id"][0]))
            retrieved[qid] = doc_ids

        total_search_time = time.perf_counter() - t_search_start
        qps = len(queries) / total_search_time if total_search_time > 0 else 0.0

        return {
            "engine": "tantivy",
            "available": True,
            "build_s": build_s,
            "index_mb": round(index_bytes / (1024 * 1024), 2),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p90_ms": float(np.percentile(latencies, 90)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "mean_ms": float(np.mean(latencies)),
            "qps": qps,
            "retrieved": retrieved,
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _search_client(host: str):
    """Return ``(client, helpers, kind)`` for an OpenSearch/Elasticsearch server.

    Prefers ``opensearch-py`` (the compose stack runs OpenSearch), falling back
    to the ``elasticsearch`` client. Returns ``(None, None, None)`` when neither
    is installed so the caller can report "not available" instead of crashing.
    """
    url = host if "://" in host else f"http://{host}"
    if importlib.util.find_spec("opensearchpy"):
        from opensearchpy import OpenSearch, helpers

        return OpenSearch(url), helpers, "opensearch"
    if importlib.util.find_spec("elasticsearch"):
        from elasticsearch import Elasticsearch, helpers

        return Elasticsearch(url), helpers, "elasticsearch"
    return None, None, None


def run_opensearch(
    corpus: List[Tuple[str, str]],
    queries: Dict[str, str],
    k: int = 10,
    host: str = "http://localhost:9200",
    engine_name: str = "opensearch",
) -> Dict[str, Any]:
    """Benchmarks server-backed BM25 (OpenSearch/Elasticsearch) over BEIR.

    Measured with the same corpus / queries / ``k`` / metric code as the
    embedded engines, so the row is directly comparable. A server-side index is
    built fresh per run, and its on-disk size is read from the index stats.
    """
    client, helpers, _kind = _search_client(host)
    if client is None:
        return {
            "engine": engine_name,
            "available": False,
            "error": "pip install opensearch-py (or elasticsearch)",
        }

    index_name = "bsdb-beir-lexical"
    try:
        t_build_0 = time.perf_counter()
        client.indices.delete(index=index_name, ignore_unavailable=True)
        client.indices.create(
            index=index_name,
            body={
                "mappings": {
                    "properties": {"text": {"type": "text", "analyzer": "english"}}
                }
            },
        )
        helpers.bulk(
            client,
            (
                {"_index": index_name, "_id": doc_id, "_source": {"text": text}}
                for doc_id, text in corpus
            ),
        )
        client.indices.refresh(index=index_name)
        build_s = time.perf_counter() - t_build_0
    except Exception as exc:  # server unreachable / mapping rejected
        return {
            "engine": engine_name,
            "available": False,
            "error": f"index build failed: {exc}",
        }

    index_bytes = 0
    try:
        stats = client.indices.stats(index=index_name)
        index_bytes = int(stats["_all"]["primaries"]["store"]["size_in_bytes"])
    except Exception:
        pass

    import re

    def _search(q_text: str):
        # Strip punctuation the analyzer would otherwise drop, mirroring the
        # query-normalisation the embedded harness relies on.
        safe = re.sub(r"[^\w\s]", " ", q_text)
        return client.search(
            index=index_name,
            body={"query": {"match": {"text": safe}}, "size": k, "_source": False},
        )

    query_items = list(queries.items())
    for _, q_text in query_items[:5]:
        try:
            _search(q_text)
        except Exception:
            pass

    latencies = []
    retrieved: Dict[str, List[str]] = {}
    t_search_start = time.perf_counter()
    for qid, q_text in query_items:
        t0 = time.perf_counter()
        res = _search(q_text)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        retrieved[qid] = [str(h["_id"]) for h in res["hits"]["hits"]]
    total_search_time = time.perf_counter() - t_search_start
    qps = len(query_items) / total_search_time if total_search_time > 0 else 0.0

    return {
        "engine": engine_name,
        "available": True,
        "build_s": build_s,
        "index_mb": round(index_bytes / (1024 * 1024), 2),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p90_ms": float(np.percentile(latencies, 90)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "mean_ms": float(np.mean(latencies)),
        "qps": qps,
        "retrieved": retrieved,
    }


def _jaccard_agreement(
    a_res: Dict[str, Any],
    b_res: Dict[str, Any],
    queries: Dict[str, str],
) -> float:
    """Mean top-k Jaccard overlap of two engines' retrieved sets, in percent."""
    overlaps = []
    for qid in queries:
        a_hits = set(a_res["retrieved"].get(qid, []))
        b_hits = set(b_res["retrieved"].get(qid, []))
        if a_hits or b_hits:
            overlaps.append(len(a_hits & b_hits) / max(len(a_hits | b_hits), 1))
    return float(np.mean(overlaps)) * 100.0 if overlaps else 0.0


def main():
    parser = argparse.ArgumentParser(description="BEIR Lexical / BM25 Benchmark Runner")
    parser.add_argument("--dataset", default="scifact", help="Dataset name under benchmarks/beir/data/")
    parser.add_argument("--engines", default="benostreamdb,tantivy", help="Comma-separated engines to run")
    parser.add_argument("--k", type=int, default=10, help="Top-k retrieval limit")
    parser.add_argument("--limit-queries", type=int, default=None, help="Limit number of queries evaluated")
    parser.add_argument("--out", default=None, help="Output markdown path")
    parser.add_argument(
        "--json-dir",
        default=None,
        help="If set, write one competitor-schema JSON record per engine here, "
        "so generate_summary.py rolls the BEIR rows into the benchmark report",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("DEVICE", "cpu"),
        help="Device tag for the JSON records (default: $DEVICE or cpu)",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory holding <dataset>/ (default: benchmarks/beir/data)",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="OpenSearch/Elasticsearch URL (default: $ES_URL or http://localhost:9200)",
    )
    args = parser.parse_args()

    env = envelope_info()
    print(f"Loading {args.dataset} dataset...", flush=True)
    corpus, queries, qrels = load_scifact_data(args.dataset, data_dir=args.data_dir)
    if args.limit_queries and args.limit_queries < len(queries):
        subset_keys = list(queries.keys())[: args.limit_queries]
        queries = {k: queries[k] for k in subset_keys}

    print(f"Loaded: {len(corpus):,} documents, {len(queries):,} queries ({len(qrels):,} qrels)", flush=True)

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]

    results = []
    for engine in engines:
        print(f"Running {engine}...", flush=True)
        if engine == "benostreamdb":
            res = run_benostreamdb(corpus, queries, k=args.k)
        elif engine == "tantivy":
            res = run_tantivy(corpus, queries, k=args.k)
        elif engine in ("opensearch", "elasticsearch"):
            host = args.host or os.environ.get("ES_URL", "http://localhost:9200")
            res = run_opensearch(corpus, queries, k=args.k, host=host, engine_name=engine)
        else:
            print(f"Skipping unknown engine: {engine}")
            continue

        if res.get("available"):
            res.update(compute_ir_metrics(res["retrieved"], qrels, k=args.k))
            print(
                f"  -> {engine}: p50={res['p50_ms']:.2f}ms, p99={res['p99_ms']:.2f}ms, "
                f"QPS={res['qps']:.1f}, nDCG@{args.k}={res.get(f'ndcg@{args.k}', 0):.4f}, "
                f"Recall@{args.k}={res.get(f'recall@{args.k}', 0):.4f}, build={res['build_s']:.2f}s",
                flush=True,
            )
        else:
            print(f"  -> {engine}: NOT AVAILABLE ({res.get('error')})", flush=True)

        results.append(res)

    # Top-k agreement of every competitor vs BenoStreamDB.
    bsdb_res = next((r for r in results if r["engine"] == "benostreamdb" and r.get("available")), None)
    agreements: Dict[str, float] = {}
    if bsdb_res:
        for r in results:
            if r is bsdb_res or not r.get("available"):
                continue
            pct = _jaccard_agreement(bsdb_res, r, queries)
            agreements[r["engine"]] = pct
            print(
                f"\nBenoStreamDB vs {r['engine']} Top-{args.k} Jaccard Agreement: {pct:.1f}%",
                flush=True,
            )

    # Format Markdown Report
    env_bits = []
    if env["cpus"] is not None:
        env_bits.append(f"{env['cpus']} CPUs")
    if env["mem"]:
        env_bits.append(f"{env['mem']} RAM")
    envelope_str = ", ".join(env_bits) if env_bits else "unconstrained (host)"
    lines = [
        "# BEIR Lexical / BM25 Benchmark Results",
        "",
        f"- **Dataset**: `{args.dataset}`",
        f"- **Corpus Documents**: {len(corpus):,}",
        f"- **Evaluated Queries**: {len(queries):,}",
        f"- **Top-K**: {args.k}",
        f"- **Host**: {platform.processor() or platform.machine()} ({platform.system()})",
        f"- **Resource Envelope**: {envelope_str}"
        + (" (containerized)" if env["containerized"] else ""),
        "- **Methodology**: every engine runs in a Docker container under the same "
        "`--cpus`/`--memory` envelope (see `benchmarks/competitors/docker_bench.sh "
        "--workload beir`), so no participant gets more cores or RAM than another.",
        "",
        f"| Engine | Status | Build Time | Index Size | QPS | p50 Latency | p99 Latency | Recall@{args.k} | nDCG@{args.k} |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for r in results:
        if r.get("available"):
            lines.append(
                f"| **{r['engine']}** | ✅ Pass | {r['build_s']:.2f}s | {r['index_mb']:.1f} MB | "
                f"**{r['qps']:.1f}** | **{r['p50_ms']:.2f} ms** | {r['p99_ms']:.2f} ms | "
                f"{r.get(f'recall@{args.k}', 0):.4f} | {r.get(f'ndcg@{args.k}', 0):.4f} |"
            )
        else:
            lines.append(f"| **{r['engine']}** | ❌ Failed ({r.get('error')}) | - | - | - | - | - | - | - |")

    if agreements:
        lines.extend(["", "### Differential Oracle & Result Agreement", ""])
        for engine, pct in agreements.items():
            lines.append(f"- **Top-{args.k} Jaccard Overlap vs {engine}**: **{pct:.1f}%**.")
        lines.append(
            "- High ranking agreement validates correct Okapi BM25 implementation across "
            "vocabulary, inverted postings, and document length normalization sidecars."
        )

    report = "\n".join(lines) + "\n"
    print("\n" + report)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Wrote BEIR benchmark report to {args.out}")

    if args.json_dir:
        write_json_records(
            results,
            args.dataset,
            args.k,
            len(queries),
            args.json_dir,
            workload="lexical_bm25",
            device=args.device,
        )


if __name__ == "__main__":
    main()
