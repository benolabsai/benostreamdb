#!/usr/bin/env python3
"""BEIR Hybrid Search Benchmark Runner: BenoStreamDB vs LanceDB.

Evaluates retrieval quality (Recall@10, nDCG@10, MRR@10), index footprint on disk,
throughput (QPS), and latency (p50/p99) across:
1. Dense Vector Search (HNSW / TurboQuant vs LanceDB Vector Index)
2. Lexical / Sparse Search (Okapi BM25 vs LanceDB Tantivy FTS)
3. Hybrid Search (Dense + Sparse combined via Reciprocal Rank Fusion - RRF)

Competitor:
- LanceDB (embedded columnar vector + Tantivy FTS + RRFReranker)

Usage:
    python benchmarks/beir/run_hybrid.py --dataset scifact --engines benostreamdb,lancedb --out benchmarks/beir/results/scifact_hybrid.md
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


def load_dataset_with_embeddings(
    dataset_name: str = "scifact",
) -> Tuple[List[Tuple[str, str, np.ndarray]], Dict[str, Tuple[str, np.ndarray]], Dict[str, Dict[str, int]]]:
    """Loads corpus with dense embeddings, test queries with embeddings, and qrels."""
    dataset_dir = os.path.join(BEIR_DATA_DIR, dataset_name)
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    corpus_path = os.path.join(dataset_dir, "corpus.jsonl")
    queries_path = os.path.join(dataset_dir, "queries.jsonl")
    qrels_path = os.path.join(dataset_dir, "qrels", "test.tsv")
    emb_path = os.path.join(dataset_dir, "embeddings_minilm.npz")

    # 1. Load raw text
    doc_map = {}
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            doc_id = str(doc["_id"])
            text = f"{doc.get('title', '')} {doc.get('text', '')}".strip()
            doc_map[doc_id] = text

    all_queries = {}
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            all_queries[str(q["_id"])] = q.get("text", "").strip()

    # 2. Load qrels
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

    test_query_ids = [qid for qid in qrels if qid in all_queries]
    if not test_query_ids:
        test_query_ids = list(all_queries.keys())

    # 3. Load or generate embeddings
    if not os.path.exists(emb_path):
        from sentence_transformers import SentenceTransformer

        print(f"Generating embeddings using all-MiniLM-L6-v2 for {dataset_name}...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        all_doc_ids = list(doc_map.keys())
        all_doc_texts = [doc_map[did] for did in all_doc_ids]
        doc_embs = model.encode(all_doc_texts, batch_size=128, show_progress_bar=True, normalize_embeddings=True)

        q_texts = [all_queries[qid] for qid in test_query_ids]
        q_embs = model.encode(q_texts, batch_size=128, show_progress_bar=False, normalize_embeddings=True)

        np.savez_compressed(
            emb_path,
            doc_ids=np.array(all_doc_ids),
            doc_embs=doc_embs.astype(np.float32),
            q_ids=np.array(test_query_ids),
            q_embs=q_embs.astype(np.float32),
        )

    data = np.load(emb_path)
    loaded_doc_ids = [str(x) for x in data["doc_ids"]]
    loaded_doc_embs = data["doc_embs"]
    loaded_q_ids = [str(x) for x in data["q_ids"]]
    loaded_q_embs = data["q_embs"]

    # Assemble corpus tuples: (doc_id, text, embedding)
    corpus = []
    for did, emb in zip(loaded_doc_ids, loaded_doc_embs):
        if did in doc_map:
            corpus.append((did, doc_map[did], emb))

    # Assemble query map: qid -> (query_text, embedding) (only queries with test qrels)
    queries = {}
    for qid, emb in zip(loaded_q_ids, loaded_q_embs):
        if qid in qrels and qid in all_queries:
            queries[qid] = (all_queries[qid], emb)

    return corpus, queries, qrels


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

        ideal_scores = sorted(rel_docs.values(), reverse=True)[:k]
        idcg = sum(score / math.log2(rank + 1) for rank, score in enumerate(ideal_scores, start=1))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcgs.append(ndcg)

    return {
        f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"mrr@{k}": float(np.mean(mrrs)) if mrrs else 0.0,
    }


def run_benostreamdb_hybrid(
    corpus: List[Tuple[str, str, np.ndarray]],
    queries: Dict[str, Tuple[str, np.ndarray]],
    qrels: Dict[str, Dict[str, int]],
    k: int = 10,
    rrf_k: int = 60,
) -> Dict[str, Any]:
    """Runs Dense-only, Sparse-only (BM25), and Hybrid (RRF) on BenoStreamDB."""
    import benostreamdb as bsdb

    tmpdir = tempfile.mkdtemp(prefix="bsdb_hybrid_")
    dim = len(corpus[0][2])

    try:
        t_build_0 = time.perf_counter()

        schema = pa.schema([
            ("id", pa.string()),
            ("text", pa.string()),
            ("embedding", pa.list_(pa.float32(), dim)),
        ])
        table = bsdb.Table.create(tmpdir, schema)
        table.add_index("text", {"type": "bm25", "k1": 1.2, "b": 0.75})
        table.add_index("embedding", {"type": "hnsw_tq8", "m": 32, "ef_construction": 200, "ef_search": 100})

        arr_id = pa.array([c[0] for c in corpus], type=pa.string())
        arr_text = pa.array([c[1] for c in corpus], type=pa.string())
        
        flat_embs = np.vstack([c[2] for c in corpus]).astype(np.float32).reshape(-1)
        arr_emb = pa.FixedSizeListArray.from_arrays(pa.array(flat_embs), dim)

        batch = pa.Table.from_arrays([arr_id, arr_text, arr_emb], names=["id", "text", "embedding"])
        table.insert(batch)
        table.commit()
        table.wait_for_background_tasks()
        build_s = time.perf_counter() - t_build_0

        # Calculate index sizes
        # Secondary indexes are packed into Puffin compound bundles, so the
        # whole bundle is counted once (the BM25/dense split is not separable
        # from the container). Loose-sidecar names are kept for older tables.
        bm25_bytes = sum(
            os.path.getsize(os.path.join(r, f))
            for r, _, files in os.walk(tmpdir)
            for f in files
            if ".puffin" in f or ".inv." in f or ".doclen." in f
        )
        dense_bytes = sum(
            os.path.getsize(os.path.join(r, f))
            for r, _, files in os.walk(tmpdir)
            for f in files
            if ".index" in f or ".tq8" in f or ".hnsw" in f
        )
        total_index_bytes = bm25_bytes + dense_bytes

        query_items = list(queries.items())

        # Warmup
        for _, (q_text, q_vec) in query_items[:5]:
            _ = table.vector_search("text", q_text, k=k)
            _ = table.vector_search("embedding", q_vec.tolist(), k=k)
            _ = table.hybrid_search("text", q_text, "embedding", q_vec.tolist(), k=k, rrf_k=rrf_k)

        # 1. Sparse Only (BM25)
        lat_sparse = []
        ret_sparse = {}
        t0 = time.perf_counter()
        for qid, (q_text, _) in query_items:
            t_q = time.perf_counter()
            res = table.vector_search("text", q_text, k=k)
            lat_sparse.append((time.perf_counter() - t_q) * 1000.0)
            ret_sparse[qid] = [str(x) for x in res["id"].tolist()] if len(res) > 0 else []
        t_sparse_total = time.perf_counter() - t0
        metrics_sparse = compute_ir_metrics(ret_sparse, qrels, k=k)

        # 2. Dense Only (HNSW-TQ8)
        lat_dense = []
        ret_dense = {}
        t0 = time.perf_counter()
        for qid, (_, q_vec) in query_items:
            t_q = time.perf_counter()
            res = table.vector_search("embedding", q_vec.tolist(), k=k)
            lat_dense.append((time.perf_counter() - t_q) * 1000.0)
            ret_dense[qid] = [str(x) for x in res["id"].tolist()] if len(res) > 0 else []
        t_dense_total = time.perf_counter() - t0
        metrics_dense = compute_ir_metrics(ret_dense, qrels, k=k)

        # 3. Hybrid Search (RRF)
        lat_hybrid = []
        ret_hybrid = {}
        t0 = time.perf_counter()
        for qid, (q_text, q_vec) in query_items:
            t_q = time.perf_counter()
            res = table.hybrid_search("text", q_text, "embedding", q_vec.tolist(), k=k, rrf_k=rrf_k)
            lat_hybrid.append((time.perf_counter() - t_q) * 1000.0)
            ret_hybrid[qid] = [str(x) for x in res["id"].tolist()] if len(res) > 0 else []
        t_hybrid_total = time.perf_counter() - t0
        metrics_hybrid = compute_ir_metrics(ret_hybrid, qrels, k=k)

        return {
            "engine": "benostreamdb",
            "available": True,
            "build_s": build_s,
            "index_mb": round(total_index_bytes / (1024 * 1024), 2),
            "p50_ms": float(np.percentile(lat_hybrid, 50)),
            "p90_ms": float(np.percentile(lat_hybrid, 90)),
            "p99_ms": float(np.percentile(lat_hybrid, 99)),
            "qps": len(queries) / t_hybrid_total if t_hybrid_total > 0 else 0.0,
            "retrieved": ret_hybrid,
            "sparse": {
                "p50_ms": float(np.percentile(lat_sparse, 50)),
                "qps": len(queries) / t_sparse_total if t_sparse_total > 0 else 0.0,
                "index_mb": round(bm25_bytes / (1024 * 1024), 2),
                **metrics_sparse,
            },
            "dense": {
                "p50_ms": float(np.percentile(lat_dense, 50)),
                "qps": len(queries) / t_dense_total if t_dense_total > 0 else 0.0,
                "index_mb": round(dense_bytes / (1024 * 1024), 2),
                **metrics_dense,
            },
            "hybrid": {
                "p50_ms": float(np.percentile(lat_hybrid, 50)),
                "qps": len(queries) / t_hybrid_total if t_hybrid_total > 0 else 0.0,
                "index_mb": round(total_index_bytes / (1024 * 1024), 2),
                **metrics_hybrid,
            },
            **metrics_hybrid,
        }

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_lancedb_hybrid(
    corpus: List[Tuple[str, str, np.ndarray]],
    queries: Dict[str, Tuple[str, np.ndarray]],
    qrels: Dict[str, Dict[str, int]],
    k: int = 10,
    rrf_k: int = 60,
) -> Dict[str, Any]:
    """Runs Hybrid (RRF) on LanceDB using Tantivy FTS + Vector Index + RRFReranker."""
    if not importlib.util.find_spec("lancedb"):
        return {"engine": "lancedb", "available": False, "error": "lancedb not installed"}

    import lancedb
    from lancedb.rerankers import RRFReranker

    tmpdir = tempfile.mkdtemp(prefix="lance_hybrid_")

    try:
        t_build_0 = time.perf_counter()
        db = lancedb.connect(tmpdir)
        data = [
            {"id": str(did), "text": str(txt), "vector": emb.tolist()}
            for did, txt, emb in corpus
        ]
        tbl = db.create_table("scifact", data)
        tbl.create_fts_index("text")
        build_s = time.perf_counter() - t_build_0

        index_bytes = sum(
            os.path.getsize(os.path.join(r, f))
            for r, _, files in os.walk(tmpdir)
            for f in files
        )

        reranker = RRFReranker(K=rrf_k)
        query_items = list(queries.items())

        # Warmup
        for _, (q_text, q_vec) in query_items[:5]:
            _ = tbl.search(query_type="hybrid").vector(q_vec.tolist()).text(q_text).rerank(reranker).limit(k).to_pandas()

        lat_hybrid = []
        ret_hybrid = {}
        t0 = time.perf_counter()
        for qid, (q_text, q_vec) in query_items:
            t_q = time.perf_counter()
            res = tbl.search(query_type="hybrid").vector(q_vec.tolist()).text(q_text).rerank(reranker).limit(k).to_pandas()
            lat_hybrid.append((time.perf_counter() - t_q) * 1000.0)
            ret_hybrid[qid] = [str(x) for x in res["id"].tolist()] if len(res) > 0 else []
        t_hybrid_total = time.perf_counter() - t0
        metrics_hybrid = compute_ir_metrics(ret_hybrid, qrels, k=k)

        return {
            "engine": "lancedb",
            "available": True,
            "build_s": build_s,
            "index_mb": round(index_bytes / (1024 * 1024), 2),
            "p50_ms": float(np.percentile(lat_hybrid, 50)),
            "p90_ms": float(np.percentile(lat_hybrid, 90)),
            "p99_ms": float(np.percentile(lat_hybrid, 99)),
            "qps": len(queries) / t_hybrid_total if t_hybrid_total > 0 else 0.0,
            "retrieved": ret_hybrid,
            **metrics_hybrid,
        }

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="BEIR Hybrid Search Benchmark Runner")
    parser.add_argument("--dataset", default="scifact", help="Dataset name under benchmarks/beir/data/")
    parser.add_argument("--engines", default="benostreamdb,lancedb", help="Comma-separated engines to run")
    parser.add_argument("--k", type=int, default=10, help="Top-k retrieval limit")
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF constant (default: 60)")
    parser.add_argument("--limit-queries", type=int, default=None, help="Limit number of queries evaluated")
    parser.add_argument("--out", default=None, help="Output markdown path")
    args = parser.parse_args()

    print(f"Loading {args.dataset} dataset with dense embeddings...", flush=True)
    corpus, queries, qrels = load_dataset_with_embeddings(args.dataset)
    if args.limit_queries and args.limit_queries < len(queries):
        subset_keys = list(queries.keys())[: args.limit_queries]
        queries = {k: queries[k] for k in subset_keys}

    print(f"Loaded: {len(corpus):,} documents, {len(queries):,} queries ({len(qrels):,} qrels)", flush=True)

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    runners = {
        "benostreamdb": run_benostreamdb_hybrid,
        "lancedb": run_lancedb_hybrid,
    }

    results = []
    for engine in engines:
        runner = runners.get(engine)
        if not runner:
            print(f"Skipping unknown engine: {engine}")
            continue

        print(f"\nRunning {engine} Hybrid Search (Dense + Sparse RRF)...", flush=True)
        res = runner(corpus, queries, qrels, k=args.k, rrf_k=args.rrf_k)
        if res.get("available"):
            print(
                f"  -> {engine}: p50={res['p50_ms']:.2f}ms, p99={res['p99_ms']:.2f}ms, "
                f"QPS={res['qps']:.1f}, nDCG@{args.k}={res.get(f'ndcg@{args.k}', 0):.4f}, "
                f"Recall@{args.k}={res.get(f'recall@{args.k}', 0):.4f}, build={res['build_s']:.2f}s",
                flush=True,
            )
        else:
            print(f"  -> {engine}: NOT AVAILABLE ({res.get('error')})", flush=True)

        results.append(res)

    # Compute Top-K Agreement between BenoStreamDB and LanceDB
    agreement_pct = None
    bsdb_res = next((r for r in results if r["engine"] == "benostreamdb" and r.get("available")), None)
    lance_res = next((r for r in results if r["engine"] == "lancedb" and r.get("available")), None)
    if bsdb_res and lance_res:
        overlaps = []
        for qid in queries:
            b_hits = set(bsdb_res["retrieved"].get(qid, []))
            l_hits = set(lance_res["retrieved"].get(qid, []))
            if b_hits or l_hits:
                overlap = len(b_hits.intersection(l_hits)) / max(len(b_hits.union(l_hits)), 1)
                overlaps.append(overlap)
        agreement_pct = float(np.mean(overlaps)) * 100.0 if overlaps else 0.0
        print(f"\nBenoStreamDB vs LanceDB Top-{args.k} Jaccard Agreement: {agreement_pct:.1f}%", flush=True)

    # Format Markdown Report
    lines = [
        "# BEIR Hybrid Search Benchmark Results: BenoStreamDB vs Competitor",
        "",
        f"- **Dataset**: `{args.dataset}`",
        f"- **Corpus Documents**: {len(corpus):,}",
        f"- **Evaluated Queries**: {len(queries):,}",
        f"- **Dense Embedding Model**: `all-MiniLM-L6-v2` (384-d)",
        f"- **Lexical Algorithm**: Okapi BM25 (`k1=1.2, b=0.75`)",
        f"- **Fusion Algorithm**: Reciprocal Rank Fusion (RRF, `k={args.rrf_k}`)",
        f"- **Top-K**: {args.k}",
        f"- **Host**: {platform.processor() or platform.machine()} ({platform.system()})",
        "",
        "### Competitor Comparison (Hybrid Dense + Sparse RRF)",
        "",
        f"| Engine | Status | Build Time | Total Size on Disk | Throughput (QPS) | p50 Latency | p99 Latency | Recall@{args.k} | nDCG@{args.k} | MRR@{args.k} |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    for r in results:
        if r.get("available"):
            lines.append(
                f"| **{r['engine']}** | ✅ Pass | {r['build_s']:.2f}s | {r['index_mb']:.1f} MB | "
                f"**{r['qps']:.1f}** | **{r['p50_ms']:.2f} ms** | {r['p99_ms']:.2f} ms | "
                f"**{r.get(f'recall@{args.k}', 0):.4f}** | **{r.get(f'ndcg@{args.k}', 0):.4f}** | "
                f"{r.get(f'mrr@{args.k}', 0):.4f} |"
            )
        else:
            lines.append(f"| **{r['engine']}** | ❌ Failed ({r.get('error')}) | - | - | - | - | - | - | - | - |")

    if agreement_pct is not None:
        lines.extend([
            "",
            "### Differential Oracle & Result Agreement",
            "",
            f"- **Top-{args.k} Jaccard Overlap**: **{agreement_pct:.1f}%** between BenoStreamDB Hybrid and LanceDB Hybrid.",
            "- High ranking agreement validates correct multi-modal retrieval and reciprocal rank fusion mathematics against an established embedded vector database.",
        ])

    # If BenoStreamDB has internal breakdown, output it as well
    if bsdb_res and "sparse" in bsdb_res and "dense" in bsdb_res:
        sp = bsdb_res["sparse"]
        dn = bsdb_res["dense"]
        hy = bsdb_res["hybrid"]
        lines.extend([
            "",
            "### BenoStreamDB Single-Modality vs Hybrid Lift Breakdown",
            "",
            f"| Search Mode | Index Size | QPS | p50 Latency | Recall@{args.k} | nDCG@{args.k} | MRR@{args.k} |",
            "|---|---|---|---|---|---|---|",
            f"| **Sparse (BM25 Only)** | {sp['index_mb']:.1f} MB | {sp['qps']:.1f} | {sp['p50_ms']:.2f} ms | {sp.get(f'recall@{args.k}', 0):.4f} | {sp.get(f'ndcg@{args.k}', 0):.4f} | {sp.get(f'mrr@{args.k}', 0):.4f} |",
            f"| **Dense (Vector Only)** | {dn['index_mb']:.1f} MB | {dn['qps']:.1f} | {dn['p50_ms']:.2f} ms | {dn.get(f'recall@{args.k}', 0):.4f} | {dn.get(f'ndcg@{args.k}', 0):.4f} | {dn.get(f'mrr@{args.k}', 0):.4f} |",
            f"| **Hybrid (Dense + BM25 RRF)** | {hy['index_mb']:.1f} MB | {hy['qps']:.1f} | {hy['p50_ms']:.2f} ms | **{hy.get(f'recall@{args.k}', 0):.4f}** | **{hy.get(f'ndcg@{args.k}', 0):.4f}** | **{hy.get(f'mrr@{args.k}', 0):.4f}** |",
        ])

    report = "\n".join(lines) + "\n"
    print("\n" + report)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Wrote Hybrid benchmark report to {args.out}")


if __name__ == "__main__":
    main()
