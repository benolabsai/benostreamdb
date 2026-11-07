#!/usr/bin/env python3
"""Focused MuSiQue parity benchmark.

Runs the four engine-native strategies in the Zilliz/HippoRAG comparison:

    vector_only, sql_recursive_join, topological_ppr, two_stage_{reranker}

The full harness (nvembed, multi-GPU encode, PPR-rerank, ensemble rerankers,
research presets) lives in run_musique_graphrag.py. This wrapper keeps the
parity path small and stable; it reuses the tested functions from that module.

`--reranker bge` is the headline configuration: it reaches 0.5943 Recall@5 on
the 1,000-question manifest, clearing Zilliz's Naive RAG baseline (0.5803) with
no hosted LLM in the retrieval loop. MiniLM is the faster default (0.5739).

Examples:

    # Exact 1,000-question Zilliz manifest (headline)
    python3 benchmarks/run_musique_parity.py --mode f32 --reranker bge --rebuild \
        --corpus-json benchmarks/musique_corpus.json \
        --triplets-json benchmarks/hippo_openie.json \
        --sample-manifest benchmarks/zilliz_manifest.json

    # Full-corpus parity (all questions with gold in the 11,656-passage corpus)
    python3 benchmarks/run_musique_parity.py --mode f32 --reranker bge --rebuild \
        --corpus-json benchmarks/musique_corpus.json \
        --triplets-json benchmarks/hippo_openie.json

Host runs need a host-writable --scratch-dir (benchmarks/scratch_musique is
root-owned by the Docker container); pass --scratch-dir benchmarks/scratch_musique_parity.
"""

import argparse
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_musique_graphrag as m  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=m.DEFAULT_DATA_DIR)
    parser.add_argument("--scratch-dir", default=m.DEFAULT_SCRATCH_DIR)
    parser.add_argument("--corpus-json", default="",
                        help="External deduplicated corpus (JSON list of {title, text}).")
    parser.add_argument("--triplets-json", default="",
                        help="HippoRAG OpenIE file (docs[].extracted_entities).")
    parser.add_argument("--sample-manifest", default="",
                        help="Zilliz/HippoRAG manifest.json (samples.<dataset>).")
    parser.add_argument("--manifest-dataset", default="musique")
    parser.add_argument("--mode", choices=["f32", "tq8"], default="f32",
                        help="f32 for quality, tq8 for latency/throughput.")
    parser.add_argument("--reranker", choices=["minilm", "bge", "both", "ensemble"],
                        default="minilm",
                        help="Stage-2 reranker: minilm (fast), bge (stronger), "
                             "both/ensemble (all three variants).")
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidates", type=int, default=24,
                        help="Stage-2 candidate pool cap (default: 24).")
    parser.add_argument("--expand-hops", type=int, default=2,
                        help="Graph expansion hops for the two-stage pool (default: 2).")
    parser.add_argument("--ef-search", type=int, default=m.DEFAULT_EF_SEARCH,
                        help=f"HNSW ef_search (default: {m.DEFAULT_EF_SEARCH}).")
    parser.add_argument("--ppr-rerank", action="store_true",
                        help="Also run the PPR-proposes/cross-encoder-reranks strategies.")
    parser.add_argument("--out-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results", "musique_parity"))
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--hardware", choices=["d8lds", "d8s", "e8ds", "local", "auto"],
                        default="d8s")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--cache-gb", type=int, default=16)
    args = parser.parse_args()

    # apply_hardware only needs the envelope fields; the parity path fixes the
    # embedder and reranker so the rest of the full harness's knobs are absent.
    hw = SimpleNamespace(
        hardware=args.hardware,
        strict_hardware=False,
        threads=args.threads,
        cache_gb=args.cache_gb,
        preload_gb=0.0,
        embed_model="bge-large",
        embed_device="auto",
    )
    m.apply_hardware(hw)
    m.configure_rerank_device("auto")
    backend = m.configure_embedding("bge-large", device="auto", dtype="auto", max_seq_length=0)
    print(f"[parity] Retrieval embedder: {backend.describe()}", flush=True)

    import benostreamdb as bsdb

    dataset_path = m.ensure_dataset(args.data_dir)
    records = m.load_musique_records(dataset_path, args.max_questions)
    print(f"[parity] Loaded {len(records):,} MuSiQue questions.", flush=True)

    if args.sample_manifest:
        with open(args.sample_manifest, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        samples = manifest.get("samples", {}).get(args.manifest_dataset, [])
        by_id = {str(r.get("id")): r for r in records}
        selected = []
        for row in samples:
            rec = by_id.get(str(row.get("id")))
            if rec is not None and (not row.get("query") or rec.get("question") == row["query"]):
                selected.append(rec)
        records = selected
        print(f"[parity] Manifest: {len(records):,}/{len(samples):,} questions matched.", flush=True)

    corpus_dir = os.path.join(args.scratch_dir, "musique_corpus")
    edge_dir = os.path.join(args.scratch_dir, "musique_edges")

    if args.rebuild or not os.path.exists(corpus_dir) or not os.path.exists(edge_dir):
        corpus_table, edge_table, unique_passages = m.build_tables(
            bsdb, args.scratch_dir, records, mode=args.mode, backend=backend,
            corpus_json=args.corpus_json, triplets_json=args.triplets_json,
        )
    else:
        print(f"[parity] Reusing tables at {args.scratch_dir}...", flush=True)
        corpus_table = bsdb.Table(corpus_dir)
        edge_table = bsdb.Table(edge_dir)
        unique_passages = {}
        tbl = corpus_table.to_arrow(columns=["id", "doc_key", "title"])
        for pid, pkey, ptitle in zip(
            tbl.column("id").to_pylist(),
            tbl.column("doc_key").to_pylist(),
            tbl.column("title").to_pylist(),
        ):
            unique_passages[pkey] = {"id": int(pid), "key": pkey, "title": ptitle}

    results = m.run_evaluation(
        corpus_table, edge_table, unique_passages, records,
        top_k=args.top_k,
        reranker=args.reranker,
        seed_k=2,
        candidates=args.candidates,
        expand_hops=args.expand_hops,
        ppr_seeds=5,
        ppr_hops=3,
        ppr_pool=30,
        ppr_max_degree=25,
        ppr_rerank=args.ppr_rerank,
        dedupe_title=False,
        ef_search=args.ef_search,
    )
    m.save_report(
        results, args.top_k, args.out_dir,
        corpus_meta=m.read_embedding_metadata(args.scratch_dir),
        graph_source=("HippoRAG OpenIE triplets" if args.triplets_json
                      else "proper-noun co-occurrence"),
    )


if __name__ == "__main__":
    main()
