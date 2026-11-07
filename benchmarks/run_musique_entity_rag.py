#!/usr/bin/env python3
"""Entity-level Vector Graph RAG benchmark for MuSiQue, running on BenoStreamDB.

Implements the three-collection architecture described in the "Vector Graph RAG"
design (entities, relations, passages) *inside BenoStreamDB*:

    1. Passages  - deduplicated MuSiQue corpus as a native Iceberg table with an
                   HNSW embedding index
    2. Entities  - deduplicated OpenIE entity names as a native Iceberg table with
                   an HNSW embedding index
    3. Relations - (subject, object) OpenIE triples as an Iceberg *edge* table
                   (source = entity, target = entity), CSR-indexed
    4. Mentions  - (entity -> passage) edge table mapping each entity to the
                   passages it was extracted from, CSR-indexed

Pipeline (per question), all retrieval steps executed by the engine (vector
search + DataFusion graph traversal), only the final cross-encoder rerank runs
in-process:

    Step 1  Seed retrieval      - vector search the *Entities* table
    Step 2  Subgraph expansion  - BFS the Relations edge table N hops
    Step 3  Passage collection  - follow Mentions edges to candidate passages
    Step 4  Reranking           - local cross-encoder over the candidate pool

Reference: https://dev.to/chen_zhang_bac430bc7f6b95/vector-graph-rag-multi-hop-rag-without-a-graph-database-3hgb

Examples:

    # Full corpus, entity-level RAG with BGE cross-encoder
    python3 benchmarks/run_musique_entity_rag.py --mode f32 --reranker bge --rebuild \
        --corpus-json benchmarks/musique_corpus.json \
        --triplets-json benchmarks/hippo_openie.json

    # Exact 1,000-question Zilliz manifest
    python3 benchmarks/run_musique_entity_rag.py --mode f32 --reranker bge --rebuild \
        --corpus-json benchmarks/musique_corpus.json \
        --triplets-json benchmarks/hippo_openie.json \
        --sample-manifest benchmarks/zilliz_manifest.json

Host runs need a host-writable --scratch-dir (benchmarks/scratch_musique is
root-owned by the Docker container); pass --scratch-dir benchmarks/scratch_entity_rag.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pyarrow as pa

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_musique_graphrag as m  # noqa: E402


GENERIC_STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "up", "about", "into", "over", "after", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "who", "what", "where", "when", "why", "how", "which", "this", "that",
    "these", "those", "it", "its", "as", "one", "two", "three", "first", "second",
    "film", "song", "album", "city", "state", "county", "man", "woman", "year",
    "time", "part", "world", "day", "american", "british", "series", "season",
    "group", "band", "team", "show", "house", "school", "station", "line",
    "river", "lake", "mountain", "north", "south", "east", "west", "new", "old",
    "national", "international", "general", "president", "united", "states",
    "french", "german", "italian", "japanese", "chinese", "mexican", "canadian"
}


def clean_entity(name) -> str:
    return str(name).strip().lower()


def is_keep_entity(name: str) -> bool:
    name = clean_entity(name)
    return len(name) >= 3 and name not in GENERIC_STOP_WORDS


def build_passages(corpus_json: str, records: list) -> dict:
    """Deduplicated passage corpus keyed by ``passage_key`` (insertion order).

    Mirrors ``run_musique_graphrag.build_tables`` so the passage-embedding cache
    built by that harness can be reused byte-for-byte: both iterate the corpus in
    the same order and assign the same integer ids.
    """
    unique_passages = {}
    if corpus_json:
        with open(corpus_json, "r", encoding="utf-8") as f:
            corpus_rows = json.load(f)
        for p in corpus_rows:
            p_key = m.passage_key(p["title"], p["text"])
            if p_key not in unique_passages:
                unique_passages[p_key] = {
                    "id": len(unique_passages),
                    "key": p_key,
                    "title": p["title"],
                    "text": p["text"],
                }
        print(f"[entity_rag] External corpus: {len(unique_passages):,} passages "
              f"from {corpus_json}", flush=True)
    else:
        for r in records:
            for p in r["paragraphs"]:
                p_key = m.passage_key(p["title"], p["paragraph_text"])
                if p_key not in unique_passages:
                    unique_passages[p_key] = {
                        "id": len(unique_passages),
                        "key": p_key,
                        "title": p["title"],
                        "text": p["paragraph_text"],
                    }
    return unique_passages


def build_entity_graph(triplets_json: str, unique_passages: dict):
    """Build the entity vocabulary + relation adjacency + entity->passage map.

    Returns ``(entity_keys, entity_id, entity_display, entity_to_passages, adj)``.
    ``adj`` maps canonical entity index -> set of canonical entity indices linked
    by any OpenIE triple (undirected, both directions present).
    """
    with open(triplets_json, "r", encoding="utf-8") as f:
        triplet_data = json.load(f)
    docs = triplet_data.get("docs", []) if isinstance(triplet_data, dict) else triplet_data

    key_to_pid = {info["key"]: info["id"] for info in unique_passages.values()}
    entity_to_passages = defaultdict(set)
    entity_display_by_key = {}
    relation_triples = []
    matched = 0

    for doc in docs:
        pid = key_to_pid.get(m.passage_key(doc.get("title", ""), doc.get("text", "")))
        if pid is None:
            continue
        matched += 1
        for ent in doc.get("extracted_entities", []):
            if not is_keep_entity(ent):
                continue
            key = clean_entity(ent)
            entity_display_by_key.setdefault(key, str(ent).strip())
            entity_to_passages[key].add(pid)
        for trip in doc.get("extracted_triples", []):
            if isinstance(trip, (list, tuple)) and len(trip) == 3:
                s_key = clean_entity(trip[0])
                o_key = clean_entity(trip[2])
                if is_keep_entity(s_key) and is_keep_entity(o_key):
                    relation_triples.append((s_key, str(trip[1]).strip().lower(), o_key))

    entity_keys = sorted(entity_to_passages)
    entity_id = {key: idx for idx, key in enumerate(entity_keys)}
    entity_display = [entity_display_by_key.get(k, k) for k in entity_keys]

    adj = defaultdict(set)
    for s_key, _rel, o_key in relation_triples:
        if s_key == o_key:
            continue
        si = entity_id.get(s_key)
        oi = entity_id.get(o_key)
        if si is None or oi is None:
            continue
        adj[si].add(oi)
        adj[oi].add(si)

    print(f"[entity_rag] Triplet docs matched: {matched:,}/{len(docs):,} | "
          f"entities: {len(entity_keys):,} | triples: {len(relation_triples):,} | "
          f"relation edges: {sum(len(v) for v in adj.values()) // 2:,} (undirected)", flush=True)
    return entity_keys, entity_id, entity_display, entity_to_passages, adj


def load_passage_embeddings(backend, unique_passages, scratch_dir):
    """Reuse the passage-embedding cache built by run_musique_graphrag, else encode."""
    embed_dim = backend.dim
    cache_path = os.path.join(
        scratch_dir, f"vectors_{m.embedding_cache_stem(backend.key)}_{embed_dim}.npy"
    )
    expected_shape = (len(unique_passages), embed_dim)
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        if cached.shape == expected_shape:
            print(f"[entity_rag] Reusing passage embeddings {cache_path} "
                  f"({cached.shape[0]:,} rows)", flush=True)
            return cached.astype(np.float32, copy=False)
        print(f"[entity_rag] Cached passage matrix {cached.shape} != expected "
              f"{expected_shape}; re-encoding.", flush=True)

    passage_texts = [f"{p['title']}: {p['text']}" for p in unique_passages.values()]
    print(f"[entity_rag] Encoding {len(passage_texts):,} passages with "
          f"{backend.hf_id}...", flush=True)
    embeds = backend.encode_documents(passage_texts, batch_size=32, show_progress_bar=True)
    return m.l2_normalize(embeds)


def load_entity_embeddings(backend, entity_display, scratch_dir, triplets_json):
    """Entity-name embeddings, cached per embedder + triplets file."""
    embed_dim = backend.dim
    tag = hashlib.sha1(os.path.basename(triplets_json).encode("utf-8")).hexdigest()[:8]
    cache_path = os.path.join(
        scratch_dir, f"entity_vectors_{m.embedding_cache_stem(backend.key)}_{embed_dim}_{tag}.npy"
    )
    expected_shape = (len(entity_display), embed_dim)
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        if cached.shape == expected_shape:
            print(f"[entity_rag] Reusing entity embeddings {cache_path} "
                  f"({cached.shape[0]:,} rows)", flush=True)
            return cached.astype(np.float32, copy=False)
        print(f"[entity_rag] Cached entity matrix {cached.shape} != expected "
              f"{expected_shape}; re-encoding.", flush=True)

    print(f"[entity_rag] Encoding {len(entity_display):,} entity names with "
          f"{backend.hf_id}...", flush=True)
    embeds = backend.encode_documents(entity_display, batch_size=64, show_progress_bar=True)
    embeds = m.l2_normalize(embeds)
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    np.save(cache_path, embeds)
    return embeds


def _hnsw_index_type(mode: str) -> str:
    return {"tq8": "hnsw_tq8", "tq4": "hnsw_tq4", "sq8": "hnsw_sq8"}.get(mode, "hnsw")


def build_entity_rag_tables(bsdb, scratch_dir, unique_passages, entity_keys, entity_id,
                            entity_to_passages, adj, corpus_matrix, entity_embeds,
                            embed_dim, mode, rebuild):
    """Materialize Passages, Entities, Relations and Mentions as native tables."""
    passages_dir = os.path.join(scratch_dir, "entity_rag_passages")
    entities_dir = os.path.join(scratch_dir, "entity_rag_entities")
    relations_dir = os.path.join(scratch_dir, "entity_rag_relations")
    mentions_dir = os.path.join(scratch_dir, "entity_rag_mentions")

    all_dirs = [passages_dir, entities_dir, relations_dir, mentions_dir]
    exists = all(os.path.exists(d) for d in all_dirs)
    if exists and not rebuild:
        print(f"[entity_rag] Reusing entity-RAG tables at {scratch_dir}...", flush=True)
        return (bsdb.Table(passages_dir), bsdb.Table(entities_dir),
                bsdb.Table(relations_dir), bsdb.Table(mentions_dir))

    for d in all_dirs:
        if os.path.exists(d):
            shutil.rmtree(d)

    index_type = _hnsw_index_type(mode)
    print(f"[entity_rag] Building entity-RAG tables ({index_type})...", flush=True)

    # 1. Passages table
    passages_table = bsdb.Table.create(
        passages_dir,
        schema=pa.schema([
            pa.field("id", pa.uint64()),
            pa.field("doc_key", pa.string()),
            pa.field("title", pa.string()),
            pa.field("text", pa.string()),
            pa.field("embedding", pa.list_(pa.float32(), embed_dim)),
        ]),
    )
    p_ids = [p["id"] for p in unique_passages.values()]
    p_keys = [p["key"] for p in unique_passages.values()]
    p_titles = [p["title"] for p in unique_passages.values()]
    p_texts = [p["text"] for p in unique_passages.values()]
    p_fsl = pa.FixedSizeListArray.from_arrays(pa.array(corpus_matrix.reshape(-1)), embed_dim)
    passages_table.write(pa.Table.from_arrays(
        [pa.array(p_ids, type=pa.uint64()), pa.array(p_keys), pa.array(p_titles),
         pa.array(p_texts), p_fsl],
        names=["id", "doc_key", "title", "text", "embedding"],
    ))
    passages_table.commit()
    passages_table.wait_for_background_tasks()
    passages_table.add_index("embedding", {"type": index_type, "complexity": 16,
                                           "quality": 100, "metric": "cosine"})
    passages_table.wait_for_background_tasks()
    print(f"    -> Passages table: {len(p_ids):,} rows", flush=True)

    # 2. Entities table (node table with an embedding index)
    entities_table = bsdb.Table.create(
        entities_dir,
        schema=pa.schema([
            pa.field("id", pa.uint64()),
            pa.field("name", pa.string()),
            pa.field("embedding", pa.list_(pa.float32(), embed_dim)),
        ]),
    )
    e_ids = list(range(len(entity_keys)))
    e_names = entity_keys  # canonical lowercased names; display casing lives in the report only
    e_fsl = pa.FixedSizeListArray.from_arrays(pa.array(entity_embeds.reshape(-1)), embed_dim)
    entities_table.write(pa.Table.from_arrays(
        [pa.array(e_ids, type=pa.uint64()), pa.array(e_names), e_fsl],
        names=["id", "name", "embedding"],
    ))
    entities_table.commit()
    entities_table.wait_for_background_tasks()
    entities_table.add_index("embedding", {"type": index_type, "complexity": 16,
                                           "quality": 100, "metric": "cosine"})
    entities_table.wait_for_background_tasks()
    print(f"    -> Entities table: {len(e_ids):,} rows", flush=True)

    # 3. Relations edge table (entity -> entity)
    relations_table = bsdb.Table.create_edge_table(
        relations_dir, node_id_type="uint64", with_relation=True, with_weight=True)
    r_src, r_dst, r_rel, r_w = [], [], [], []
    for si in sorted(adj):
        for oi in sorted(adj[si]):
            r_src.append(si)
            r_dst.append(oi)
            r_rel.append("related")
            r_w.append(1.0)
    if r_src:
        relations_table.write(pa.Table.from_arrays(
            [pa.array(r_src, type=pa.uint64()), pa.array(r_dst, type=pa.uint64()),
             pa.array(r_rel, type=pa.string()), pa.array(r_w, type=pa.float64())],
            names=["source", "target", "relation", "weight"],
        ))
        relations_table.commit()
        relations_table.wait_for_background_tasks()
        relations_table.add_index("source", {"type": "csr", "src_column": "source",
                                             "dst_column": "target"})
        relations_table.wait_for_background_tasks()
    print(f"    -> Relations table: {len(r_src):,} edges", flush=True)

    # 4. Mentions edge table (entity -> passage)
    mentions_table = bsdb.Table.create_edge_table(
        mentions_dir, node_id_type="uint64", with_relation=True, with_weight=True)
    n_src, n_dst, n_rel, n_w = [], [], [], []
    for key in entity_keys:
        si = entity_id[key]
        for pid in sorted(entity_to_passages[key]):
            n_src.append(si)
            n_dst.append(pid)
            n_rel.append("mentions")
            n_w.append(1.0)
    if n_src:
        mentions_table.write(pa.Table.from_arrays(
            [pa.array(n_src, type=pa.uint64()), pa.array(n_dst, type=pa.uint64()),
             pa.array(n_rel, type=pa.string()), pa.array(n_w, type=pa.float64())],
            names=["source", "target", "relation", "weight"],
        ))
        mentions_table.commit()
        mentions_table.wait_for_background_tasks()
        mentions_table.add_index("source", {"type": "csr", "src_column": "source",
                                            "dst_column": "target"})
        mentions_table.wait_for_background_tasks()
    print(f"    -> Mentions table: {len(n_src):,} edges", flush=True)

    return passages_table, entities_table, relations_table, mentions_table


def vector_ids(table, q_embed, k, ef_search):
    res = table.to_arrow(
        vector_filter={"column": "embedding", "query": q_embed.tolist(),
                       "k": k, "ef_search": ef_search},
        columns=["id"],
    )
    if "id" in res.column_names:
        return [int(x) for x in res.column("id").to_pylist()]
    return []


def bfs_relations(relations_table, seed_ids, hops, entity_cap, max_frontier=64):
    """BFS the Relations edge table from entity seed ids; seed-first order."""
    visited = []
    seen = set()
    frontier = [int(s) for s in seed_ids]
    for _ in range(max(0, hops) + 1):
        if not frontier or len(visited) >= entity_cap:
            break
        level = []
        for eid in frontier:
            if eid in seen or len(visited) >= entity_cap:
                continue
            seen.add(eid)
            visited.append(eid)
            level.append(eid)
        if not level:
            break
        id_list = ", ".join(str(i) for i in level[:max_frontier])
        try:
            res = relations_table.to_arrow(filter=f"source IN ({id_list})",
                                           columns=["target"])
        except Exception:
            break
        next_frontier = []
        for dst in res.column("target").to_pylist():
            dst = int(dst)
            if dst not in seen:
                next_frontier.append(dst)
        frontier = next_frontier
    return visited


def collect_passages(mentions_table, entity_ids, candidates, max_frontier=256):
    """Follow Mentions edges from entity ids to passages, seed-first, capped."""
    pool = []
    seen = set()
    for start in range(0, len(entity_ids), max_frontier):
        chunk = entity_ids[start:start + max_frontier]
        id_list = ", ".join(str(e) for e in chunk)
        try:
            res = mentions_table.to_arrow(filter=f"source IN ({id_list})",
                                          columns=["target"])
        except Exception:
            continue
        for dst in res.column("target").to_pylist():
            dst = int(dst)
            if dst not in seen:
                seen.add(dst)
                pool.append(dst)
                if len(pool) >= candidates:
                    return pool
    return pool


def evaluate(passages_table, entities_table, relations_table, mentions_table,
             unique_passages, records, *, top_k, seed_k, hops, entity_cap,
             candidates, reranker, ef_search):
    backend = m.get_embedding_model()
    id_to_key = {info["id"]: info["key"] for info in unique_passages.values()}
    id_to_title = {info["id"]: info.get("title", "") for info in unique_passages.values()}
    id_to_text = {}
    for info in unique_passages.values():
        body = (info.get("text") or "").strip()
        title = (info.get("title") or "").strip()
        id_to_text[info["id"]] = f"{title}: {body}" if title and body else (body or title)

    print(f"[entity_rag] Pre-encoding {len(records):,} queries with {backend.hf_id}...", flush=True)
    q_texts = [r["question"] for r in records]
    q_embeds = backend.encode_queries(q_texts, batch_size=32, show_progress_bar=False)

    reranker_model = None
    if reranker in ("minilm", "bge"):
        reranker_model = m.get_minilm_reranker() if reranker == "minilm" else m.get_bge_reranker()
        if reranker_model is not None:
            reranker_model.predict([("warmup question", "warmup passage")],
                                   show_progress_bar=False)

    modes = ["vector", "entity_seed", "entity_bfs", "entity_rag"]
    results = {}

    for mode in modes:
        recalls, hit_rates, all_hit_rates, precisions = [], [], [], []
        latencies_ms = []
        t_start = time.perf_counter()

        for r_idx, r in enumerate(records):
            q_text = r["question"]
            q_embed = q_embeds[r_idx]

            supporting_keys = {
                m.passage_key(p["title"], p["paragraph_text"])
                for p in r["paragraphs"] if p["is_supporting"]
            }
            gt_ids = {info["id"] for key, info in unique_passages.items()
                      if key in supporting_keys}
            if not gt_ids:
                continue

            t0 = time.perf_counter()

            if mode == "vector":
                retrieved = vector_ids(passages_table, q_embed, top_k, ef_search)[:top_k]

            elif mode == "entity_seed":
                seeds = vector_ids(entities_table, q_embed, seed_k, ef_search)
                pool = collect_passages(mentions_table, seeds, top_k)
                retrieved = pool[:top_k]

            elif mode == "entity_bfs":
                seeds = vector_ids(entities_table, q_embed, seed_k, ef_search)
                visited = bfs_relations(relations_table, seeds, hops, entity_cap)
                pool = collect_passages(mentions_table, visited, top_k)
                retrieved = pool[:top_k]

            else:  # entity_rag
                seeds = vector_ids(entities_table, q_embed, seed_k, ef_search)
                visited = bfs_relations(relations_table, seeds, hops, entity_cap)
                pool = collect_passages(mentions_table, visited, candidates)
                candidate_texts = [id_to_text.get(pid) or id_to_key.get(pid, "") or str(pid)
                                   for pid in pool]
                scores = m.cross_encoder_scores(reranker_model, q_text, candidate_texts)
                order = np.argsort(-scores, kind="stable") if scores.size else np.arange(len(pool))
                retrieved = [pool[int(j)] for j in order[:top_k]]

            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

            ret_set = set(retrieved)
            hits = len(ret_set.intersection(gt_ids))
            recall = hits / len(gt_ids) if gt_ids else 0.0
            precision = hits / len(ret_set) if ret_set else 0.0
            recalls.append(recall)
            hit_rates.append(1.0 if hits > 0 else 0.0)
            all_hit_rates.append(1.0 if hits == len(gt_ids) else 0.0)
            precisions.append(precision)

            if r_idx < 3:
                gt_titles = [id_to_title.get(g, str(g)) for g in sorted(gt_ids)]
                ret_titles = [id_to_title.get(rid, str(rid)) for rid in retrieved]
                print(f"    [probe {mode} q{r_idx}] hits={hits}/{len(gt_ids)} "
                      f"recall={recall:.2f}\n      Q: {q_text[:70]}\n"
                      f"      GT: {list(zip(sorted(gt_ids), gt_titles))}\n"
                      f"      Retrieved: {list(zip(retrieved, ret_titles))}", flush=True)

        total_s = time.perf_counter() - t_start
        results[mode] = {
            "evidence_recall_at_k": float(np.mean(recalls)) if recalls else 0.0,
            "hit_rate_at_k": float(np.mean(hit_rates)) if hit_rates else 0.0,
            "all_hit_rate_at_k": float(np.mean(all_hit_rates)) if all_hit_rates else 0.0,
            "precision_at_k": float(np.mean(precisions)) if precisions else 0.0,
            "qps": len(latencies_ms) / max(total_s, 0.001),
            "p50_ms": float(np.percentile(latencies_ms, 50)) if latencies_ms else 0.0,
            "p99_ms": float(np.percentile(latencies_ms, 99)) if latencies_ms else 0.0,
            "queries_evaluated": len(latencies_ms),
        }
        print(f"    -> {mode:<12} Recall@{top_k}: {results[mode]['evidence_recall_at_k']:.4f} | "
              f"Hit@{top_k}: {results[mode]['hit_rate_at_k']:.4f} | "
              f"All-hit@{top_k}: {results[mode]['all_hit_rate_at_k']:.4f} | "
              f"Precision@{top_k}: {results[mode]['precision_at_k']:.4f} | "
              f"p50: {results[mode]['p50_ms']:.2f} ms", flush=True)

    return results


def save_report(results, top_k, out_dir, *, seed_k, hops, entity_cap, candidates,
                reranker, n_passages, n_entities, graph_source):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "entity_rag_results.json")
    payload = {
        "description": "Entity-level Vector Graph RAG on BenoStreamDB "
                       "(Passages + Entities + Relations + Mentions tables)",
        "top_k": top_k,
        "seed_k": seed_k,
        "hops": hops,
        "entity_cap": entity_cap,
        "candidates": candidates,
        "reranker": reranker,
        "passages": n_passages,
        "entities": n_entities,
        "graph_source": graph_source,
        "strategies": results,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[entity_rag] Wrote report to {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=m.DEFAULT_DATA_DIR)
    parser.add_argument("--scratch-dir", default=m.DEFAULT_SCRATCH_DIR)
    parser.add_argument("--corpus-json", default="",
                        help="External deduplicated passage corpus (JSON list of "
                             "{title, text}). Recommended for parity.")
    parser.add_argument("--triplets-json", default="",
                        help="HippoRAG OpenIE file (docs[].extracted_entities / "
                             "extracted_triples). Required for the entity graph.")
    parser.add_argument("--sample-manifest", default="",
                        help="Zilliz/HippoRAG manifest.json (samples.<dataset>).")
    parser.add_argument("--manifest-dataset", default="musique")
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed-k", type=int, default=2,
                        help="Entity seeds retrieved by the query (default: 2).")
    parser.add_argument("--hops", type=int, default=2,
                        help="Relation-graph BFS hops from the entity seeds (default: 2).")
    parser.add_argument("--entity-cap", type=int, default=64,
                        help="Maximum entities visited by the BFS (default: 64).")
    parser.add_argument("--candidates", type=int, default=24,
                        help="Passage pool size passed to the reranker (default: 24).")
    parser.add_argument("--reranker", choices=["none", "minilm", "bge"],
                        default="bge",
                        help="Cross-encoder for the entity_rag strategy (default: bge).")
    parser.add_argument("--mode", choices=["f32", "tq8", "tq4", "sq8"], default="f32",
                        help="Vector index quantizer mode (default: f32).")
    parser.add_argument("--ef-search", type=int, default=m.DEFAULT_EF_SEARCH,
                        help=f"HNSW ef_search (default: {m.DEFAULT_EF_SEARCH}).")
    parser.add_argument("--embed-model", choices=list(m.EMBED_MODEL_PROFILES),
                        default=m.DEFAULT_EMBED_MODEL)
    parser.add_argument("--embed-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--rebuild", action="store_true",
                        help="Rebuild the entity-RAG tables even if they exist.")
    parser.add_argument("--out-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results", "musique_entity_rag"))
    parser.add_argument("--hardware", choices=["d8lds", "d8s", "e8ds", "local", "auto"],
                        default="d8s")
    parser.add_argument("--strict-hardware", action="store_true",
                        help="Fail if actual cgroup limits exceed the chosen --hardware profile.")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--cache-gb", type=int, default=16)
    args = parser.parse_args()

    hw = SimpleNamespace(
        hardware=args.hardware,
        strict_hardware=args.strict_hardware,
        threads=args.threads,
        cache_gb=args.cache_gb,
        preload_gb=0.0,
        embed_model=args.embed_model,
        embed_device=args.embed_device,
    )
    m.apply_hardware(hw)
    m.configure_rerank_device("auto")
    backend = m.configure_embedding(args.embed_model, device=args.embed_device,
                                    dtype="auto", max_seq_length=0)
    print(f"[entity_rag] Retrieval embedder: {backend.describe()}", flush=True)

    import benostreamdb as bsdb

    dataset_path = m.ensure_dataset(args.data_dir)
    records = m.load_musique_records(dataset_path, args.max_questions)
    print(f"[entity_rag] Loaded {len(records):,} MuSiQue questions.", flush=True)

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
        print(f"[entity_rag] Manifest: {len(records):,}/{len(samples):,} questions matched.",
              flush=True)

    unique_passages = build_passages(args.corpus_json, records)

    if not args.triplets_json:
        raise SystemExit("[ERROR] --triplets-json is required for the entity graph.")

    entity_keys, entity_id, entity_display, entity_to_passages, adj = build_entity_graph(
        args.triplets_json, unique_passages)

    corpus_matrix = load_passage_embeddings(backend, unique_passages, args.scratch_dir)
    entity_embeds = load_entity_embeddings(backend, entity_display, args.scratch_dir,
                                           args.triplets_json)

    passages_table, entities_table, relations_table, mentions_table = build_entity_rag_tables(
        bsdb, args.scratch_dir, unique_passages, entity_keys, entity_id,
        entity_to_passages, adj, corpus_matrix, entity_embeds, backend.dim,
        args.mode, args.rebuild)

    results = evaluate(
        passages_table, entities_table, relations_table, mentions_table,
        unique_passages, records,
        top_k=args.top_k,
        seed_k=args.seed_k,
        hops=args.hops,
        entity_cap=args.entity_cap,
        candidates=args.candidates,
        reranker=args.reranker,
        ef_search=args.ef_search,
    )
    save_report(
        results, args.top_k, args.out_dir,
        seed_k=args.seed_k, hops=args.hops, entity_cap=args.entity_cap,
        candidates=args.candidates, reranker=args.reranker,
        n_passages=len(unique_passages), n_entities=len(entity_keys),
        graph_source="HippoRAG OpenIE triplets",
    )


if __name__ == "__main__":
    main()
