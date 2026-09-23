#!/usr/bin/env python3
"""End-to-end smoke of the full-site demo tables — mirrors every app.py engine call.

Usage: python scripts/demo_e2e_smoke.py   (engine-side only; no LLM required)
"""
import os
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HYPERSTREAM_CACHE_GB", "40")  # same default as app.py
import hyperstreamdb as hdb

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Tables live on the SSD in the repo's original location.
DB = f"file://{os.path.join(REPO, 'data', 'wiki_graph_db')}"

t0 = time.time()
nodes = hdb.Table(f"{DB}/nodes")
edges = hdb.Table(f"{DB}/edges")
print(f"[open tables] {time.time()-t0:.1f}s")

t0 = time.time()
nc = int(nodes.execute_sql("SELECT count(*) c FROM t").to_pandas()["c"].iloc[0])
print(f"[count nodes] {nc:,} in {time.time()-t0:.1f}s")
t0 = time.time()
ec = int(edges.execute_sql("SELECT count(*) c FROM t").to_pandas()["c"].iloc[0])
print(f"[count edges] {ec:,} in {time.time()-t0:.1f}s")

from sentence_transformers import SentenceTransformer

# CPU embedder by default: leaves the GPU free for vLLM while the demo serves.
dev = "cuda" if os.environ.get("HDB_SMOKE_GPU") == "1" else "cpu"
m = SentenceTransformer("all-MiniLM-L6-v2", device=dev)
vec = m.encode(["how do neural networks relate to the Turing test?"]).tolist()[0]

t0 = time.time()
r = nodes.vector_search("embedding", vec, k=5, columns=["id", "title"])
print(f"[vector_search COLD] {time.time()-t0:.1f}s")
t0 = time.time()
r = nodes.vector_search("embedding", vec, k=5, columns=["id", "title"])
print(f"[vector_search WARM] {time.time()-t0:.2f}s")
print(r[["id", "title", "distance"]].to_string(index=False))

t0 = time.time()
h = nodes.hybrid_search(text_column="title", query_text="Turing test machine intelligence",
                        vector_column="embedding", query_vector=vec, k=5,
                        columns=["id", "title"])
print(f"[hybrid_search] {time.time()-t0:.2f}s rows={len(h)}")
print(h[["id", "title"]].head(3).to_string(index=False) if "title" in h.columns else h.head(3))

t0 = time.time()
res = nodes.graph_rag_search(query=vec, edge_table=edges, doc_table=nodes, mode="local",
                             vector_column="embedding", id_column="id", top_k=5, hops=2)
ctx = res.nodes
print(f"[graph_rag_search] {time.time()-t0:.1f}s seeds={len(res.seeds)} "
      f"ctx_rows={0 if ctx is None else len(ctx)} edges={0 if res.edges is None else len(res.edges)}")

if ctx is not None and len(ctx):
    pool = [int(x) for x in ctx["id"].tolist()]
    t0 = time.time()
    rr = nodes.vector_search("embedding", vec, k=8,
                             filter=f"id IN ({', '.join(map(str, pool))})",
                             columns=["id", "title"])
    print(f"[rerank bitmap-filtered] {time.time()-t0:.2f}s pool={len(pool)} -> {len(rr)}")
    print(rr[["id", "title", "distance"]].to_string(index=False))

seed = int(r["id"].iloc[0])
t0 = time.time()
sg = edges.subgraph(seeds=[seed], hops=1, directed=False)
print(f"[subgraph 1-hop from {seed}] {time.time()-t0:.2f}s rows={len(sg)}")

if len(r) > 1:
    t0 = time.time()
    sp = edges.shortest_path(seed, int(r["id"].iloc[1]), graph_column="source")
    path = list(sp) if sp is not None else []
    print(f"[shortest_path] {time.time()-t0:.2f}s hops={len(path)}")

print("E2E_SMOKE_OK")
