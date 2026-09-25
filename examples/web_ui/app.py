"""BenoStreamDB — full-site Wikipedia Graph RAG demo.

Runs against the WHOLE English-Wikipedia graph prepared by:

    python scripts/prepare_demo.py

Two persistent BenoStreamDB tables are used:
  * data/wiki_graph_db/edges — (source, target) int64 + CSR graph index
  * data/wiki_graph_db/nodes — (id, title, summary, embedding) + HNSW-TQ
                               vector index (384-d article centroids) + BM25

Capabilities on display (each degrades gracefully if its dependency is absent):
  Browse        keyset pagination through 51M pages (engine-side SQL)
  Semantic      hybrid retrieval: BM25 + dense vectors fused with RRF
  Graph RAG     end-to-end local search: seed discovery -> multi-hop subgraph
                -> Personalized PageRank (HippoRAG-style) -> context
  DRIFT         regional community summarization + DRIFT search
  Traversals    CSR microsecond hops: shortest_path / connecting_paths /
                graph_neighbors / induced subgraph

The LLM (any OpenAI-compatible endpoint, e.g. local vLLM) is optional and only
used for query parsing and answer synthesis; configure via
.streamlit/secrets.toml [llm].
"""

import json
import os

# Serving default: the engine's 1 GB index-cache cannot hold whole-site segment
# indexes (69 segments x ~300 MB TQ8 graphs), so every query would evict and
# re-deserialize them. Size it to 40 GB unless the operator set it explicitly.
# On smaller hosts lower it via BENOSTREAM_CACHE_GB, or use the pruned dataset.
os.environ.setdefault("BENOSTREAM_CACHE_GB", "40")

import shutil
import time

import pandas as pd
import streamlit as st

import benostreamdb

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DB = os.environ.get("BSDB_DEMO_DB", os.path.join(REPO, "data", "wiki_graph_db"))
EDGES_URI = f"file://{os.path.join(DB, 'edges')}"
NODES_URI = f"file://{os.path.join(DB, 'nodes')}"
EMBED_MODEL = os.environ.get("BSDB_DEMO_EMBED_MODEL", "all-MiniLM-L6-v2")

st.set_page_config(page_title="BenoStreamDB — Wikipedia Graph RAG", layout="wide")
st.title("BenoStreamDB — Wikipedia Graph RAG")


# ── LLM (optional) ──────────────────────────────────────────────────────────
def llm_config():
    """LLM endpoint resolution: standard OpenAI env vars > secrets.toml > defaults.

    Any OpenAI-compatible provider works — a local vLLM, or a hosted one like
    OpenRouter (OPENAI_BASE_URL=https://openrouter.ai/api/v1,
    OPENAI_API_KEY=sk-or-v1-…, OPENAI_MODEL=qwen/qwen3.8-27b:free).
    """
    try:
        secrets = st.secrets.get("llm", {})
    except Exception:
        secrets = {}
    return {
        "base_url": os.environ.get("OPENAI_BASE_URL",
                                   secrets.get("base_url", "http://127.0.0.1:18020/v1")),
        "api_key": os.environ.get("OPENAI_API_KEY",
                                  secrets.get("api_key", "empty")),
        "model": os.environ.get("OPENAI_MODEL",
                                secrets.get("model", "qwen3.8-27b")),
    }


def llm_available() -> bool:
    return os.environ.get("BSDB_DEMO_LLM", "1") != "0"


def llm_chat(messages, temperature: float = 0.0, json_mode: bool = False) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai package not installed: pip install openai")
    cfg = llm_config()
    client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])
    kwargs = {"model": cfg["model"], "messages": messages, "temperature": temperature}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    return client.chat.completions.create(**kwargs).choices[0].message.content


# ── Query embedder (must match the dataset embedding model) ────────────────
@st.cache_resource(show_spinner="Loading query embedder...")
def get_embedder():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(EMBED_MODEL, device="cpu")
    except Exception as e:
        st.warning(f"Embedder unavailable ({e}); semantic features fall back to keyword search.")
        return None


def embed_text(text: str):
    m = get_embedder()
    if m is None:
        return None
    return m.encode([text], normalize_embeddings=True)[0].tolist()


# ── Tables ──────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Opening BenoStreamDB tables...")
def open_tables():
    edges_dir = os.path.join(DB, "edges")
    nodes_dir = os.path.join(DB, "nodes")
    if not (os.path.isdir(edges_dir) and os.path.isdir(nodes_dir)):
        # Preparation is a multi-hour pipeline; report which stage we're at.
        emb_dir = os.path.join(REPO, "data", "embeddings")
        shards = len([f for f in os.listdir(emb_dir)
                      if f.endswith(".parquet")]) if os.path.isdir(emb_dir) else 0
        stages = [
            ("downloaded dumps", any(f.endswith(".xml.bz2")
                                     for f in os.listdir(os.path.join(REPO, "data")))),
            ("resolved parquet", os.path.exists(os.path.join(REPO, "data", "wiki_edges.parquet"))),
            (f"embedding shards ({shards} done)", shards > 0),
            ("graph tables", os.path.isdir(edges_dir)),
        ]
        progress = "\n".join(f"- {'✅' if ok else '⏳'} {name}" for name, ok in stages)
        st.error(
            f"Demo tables not ready under `{DB}`.\n\n"
            f"**Preparation progress:**\n{progress}\n\n"
            "Build them with:  `python scripts/prepare_demo.py` "
            "(idempotent — rerun to resume; see README.md)"
        )
        st.stop()
    edges = benostreamdb.Table(EDGES_URI)
    nodes = benostreamdb.Table(NODES_URI)
    return edges, nodes


with st.sidebar:
    st.header("Setup")
    st.caption(f"DB: `{DB}`")
    st.caption(f"Embed model: `{EMBED_MODEL}` · LLM: `{llm_config()['model']}`")
    llm_on = st.toggle("Use LLM (parsing + synthesis)", value=llm_available())
    if st.button("Clear caches"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

edges_t, nodes_t = open_tables()


# ── Helpers over the engine (no full-table pandas loads) ─────────────────────
def as_df(res):
    if res is None:
        return pd.DataFrame()
    if hasattr(res, "to_pandas"):
        return res.to_pandas()
    return res if isinstance(res, pd.DataFrame) else pd.DataFrame(res)


@st.cache_data(ttl=300, show_spinner=False)
def lookup_titles(ids):
    if not ids:
        return {}
    id_list = ", ".join(str(int(i)) for i in ids)
    df = as_df(nodes_t.execute_sql(f"SELECT id, title FROM t WHERE id IN ({id_list})"))
    if df.empty:
        return {}
    return {int(r["id"]): str(r["title"]) for _, r in df.iterrows()}


def find_pages(term: str, limit: int = 5):
    """Engine-side entity resolution: BM25 if available, LIKE fallback."""
    safe = term.replace("'", "''").replace("%", r"\%")
    try:
        df = as_df(nodes_t.execute_sql(
            f"SELECT id, title FROM t WHERE lower(title) LIKE lower('%{safe}%') LIMIT {limit}"
        ))
    except Exception:
        df = pd.DataFrame()
    return df


def graphviz_from_edges(edge_pairs, titles):
    dot = ['digraph G {', '  bgcolor="transparent";',
           '  node [shape=box, style=filled, fillcolor=lightblue, fontname="Helvetica"];']
    for u, v in edge_pairs:
        ut = str(titles.get(int(u), u)).replace('"', '\\"')
        vt = str(titles.get(int(v), v)).replace('"', '\\"')
        dot.append(f'  "{ut}" -> "{vt}";')
    dot.append("}")
    return "\n".join(dot)


TAB_BROWSE, TAB_SEMANTIC, TAB_GRAPHRAG, TAB_DRIFT, TAB_TRAVERSAL = st.tabs([
    "Browse", "Semantic search", "Graph RAG (local)", "DRIFT (regional)", "Graph traversals",
])

# ── 1. Browse ────────────────────────────────────────────────────────────────
with TAB_BROWSE:
    st.markdown("#### Browse the 51M-page corpus (keyset pagination, engine-side)")
    if "cursor" not in st.session_state:
        st.session_state.cursor = -1
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        if st.button("First page") or st.session_state.cursor < 0:
            st.session_state.cursor = -1
            st.session_state.stack = []
    with col3:
        if st.button("Next page"):
            st.session_state.stack.append(st.session_state.cursor)
    with col2:
        jump = st.text_input("Jump to page containing title…", placeholder="e.g. Anarchism")
    if jump:
        hit = find_pages(jump, 1)
        if not hit.empty:
            st.session_state.cursor = int(hit.iloc[0]["id"]) - 1
            st.session_state.stack = []
        else:
            st.warning("No match.")

    page = as_df(nodes_t.execute_sql(
        f"SELECT id, title FROM t WHERE id > {st.session_state.cursor} ORDER BY id LIMIT 15"
    ))
    if page.empty:
        st.info("End of corpus.")
    else:
        st.session_state.cursor = int(page["id"].iloc[-1])
        sel = st.dataframe(page, hide_index=True, height=380,
                           on_select="rerun", selection_mode="single-row")
        rows = sel.selection.rows
        if rows:
            pid = int(page.iloc[rows[0]]["id"])
            detail = as_df(nodes_t.execute_sql(
                f"SELECT id, title, summary FROM t WHERE id = {pid}"))
            if not detail.empty:
                r = detail.iloc[0]
                st.markdown(f"**{r['title']}**  \n  id: `{r['id']}` · "
                            f"[Wikipedia](https://en.wikipedia.org/wiki/?curid={r['id']})")
                st.info(str(r["summary"])[:1200])
    if st.button("Previous page") and st.session_state.get("stack"):
        st.session_state.cursor = st.session_state.stack.pop()
        st.rerun()

# ── 2. Semantic search ───────────────────────────────────────────────────────
with TAB_SEMANTIC:
    st.markdown("#### Hybrid retrieval — BM25 + dense vectors fused via RRF")
    q = st.text_input("Query", key="sem_q", placeholder="what causes auroras?")
    k = st.slider("Top k", 5, 50, 10, key="sem_k")
    if st.button("Search", key="sem_go", type="primary") and q:
        vec = embed_text(q)
        with st.spinner("Querying vector + BM25 indexes..."):
            try:
                if vec is not None:
                    res = nodes_t.hybrid_search(text_column="title", query_text=q,
                                                vector_column="embedding",
                                                query_vector=vec, k=k)
                    mode = "hybrid (BM25 + HNSW-TQ, RRF)"
                else:
                    safe_q = q.replace("'", "''")
                    res = nodes_t.execute_sql(
                        f"SELECT id, title FROM t WHERE lower(title) LIKE lower('%{safe_q}%') LIMIT {k}")
                    mode = "keyword only (no embedder)"
            except Exception as e:
                res, mode = None, f"failed: {e}"
        df = as_df(res)
        st.caption(f"Mode: {mode}")
        if df is None or df.empty:
            st.info("No results.")
        else:
            show = df[["id", "title"]] if "title" in df.columns else df
            st.dataframe(show.head(k), hide_index=True, height=420)

# ── 3. Graph RAG (local search: seeds -> subgraph -> PPR) ───────────────────
with TAB_GRAPHRAG:
    st.markdown("#### Local Graph RAG — seed discovery, multi-hop subgraph, Personalized PageRank")
    q = st.text_input("Question", key="gr_q",
                      placeholder="How are neural networks connected to the Turing test?")
    c1, c2 = st.columns([1, 4])
    with c1:
        top_k = st.number_input("Seeds", 1, 20, 5)
        hops = st.slider("Hops", 1, 3, 2, key="gr_hops")
        rerank_n = st.number_input("Rerank top-N", 1, 20, 8, key="gr_rerank")
    with c2:
        st.caption("Two-level retrieval: seed pages found via the 384-d centroid index, an "
                   "induced subgraph is expanded over the CSR edges and ranked with "
                   "Personalized PageRank (HippoRAG-style). The PPR pool is then re-scored by "
                   "a vector search **filtered to that pool** (`id IN (...)` → RoaringBitmap "
                   "predicate pushdown): topology prunes, semantics orders.")
    if st.button("Run Graph RAG", key="gr_go", type="primary") and q:
        vec = embed_text(q)
        with st.expander("Pipeline", expanded=True):
            with st.spinner("graph_rag_search: vector seeds -> CSR subgraph -> PPR -> bitmap-filtered rerank..."):
                try:
                    result = nodes_t.graph_rag_search(
                        query=vec if vec is not None else q,
                        edge_table=edges_t, doc_table=nodes_t,
                        mode="local", vector_column="embedding",
                        id_column="id", top_k=int(top_k), hops=int(hops),
                    )
                    seeds = [int(s) for s in (result.seeds or [])]
                    seed_titles = lookup_titles(seeds)
                    st.write("**Seeds:**", [seed_titles.get(i, i) for i in seeds])
                    ctx_df = as_df(result.nodes) if getattr(result, "nodes", None) is not None else pd.DataFrame()
                    if not ctx_df.empty:
                        st.dataframe(ctx_df.head(30), hide_index=True, height=300)
                    ids = seeds + ([int(x) for x in ctx_df["id"]] if "id" in ctx_df.columns else [])
                    titles = lookup_titles(ids)
                    if titles and getattr(result, "edges", None) is not None:
                        e_df = as_df(result.edges)
                        if not e_df.empty:
                            pairs = list(zip(e_df["source"].astype("int64"), e_df["target"].astype("int64")))[:120]
                            st.graphviz_chart(graphviz_from_edges(pairs, titles))
                    context = result.format_context() if hasattr(result, "format_context") else ""

                    # Two-level rerank: the PPR neighborhood becomes a bitmap filter on the
                    # vector index — semantic re-scoring constrained to the subgraph that the
                    # CSR expansion proved topologically relevant.
                    if vec is not None and not ctx_df.empty and "id" in ctx_df.columns:
                        pool = [int(x) for x in ctx_df["id"].tolist()]
                        try:
                            rr = nodes_t.vector_search(
                                "embedding", vec, k=min(int(rerank_n), len(pool)),
                                filter=f"id IN ({', '.join(map(str, pool))})",
                                columns=["id", "title"],
                            )
                            if rr is not None and not rr.empty:
                                st.markdown(f"**Bitmap-filtered rerank** — same query vector, "
                                            f"search space constrained to the {len(pool)}-node "
                                            f"PPR neighborhood:")
                                show = rr[[c for c in ["_distance", "distance", "id", "title"] if c in rr.columns]]
                                st.dataframe(show, hide_index=True,
                                             height=min(40 + 35 * len(show), 350))
                                rr_ids = ", ".join(str(int(i)) for i in rr["id"].tolist())
                                rr_full = as_df(nodes_t.execute_sql(
                                    f"SELECT id, title, summary FROM t WHERE id IN ({rr_ids})"))
                                if not rr_full.empty:
                                    order = {int(i): r for r, i in enumerate(rr["id"].tolist())}
                                    rr_full = rr_full.assign(
                                        _o=rr_full["id"].astype(int).map(order)).sort_values("_o")
                                    block = "\n#### Reranked neighborhood (topology ∩ semantics)\n" + "\n".join(
                                        f"- **{row['title']}** — {str(row['summary'])[:400]}"
                                        for _, row in rr_full.iterrows())
                                    context = (context + "\n" + block) if context else block
                        except Exception as re_err:
                            st.warning(f"Rerank skipped ({type(re_err).__name__}: {re_err})")
                except Exception as e:
                    st.error(f"{type(e).__name__}: {e}")
                    context = ""
        if context and llm_on:
            with st.spinner("Synthesizing answer with LLM..."):
                try:
                    ans = llm_chat([{"role": "user", "content":
                        "Answer the question using ONLY the retrieved context. Cite page titles.\n\n"
                        f"Question: {q}\n\nContext:\n{context[:12000]}"}], temperature=0.3)
                    st.markdown("### Answer")
                    st.markdown(ans)
                except Exception as e:
                    st.warning(f"LLM unavailable ({e}). Raw context shown above.")
        elif context:
            st.markdown("#### Retrieved context")
            st.text(context[:4000])

# ── 4. DRIFT (regional communities) ─────────────────────────────────────────
with TAB_DRIFT:
    st.markdown("#### DRIFT — regional community summaries + iterative deepening search")
    q = st.text_input("Query", key="d_q", placeholder="what is the history of computing?")
    seeds_n = st.slider("Region seeds", 2, 10, 4, key="d_seeds")
    if st.button("Build region + search", key="d_go", type="primary") and q:
        tmp_edges = "/tmp/hdb_demo_region"
        tmp_comm = "file:///tmp/hdb_demo_communities"
        shutil.rmtree(tmp_edges, ignore_errors=True)
        shutil.rmtree("/tmp/hdb_demo_communities", ignore_errors=True)
        with st.status("Preparing regional graph...", expanded=True) as status:
            try:
                vec = embed_text(q)
                hits = as_df(nodes_t.vector_search("embedding", vec, k=seeds_n)
                             if vec else find_pages(q, seeds_n))
                seed_ids = [int(i) for i in hits["id"]][:seeds_n]
                st.write(f"Seeds: {[lookup_titles(seed_ids).get(i, i) for i in seed_ids]}")
                # 1-hop region via the CSR graph index (memory-mapped, ~ms/seed).
                # The generic edges_t.subgraph() runs a SQL frontier BFS over the
                # whole 383M-edge table (100 s+); for a 1-hop region the CSR path
                # is the same intent and orders of magnitude faster.
                t_expand = time.time()
                pairs = [(int(s), int(n)) for s in seed_ids
                         for n in edges_t.graph_neighbors(int(s), 1, graph_column="source")]
                region = pd.DataFrame(pairs, columns=["source", "target"])
                st.write(f"Region expansion: {time.time() - t_expand:.2f}s "
                         f"({len(region):,} edges from {len(seed_ids)} seeds)")
                if region.empty:
                    st.error("Empty region.");  status.update(label="No region", state="error")
                    st.stop()
                region = region.head(60_000)
                st.write(f"Region: {len(region):,} edges")
                import pyarrow as pa
                schema = pa.schema([("source", pa.int64()), ("target", pa.int64())])
                rt = benostreamdb.Table.create(f"file://{tmp_edges}", schema)
                rt.insert(pa.Table.from_pandas(region.astype("int64"), schema=schema, preserve_index=False))
                rt.commit(); rt.wait_for_background_tasks()
                st.write("Summarizing communities (Louvain + text rollup)...")
                comm = rt.summarize_communities(doc_table=nodes_t,
                                                target_uri=tmp_comm,
                                                id_column="id", content_column="summary")
                st.write("Running DRIFT search...")
                res = rt.drift_search(query=q, community_table=comm,
                                      follow_up_llm=None, n_depth=1,
                                      k_followups=2, top_k=5, hops=1)
                status.update(label="DRIFT complete", state="complete")
                nodes_found = res.get("all_discovered_nodes", [])
                titles = lookup_titles([int(n) for n in nodes_found][:80])
                st.markdown("**Discovered nodes:** " + ", ".join(
                    str(titles.get(int(n), n)) for n in nodes_found[:40]))
                acts = res.get("actions", [])
                if acts:
                    st.dataframe(pd.DataFrame(acts).head(20), hide_index=True)
            except Exception as e:
                status.update(label=f"Failed: {e}", state="error")

# ── 5. Traversals ────────────────────────────────────────────────────────────
with TAB_TRAVERSAL:
    st.markdown("#### Native CSR traversals — microsecond adjacency over memory-mapped index")
    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        a = st.text_input("Entity A", value="Artificial intelligence")
    with c2:
        b = st.text_input("Entity B", value="Alan Turing")
    with c3:
        op = st.selectbox("Operation", ["shortest_path", "connecting_paths",
                                        "graph_neighbors", "subgraph"])
    if st.button("Run traversal", type="primary"):
        ra, rb = find_pages(a, 1), find_pages(b, 1)
        if ra.empty:
            st.error(f"Could not resolve '{a}'.")
        else:
            ia = int(ra.iloc[0]["id"])
            ids = [ia]
            if not rb.empty:
                ids.append(int(rb.iloc[0]["id"]))
            titles = lookup_titles(ids)
            st.write("Resolved:", [titles.get(i, i) for i in ids])
            try:
                if op == "shortest_path" and len(ids) == 2:
                    path = [int(n) for n in edges_t.shortest_path(ids[0], ids[1], graph_column="source")]
                    pt = lookup_titles(path)
                    st.write(f"Path ({len(path)} hops):", " → ".join(str(pt.get(p, p)) for p in path))
                    st.graphviz_chart(graphviz_from_edges(list(zip(path[:-1], path[1:])), pt))
                elif op == "connecting_paths" and len(ids) == 2:
                    eps = [(int(u), int(v)) for u, v in edges_t.connecting_paths(ids, graph_column="source")]
                    st.write(f"{len(eps)} connecting edges")
                    st.graphviz_chart(graphviz_from_edges(eps[:120], lookup_titles({u for u, v in eps} | {v for u, v in eps})))
                elif op == "graph_neighbors":
                    ns = [int(n) for n in edges_t.graph_neighbors(ia, graph_column="source")]
                    nt = lookup_titles(ns)
                    st.write("Neighbors:", ", ".join(str(nt.get(n, n)) for n in ns[:50]))
                    st.graphviz_chart(graphviz_from_edges([(ia, n) for n in ns[:60]], {**nt, ia: titles.get(ia, ia)}))
                elif op == "subgraph":
                    sg = as_df(edges_t.subgraph(seeds=ids, hops=2, directed=False)).head(150)
                    st.graphviz_chart(graphviz_from_edges(
                        list(zip(sg["source"].astype("int64"), sg["target"].astype("int64"))),
                        lookup_titles(set(sg["source"].astype("int64")) | set(sg["target"].astype("int64")))))
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")
