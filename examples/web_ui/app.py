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
import queue
import threading
from contextlib import contextmanager

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


def embed_text(text: str, model=None):
    # `model` is passed in when embedding runs on a worker thread: `get_embedder`
    # is an `st.cache_resource`, which needs the script context, so it must be
    # resolved on the script thread and handed to the worker.
    m = model if model is not None else get_embedder()
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
    show_trace = st.toggle("Show engine trace", value=True)
    show_explain = st.toggle("Show query plan (explain)", value=True)
    if st.button("Clear caches"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

edges_t, nodes_t = open_tables()


# ── Startup summary: what's loaded ──────────────────────────────────────────
def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:,.1f} TB"


def _dir_sizes(path: str) -> tuple[int, int]:
    """(total_bytes, index_bytes) on disk for a table directory. Index bytes are
    everything that isn't a data `.parquet` file — the CSR / HNSW / inverted
    sidecars. The engine's `total_index_size_bytes` is not populated, so we read
    the filesystem directly."""
    total = index = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                sz = os.path.getsize(os.path.join(root, f))
            except OSError:
                continue
            total += sz
            if not f.endswith(".parquet"):
                index += sz
    return total, index


@st.cache_data(ttl=600, show_spinner=False)
def graph_summary() -> dict:
    """Row counts, segment counts, and on-disk data/index sizes for the two
    tables backing the demo."""
    e = edges_t.statistics
    n = nodes_t.statistics
    e_total, e_idx = _dir_sizes(os.path.join(DB, "edges"))
    n_total, n_idx = _dir_sizes(os.path.join(DB, "nodes"))
    return {
        "pages": n.row_count,
        "edges": e.row_count,
        "segments": n.file_count + e.file_count,
        "data_bytes": n.total_size_bytes + e.total_size_bytes,
        "index_bytes": e_idx + n_idx,
        "disk_bytes": e_total + n_total,
    }


try:
    _s = graph_summary()
    st.markdown(
        f"**Loaded graph: {_s['pages']:,} pages · {_s['edges']:,} edges · "
        f"{_s['segments']} segments · data {_fmt_bytes(_s['data_bytes'])} · "
        f"indexes {_fmt_bytes(_s['index_bytes'])} · on disk {_fmt_bytes(_s['disk_bytes'])}**"
    )
except Exception as _e:
    st.caption(f"Graph summary unavailable ({type(_e).__name__}: {_e})")


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


# ── Live trace, query plan (explain), and document text ─────────────────────
def _scroll_log_to_bottom() -> None:
    """Best-effort auto-scroll of the newest log panel to its latest line."""
    try:
        st.html(
            """
            <script>
            (function () {
              const doc = window.parent.document;
              const codes = doc.querySelectorAll('[data-testid="stCode"]');
              if (!codes.length) return;
              const el = codes[codes.length - 1];
              let p = el.parentElement;
              while (p && p !== doc.body && p.scrollHeight <= p.clientHeight + 4) {
                p = p.parentElement;
              }
              if (p && p !== doc.body) p.scrollTop = p.scrollHeight;
            })();
            </script>
            """,
            unsafe_allow_javascript=True,
        )
    except Exception:
        pass


def _proc_status():
    rss = peak = 0
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1])
                elif line.startswith("VmHWM:"):
                    peak = int(line.split()[1])
    except Exception:
        pass
    return rss, peak


def _proc_io():
    r = w = 0
    try:
        with open("/proc/self/io") as f:
            for line in f:
                if line.startswith("read_bytes:"):
                    r = int(line.split()[1])
                elif line.startswith("write_bytes:"):
                    w = int(line.split()[1])
    except Exception:
        pass
    return r, w


def _proc_cpu_seconds():
    try:
        with open("/proc/self/stat") as f:
            parts = f.read().split()
        return (int(parts[13]) + int(parts[14])) / os.sysconf("SC_CLK_TCK")
    except Exception:
        return 0.0


def sample_resources() -> dict:
    """Snapshot this process's memory, disk I/O, and CPU usage."""
    rss_kb, peak_kb = _proc_status()
    read_b, write_b = _proc_io()
    return {
        "rss_mb": rss_kb / 1024.0,
        "peak_mb": peak_kb / 1024.0,
        "read_mb": read_b / 1e6,
        "write_mb": write_b / 1e6,
        "cpu_s": _proc_cpu_seconds(),
    }


def disk_headroom(path: str):
    """(total, used, free) bytes for the filesystem holding ``path``."""
    try:
        return shutil.disk_usage(path)
    except Exception:
        return None


@st.fragment(run_every=1.0)
def sidebar_resources() -> None:
    """Always-on sidebar process readout (re-samples every second)."""
    r = sample_resources()
    st.caption(f"RSS {r['rss_mb']:.0f} MB · peak {r['peak_mb']:.0f} MB · "
               f"CPU {r['cpu_s']:.1f} s · {time.strftime('%H:%M:%S')}")
    st.caption(f"Disk I/O: read {r['read_mb']:.1f} MB · write {r['write_mb']:.1f} MB")
    dh = disk_headroom(DB)
    if dh:
        st.caption(f"DB disk: {dh[2] / 1e9:.1f} GB free / {dh[0] / 1e9:.1f} GB")


def _render_resources(base: dict) -> None:
    """Render the live process-resource metrics against a base snapshot."""
    s = sample_resources()
    with st.container(border=True):
        st.caption(f"Process resources  ·  live  ·  {time.strftime('%H:%M:%S')}")
        cols = st.columns(5)
        cols[0].metric("RSS", f"{s['rss_mb']:.1f} MB",
                       f"{s['rss_mb'] - base['rss_mb']:+.1f} MB")
        cols[1].metric("Peak RSS", f"{s['peak_mb']:.1f} MB")
        cols[2].metric("Disk read", f"{s['read_mb']:.1f} MB",
                       f"{s['read_mb'] - base['read_mb']:+.1f} MB")
        cols[3].metric("Disk write", f"{s['write_mb']:.1f} MB",
                       f"{s['write_mb'] - base['write_mb']:+.1f} MB")
        cols[4].metric("CPU time", f"{s['cpu_s']:.2f} s",
                       f"{s['cpu_s'] - base['cpu_s']:+.2f} s")
        dh = disk_headroom(DB)
        if dh:
            total, used, free = dh
            st.caption(
                f"DB filesystem: {free / 1e9:.1f} GB free / {total / 1e9:.1f} GB "
                f"({used / 1e9:.1f} GB used)")


class LiveRun:
    """A query running on a worker thread, streaming trace events to the UI.

    The engine call runs off Streamlit's script thread, so the script thread
    stays free and a self-refreshing fragment can drain the event queue and
    repaint the trace + resource panels every half second — during the query and
    after it. Results are rendered by the main script once ``done`` is set.
    """

    def __init__(self, title: str):
        self.title = title
        self.events: queue.Queue = queue.Queue()
        self.result: dict = {}
        self.error: BaseException | None = None
        self.done = threading.Event()
        self.t0 = time.time()
        self.lines: list[str] = []
        self.stages: list[tuple[str, float]] = []
        self.base_res = sample_resources()

    def log(self, msg: str) -> None:
        self.events.put(("log", msg))

    def stage(self, label: str, ms: float) -> None:
        self.events.put(("stage", (label, ms)))

    def start(self, work) -> None:
        def runner():
            try:
                work(self)
            except BaseException as e:  # surfaced in the UI
                self.error = e
            finally:
                self.done.set()
        threading.Thread(target=runner, daemon=True).start()

    def drain(self) -> None:
        while True:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                return
            if kind == "log":
                self.lines.append(f"{time.time() - self.t0:7.3f}s  {payload}")
            elif kind == "stage":
                label, ms = payload
                self.stages.append((label, ms))
                self.lines.append(f"{time.time() - self.t0:7.3f}s  ✓ {label}  ({ms:.1f} ms)")

    def summary(self) -> None:
        """Rank stages by wall time and call out the dominant one."""
        if not self.stages:
            return
        total = sum(ms for _, ms in self.stages)
        ranked = sorted(self.stages, key=lambda x: x[1], reverse=True)
        slowest, slow_ms = ranked[0]
        share = (slow_ms / total * 100) if total else 0.0
        st.markdown(f"**Bottleneck breakdown** — {total:.1f} ms across {len(self.stages)} stages")
        st.markdown(f":orange[**Bottleneck:**] `{slowest}` — {slow_ms:.1f} ms "
                    f"({share:.0f}% of traced time)")
        st.bar_chart(pd.DataFrame({"ms": [ms for _, ms in ranked]},
                                  index=[s for s, _ in ranked]))
        s = sample_resources()
        b = self.base_res
        st.caption(
            f"Resource deltas over the run: RSS {s['rss_mb'] - b['rss_mb']:+.0f} MB · "
            f"disk read {s['read_mb'] - b['read_mb']:+.1f} MB · "
            f"disk write {s['write_mb'] - b['write_mb']:+.1f} MB")


@st.fragment(run_every=0.5)
def live_run_panel(run_id: str, height: int = 240) -> None:
    """Self-refreshing trace + resource panel for an in-flight :class:`LiveRun`.

    Drains the run's event queue and repaints every half second. When the run
    finishes it triggers a full rerun so the main script can render results.
    """
    run = st.session_state.get(run_id)
    if run is None:
        return
    run.drain()
    with st.container(height=height, border=True):
        st.caption(run.title)
        st.code("\n".join(run.lines) or "…", language="text")
    _scroll_log_to_bottom()
    _render_resources(run.base_res)
    if run.done.is_set():
        st.rerun()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_documents(ids):
    """Fetch the stored document text (Wikipedia summary) for node ids."""
    if not ids:
        return pd.DataFrame()
    id_list = ", ".join(str(int(i)) for i in ids)
    return as_df(nodes_t.execute_sql(
        f"SELECT id, title, summary FROM t WHERE id IN ({id_list})"))


def render_documents(ids, heading: str = "Retrieved documents", limit: int = 10) -> None:
    """Expose the actual document text behind a set of result ids."""
    ids = [int(i) for i in ids][:limit]
    if not ids:
        return
    df = fetch_documents(ids)
    if df.empty:
        return
    order = {i: r for r, i in enumerate(ids)}
    df = df.assign(_o=df["id"].astype(int).map(order)).sort_values("_o")
    st.markdown(f"**{heading}**")
    for _, r in df.iterrows():
        with st.expander(f"{r['title']}  ·  id {int(r['id'])}"):
            st.markdown(str(r["summary"]) or "_No summary stored._")
            st.caption(f"[Wikipedia](https://en.wikipedia.org/wiki/?curid={int(r['id'])})")


def render_explain(table, label: str, filter: str | None = None,
                   vector_filter: dict | None = None) -> None:
    """Render the engine's query plan: pruning activity + index access paths."""
    t = time.time()
    try:
        plan = table.explain(filter=filter, vector_filter=vector_filter)
    except Exception as e:
        st.warning(f"explain failed ({type(e).__name__}: {e})")
        return
    with st.expander(f"Query plan — {label}  ({(time.time() - t) * 1000:.1f} ms)"):
        st.code(plan, language="text")


def _render_semantic_results(run: "LiveRun") -> None:
    """Render the results of a completed semantic-search :class:`LiveRun`."""
    k = st.session_state.get("sem_k", 10)
    vec = run.result.get("vec")
    res = run.result.get("res")
    mode = run.result.get("mode")
    df = as_df(res)
    st.caption(f"Mode: {mode}")
    if df is None or df.empty:
        st.info("No results.")
    else:
        show = df[[c for c in ["id", "title", "rrf_score"] if c in df.columns]]
        st.dataframe(show.head(k), hide_index=True, height=420)
        ids = [int(i) for i in df["id"].tolist()] if "id" in df.columns else []
        if show_explain and vec is not None:
            if "explain" not in run.result:
                try:
                    run.result["explain"] = nodes_t.explain(
                        vector_filter={"column": "embedding", "query": vec, "k": k})
                except Exception as e:
                    run.result["explain"] = f"explain failed ({type(e).__name__}: {e})"
            with st.expander("Query plan — hybrid vector leg (HNSW-TQ)"):
                st.code(run.result["explain"], language="text")
        if ids:
            render_documents(ids, "Retrieved documents")
    run.summary()
    st.caption(f"End-to-end: {(time.time() - run.t0) * 1000:.1f} ms")


def _render_graphrag_results(run: "LiveRun") -> None:
    """Render the results of a completed Graph RAG :class:`LiveRun`."""
    top_k = run.result.get("top_k", 5)
    rerank_n = run.result.get("rerank_n", 8)
    vec = run.result.get("vec")
    result = run.result.get("result")
    ids: list[int] = []
    context = ""
    if run.error is not None:
        st.error(f"{type(run.error).__name__}: {run.error}")
    elif result is not None:
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
                    st.dataframe(show, hide_index=True, height=min(40 + 35 * len(show), 350))
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
    if ids:
        render_documents(ids, "Subgraph documents")
    if show_explain and vec is not None:
        if "explain" not in run.result:
            try:
                run.result["explain"] = nodes_t.explain(
                    vector_filter={"column": "embedding", "query": vec, "k": int(top_k)})
            except Exception as e:
                run.result["explain"] = f"explain failed ({type(e).__name__}: {e})"
        with st.expander("Query plan — graph RAG seed vector leg (HNSW-TQ)"):
            st.code(run.result["explain"], language="text")
    run.summary()
    st.caption(f"End-to-end: {(time.time() - run.t0) * 1000:.1f} ms")
    if context and llm_on:
        with st.spinner("Synthesizing answer with LLM..."):
            try:
                ans = llm_chat([{"role": "user", "content":
                    "Answer the question using ONLY the retrieved context. Cite page titles.\n\n"
                    f"Question: {st.session_state.get('gr_q', '')}\n\nContext:\n{context[:12000]}"}],
                    temperature=0.3)
                st.markdown("### Answer")
                st.markdown(ans)
            except Exception as e:
                st.warning(f"LLM unavailable ({e}). Raw context shown above.")
    elif context:
        st.markdown("#### Retrieved context")
        st.text(context[:4000])


def _csr_or_sql(csr_fn, sql_fn, what: str):
    """Run the CSR fast path; if the graph index isn't available yet — the
    one-time v1→v2 graph-index migration runs in the background on first open —
    fall back to the SQL BFS path and say so, instead of erroring."""
    try:
        return csr_fn()
    except Exception as e:
        if "CSR graph index" in str(e):
            st.info(f"{what}: CSR index not ready yet (background migration in "
                    f"progress) — using the SQL BFS fallback.")
            return sql_fn()
        raise


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


with st.sidebar:
    st.divider()
    st.subheader("Process")
    sidebar_resources()


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
        run = LiveRun("Semantic search trace")
        st.session_state["sem_run"] = run
        model = get_embedder()  # warm on the script thread (st.cache_resource)

        def work(r):
            r.log(f"query={q!r}  k={k}")
            t0 = time.time()
            r.log("▶ embed query (all-MiniLM-L6-v2)")
            vec = embed_text(q, model)
            r.stage("embed query (all-MiniLM-L6-v2)", (time.time() - t0) * 1000)
            r.result["vec"] = vec
            if vec is not None:
                t1 = time.time()
                r.log("▶ hybrid_search: BM25 + HNSW-TQ, RRF fusion")
                try:
                    r.result["res"] = nodes_t.hybrid_search(
                        text_column="title", query_text=q,
                        vector_column="embedding", query_vector=vec, k=k)
                    r.result["mode"] = "hybrid (BM25 + HNSW-TQ, RRF)"
                except Exception as e:
                    r.result["res"], r.result["mode"] = None, f"failed: {e}"
                r.stage("hybrid_search: BM25 + HNSW-TQ, RRF fusion", (time.time() - t1) * 1000)
            else:
                t2 = time.time()
                r.log("▶ keyword fallback (no embedder)")
                safe_q = q.replace("'", "''")
                r.result["res"] = nodes_t.execute_sql(
                    f"SELECT id, title FROM t WHERE lower(title) LIKE lower('%{safe_q}%') LIMIT {k}")
                r.result["mode"] = "keyword only (no embedder)"
                r.stage("keyword fallback (no embedder)", (time.time() - t2) * 1000)

        run.start(work)

    _sem = st.session_state.get("sem_run")
    if _sem is not None:
        if _sem.done.is_set():
            _render_semantic_results(_sem)
        else:
            live_run_panel("sem_run")

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
        run = LiveRun("Graph RAG trace")
        run.result["top_k"] = int(top_k)
        run.result["rerank_n"] = int(rerank_n)
        st.session_state["gr_run"] = run
        model = get_embedder()  # warm on the script thread (st.cache_resource)

        def work(r):
            t0 = time.time()
            r.log("▶ embed query")
            vec = embed_text(q, model)
            r.stage("embed query", (time.time() - t0) * 1000)
            r.result["vec"] = vec
            t1 = time.time()
            r.log("▶ graph_rag_search: vector seeds -> CSR subgraph -> PPR")
            try:
                r.result["result"] = nodes_t.graph_rag_search(
                    query=vec if vec is not None else q,
                    edge_table=edges_t, doc_table=nodes_t,
                    mode="local", vector_column="embedding",
                    id_column="id", top_k=int(top_k), hops=int(hops),
                )
            except Exception as e:
                r.error = e
            r.stage("graph_rag_search (seeds+CSR+PPR)", (time.time() - t1) * 1000)

        run.start(work)

    _gr = st.session_state.get("gr_run")
    if _gr is not None:
        if _gr.done.is_set():
            _render_graphrag_results(_gr)
        else:
            with st.expander("Pipeline", expanded=True):
                live_run_panel("gr_run")

# ── 4. DRIFT (regional communities) ─────────────────────────────────────────
with TAB_DRIFT:
    st.markdown("#### DRIFT — regional community summaries + iterative deepening search")
    q = st.text_input("Query", key="d_q", placeholder="what is the history of computing?")
    seeds_n = st.slider("Region seeds", 2, 10, 4, key="d_seeds")
    if st.button("Build region + search", key="d_go", type="primary") and q:
        t_e2e = time.time()
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
        st.caption(f"End-to-end: {(time.time() - t_e2e) * 1000:.1f} ms")

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
        t_e2e = time.time()
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
                    path = _csr_or_sql(
                        lambda: [int(n) for n in edges_t.shortest_path(ids[0], ids[1], graph_column="source")],
                        lambda: [int(n) for n in as_df(edges_t.shortest_path(ids[0], ids[1]))["node"].tolist()],
                        "shortest_path")
                    pt = lookup_titles(path)
                    st.write(f"Path ({len(path)} hops):", " → ".join(str(pt.get(p, p)) for p in path))
                    st.graphviz_chart(graphviz_from_edges(list(zip(path[:-1], path[1:])), pt))
                elif op == "connecting_paths" and len(ids) == 2:
                    eps = _csr_or_sql(
                        lambda: [(int(u), int(v)) for u, v in edges_t.connecting_paths(ids, graph_column="source")],
                        lambda: [(int(r["source"]), int(r["target"]))
                                 for _, r in as_df(edges_t.connecting_paths(ids)).iterrows()],
                        "connecting_paths")
                    st.write(f"{len(eps)} connecting edges")
                    st.graphviz_chart(graphviz_from_edges(eps[:120], lookup_titles({u for u, v in eps} | {v for u, v in eps})))
                elif op == "graph_neighbors":
                    ns = _csr_or_sql(
                        lambda: [int(n) for n in edges_t.graph_neighbors(ia, graph_column="source")],
                        lambda: [int(n) for n in as_df(edges_t.graph_neighbors(ia))["neighbor"].tolist()],
                        "graph_neighbors")
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
        st.caption(f"End-to-end: {(time.time() - t_e2e) * 1000:.1f} ms")
