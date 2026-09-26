"""BenoStreamDB — Plotly Dash demo (side-by-side with the Streamlit app).

Same engine, same data, same panels as ``examples/web_ui/app.py`` — but built
with Dash's native live-update model:

  * a background worker thread runs the (blocking) engine call and pushes trace
    events + resource samples into a shared :class:`RunState`;
  * a ``dcc.Interval`` fires a callback every 500 ms that drains the queue and
    repaints the trace, the process-resource cards, and (when the run finishes)
    the results.

Because the engine call never runs on the request thread, the interval keeps
ticking during the query and after it — no script-thread workarounds.

Run:  python examples/web_ui/dash_app.py   (serves on http://localhost:8050)
"""

import os
import queue
import threading
import time

import pandas as pd
from dash import Dash, Input, Output, State, callback, dash_table, dcc, html

import benostreamdb

# ── Config (mirrors the Streamlit demo) ─────────────────────────────────────
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DB = os.environ.get("BSDB_DEMO_DB", os.path.join(REPO, "data", "wiki_graph_db"))
EDGES_URI = f"file://{os.path.join(DB, 'edges')}"
NODES_URI = f"file://{os.path.join(DB, 'nodes')}"
EMBED_MODEL = os.environ.get("BSDB_DEMO_EMBED_MODEL", "all-MiniLM-L6-v2")

# ── Tables (opened once) ────────────────────────────────────────────────────
edges_t = benostreamdb.Table(EDGES_URI)
nodes_t = benostreamdb.Table(NODES_URI)

_embedder = None
_embedder_failed = False
_embedder_lock = threading.Lock()


def get_embedder():
    global _embedder, _embedder_failed
    with _embedder_lock:
        if _embedder is None and not _embedder_failed:
            try:
                from sentence_transformers import SentenceTransformer
                _embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
            except Exception as e:  # pragma: no cover - optional dependency
                print(f"embedder unavailable: {e}")
                _embedder_failed = True
    return _embedder


def embed_text(text: str, model=None):
    m = model if model is not None else get_embedder()
    if m is None:
        return None
    return m.encode([text], normalize_embeddings=True)[0].tolist()


# ── Process resource sampling ───────────────────────────────────────────────
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
    rss_kb, peak_kb = _proc_status()
    read_b, write_b = _proc_io()
    return {
        "rss_mb": rss_kb / 1024.0,
        "peak_mb": peak_kb / 1024.0,
        "read_mb": read_b / 1e6,
        "write_mb": write_b / 1e6,
        "cpu_s": _proc_cpu_seconds(),
    }


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:,.1f} TB"


def _dir_sizes(path: str):
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


def graph_summary() -> dict:
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


# ── Run state: worker thread + event queue ──────────────────────────────────
class RunState:
    """A query running on a worker thread, streaming trace events to the UI."""

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


RUNS: dict[str, RunState] = {}


# ── Rendering helpers ───────────────────────────────────────────────────────
def _card(label: str, value: str, delta: str | None = None):
    children = [html.Div(label, className="metric-label"),
                html.Div(value, className="metric-value")]
    if delta is not None:
        children.append(html.Div(delta, className="metric-delta"))
    return html.Div(children, className="metric-card")


def resource_cards(base: dict):
    s = sample_resources()
    return html.Div([
        _card("RSS", f"{s['rss_mb']:.1f} MB", f"{s['rss_mb'] - base['rss_mb']:+.1f} MB"),
        _card("Peak RSS", f"{s['peak_mb']:.1f} MB"),
        _card("Disk read", f"{s['read_mb']:.1f} MB", f"{s['read_mb'] - base['read_mb']:+.1f} MB"),
        _card("Disk write", f"{s['write_mb']:.1f} MB", f"{s['write_mb'] - base['write_mb']:+.1f} MB"),
        _card("CPU time", f"{s['cpu_s']:.2f} s", f"{s['cpu_s'] - base['cpu_s']:+.2f} s"),
    ], className="metric-row")


def trace_panel(run: RunState):
    return html.Pre("\n".join(run.lines) or "…", className="trace")


def bottleneck(run: RunState):
    if not run.stages:
        return html.Div()
    total = sum(ms for _, ms in run.stages)
    ranked = sorted(run.stages, key=lambda x: x[1], reverse=True)
    slowest, slow_ms = ranked[0]
    share = (slow_ms / total * 100) if total else 0.0
    return html.Div([
        html.Div(f"Bottleneck breakdown — {total:.1f} ms across {len(run.stages)} stages",
                 className="section-title"),
        html.Div([html.Span("Bottleneck: ", className="bottleneck-label"),
                  html.Code(f"{slowest} — {slow_ms:.1f} ms ({share:.0f}% of traced time)")]),
        dcc.Graph(
            figure={
                "data": [{"type": "bar", "orientation": "h",
                          "x": [ms for _, ms in ranked][::-1],
                          "y": [s for s, _ in ranked][::-1],
                          "marker": {"color": "#4c78a8"}}],
                "layout": {"height": max(120, 30 * len(ranked) + 60),
                           "margin": {"l": 10, "r": 10, "t": 10, "b": 10},
                           "xaxis": {"title": "ms"}},
            },
            config={"displayModeBar": False},
        ),
    ])


def documents(ids, heading: str = "Retrieved documents", limit: int = 10):
    ids = [int(i) for i in ids][:limit]
    if not ids:
        return html.Div()
    id_list = ", ".join(str(i) for i in ids)
    df = nodes_t.execute_sql(
        f"SELECT id, title, summary FROM t WHERE id IN ({id_list})")
    df = df.to_pandas() if hasattr(df, "to_pandas") else pd.DataFrame(df)
    if df.empty:
        return html.Div()
    order = {i: r for r, i in enumerate(ids)}
    df = df.assign(_o=df["id"].astype(int).map(order)).sort_values("_o")
    items = []
    for _, r in df.iterrows():
        items.append(html.Details([
            html.Summary(f"{r['title']}  ·  id {int(r['id'])}"),
            html.Div((str(r["summary"]) or "No summary stored.")[:600], className="doc-body"),
            html.A("Wikipedia", href=f"https://en.wikipedia.org/wiki/?curid={int(r['id'])}",
                   target="_blank"),
        ]))
    return html.Div([html.Div(heading, className="section-title")] + items)


def lookup_titles(ids):
    ids = [int(i) for i in ids]
    if not ids:
        return {}
    id_list = ", ".join(str(i) for i in ids)
    df = nodes_t.execute_sql(f"SELECT id, title FROM t WHERE id IN ({id_list})")
    df = df.to_pandas() if hasattr(df, "to_pandas") else pd.DataFrame(df)
    return {int(r["id"]): str(r["title"]) for _, r in df.iterrows()}


def find_pages(term, limit=5):
    safe = term.replace("'", "''").replace("%", r"\%")
    try:
        df = nodes_t.execute_sql(
            f"SELECT id, title FROM t WHERE lower(title) LIKE lower('%{safe}%') LIMIT {limit}")
        return df.to_pandas() if hasattr(df, "to_pandas") else pd.DataFrame(df)
    except Exception:
        return pd.DataFrame()


def _csr_or_sql(csr_fn, sql_fn, what):
    """Run the CSR fast path; fall back to SQL BFS if the graph index isn't
    ready yet (the one-time v1→v2 migration runs in the background on open)."""
    try:
        return csr_fn(), None
    except Exception as e:
        if "CSR graph index" in str(e):
            return sql_fn(), f"{what}: CSR index not ready (background migration) — SQL BFS fallback."
        raise


def explain_block(run: RunState, label: str, vector_filter: dict):
    if "explain" not in run.result:
        try:
            run.result["explain"] = nodes_t.explain(vector_filter=vector_filter)
        except Exception as e:
            run.result["explain"] = f"explain failed ({type(e).__name__}: {e})"
    return html.Details([
        html.Summary(f"Query plan — {label}"),
        html.Pre(run.result["explain"], className="plan"),
    ])


def _df_table(df, height="360px"):
    return dash_table.DataTable(
        data=df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in df.columns],
        page_size=15,
        style_table={"height": height, "overflowY": "auto"},
        style_cell={"fontFamily": "monospace", "fontSize": 12, "textAlign": "left"},
        style_header={"fontWeight": "bold"},
    )


# ── App layout ──────────────────────────────────────────────────────────────
_sum = graph_summary()
_summary_text = (
    f"Loaded graph: {_sum['pages']:,} pages · {_sum['edges']:,} edges · "
    f"{_sum['segments']} segments · data {_fmt_bytes(_sum['data_bytes'])} · "
    f"indexes {_fmt_bytes(_sum['index_bytes'])} · on disk {_fmt_bytes(_sum['disk_bytes'])}"
)

app = Dash(__name__, title="BenoStreamDB — Dash demo")
app.layout = html.Div([
    html.H2("BenoStreamDB — Wikipedia Graph RAG (Dash)"),
    html.Div(_summary_text, className="summary"),
    html.Div(id="global-res", className="global-res"),
    dcc.Interval(id="global-interval", interval=1000),

    dcc.Tabs([
        dcc.Tab(label="Semantic search", children=[
            html.Div("Hybrid retrieval — BM25 + dense vectors fused via RRF",
                     className="section-title"),
            dcc.Input(id="sem-q", type="text", placeholder="what causes auroras?",
                      style={"width": "60%"}),
            html.Span("  Top k: "),
            dcc.Slider(id="sem-k", min=5, max=50, step=5, value=10,
                       marks={5: "5", 25: "25", 50: "50"}),
            html.Button("Search", id="sem-go", n_clicks=0, className="go"),
            dcc.Store(id="sem-run"),
            dcc.Interval(id="sem-interval", interval=500, disabled=True),
            html.Div(id="sem-trace-wrap", children=[html.Div("Engine trace", className="section-title"),
                                                    html.Div(id="sem-trace")]),
            html.Div(id="sem-res"),
            html.Div(id="sem-results"),
        ]),

        dcc.Tab(label="Graph RAG (local)", children=[
            html.Div("Local Graph RAG — seed discovery, multi-hop subgraph, Personalized PageRank",
                     className="section-title"),
            dcc.Input(id="gr-q", type="text",
                      placeholder="How are neural networks connected to the Turing test?",
                      style={"width": "60%"}),
            html.Div([
                html.Span("Seeds: "),
                dcc.Input(id="gr-topk", type="number", value=5, min=1, max=20, step=1),
                html.Span("  Hops: "),
                dcc.Input(id="gr-hops", type="number", value=2, min=1, max=3, step=1),
                html.Span("  Rerank top-N: "),
                dcc.Input(id="gr-rerank", type="number", value=8, min=1, max=20, step=1),
            ]),
            html.Button("Run Graph RAG", id="gr-go", n_clicks=0, className="go"),
            dcc.Store(id="gr-run"),
            dcc.Interval(id="gr-interval", interval=500, disabled=True),
            html.Div(id="gr-trace-wrap", children=[html.Div("Engine trace", className="section-title"),
                                                   html.Div(id="gr-trace")]),
            html.Div(id="gr-res"),
            html.Div(id="gr-results"),
        ]),

        dcc.Tab(label="Browse", children=[
            html.Div("Browse the 51M-page corpus (keyset pagination, engine-side)",
                     className="section-title"),
            html.Button("First page", id="browse-first", n_clicks=0),
            html.Button("Next page", id="browse-next", n_clicks=0),
            html.Button("Previous page", id="browse-prev", n_clicks=0),
            dcc.Input(id="browse-jump", type="text", placeholder="Jump to title…"),
            dcc.Store(id="browse-cursor", data=-1),
            dcc.Store(id="browse-stack", data=[]),
            html.Div(id="browse-table"),
            html.Div(id="browse-detail"),
        ]),

        dcc.Tab(label="DRIFT (regional)", children=[
            html.Div("DRIFT — regional community summaries + iterative deepening search",
                     className="section-title"),
            dcc.Input(id="drift-q", type="text", placeholder="what is the history of computing?",
                      style={"width": "60%"}),
            html.Span("  Region seeds: "),
            dcc.Slider(id="drift-seeds", min=2, max=10, step=1, value=4,
                       marks={2: "2", 6: "6", 10: "10"}),
            html.Button("Build region + search", id="drift-go", n_clicks=0, className="go"),
            html.Div(id="drift-out"),
        ]),

        dcc.Tab(label="Graph traversals", children=[
            html.Div("Native CSR traversals — microsecond adjacency over memory-mapped index",
                     className="section-title"),
            html.Span("Entity A: "),
            dcc.Input(id="trav-a", type="text", value="Artificial intelligence"),
            html.Span("  Entity B: "),
            dcc.Input(id="trav-b", type="text", value="Alan Turing"),
            html.Span("  Operation: "),
            dcc.Dropdown(id="trav-op", value="shortest_path",
                         options=[{"label": o, "value": o} for o in
                                  ["shortest_path", "connecting_paths", "graph_neighbors", "subgraph"]],
                         clearable=False, style={"width": "220px", "display": "inline-block"}),
            html.Button("Run traversal", id="trav-go", n_clicks=0, className="go"),
            html.Div(id="trav-out"),
        ]),
    ]),
], className="app")


# ── Global resource panel ───────────────────────────────────────────────────
@callback(Output("global-res", "children"), Input("global-interval", "n_intervals"))
def _global_res(_n):
    r = sample_resources()
    return html.Div([
        html.Span(f"RSS {r['rss_mb']:.0f} MB · peak {r['peak_mb']:.0f} MB · "
                  f"CPU {r['cpu_s']:.1f} s · {time.strftime('%H:%M:%S')}",
                  className="global-res-text"),
    ])


# ── Semantic search ─────────────────────────────────────────────────────────
@callback(
    Output("sem-run", "data"),
    Output("sem-interval", "disabled"),
    Input("sem-go", "n_clicks"),
    State("sem-q", "value"),
    State("sem-k", "value"),
    prevent_initial_call=True,
)
def _start_sem(_n, q, k):
    if not q:
        return None, True
    run = RunState("Semantic search trace")
    run_id = f"sem-{time.time()}"
    RUNS[run_id] = run
    model = get_embedder()

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
    return run_id, False


@callback(
    Output("sem-trace", "children"),
    Output("sem-res", "children"),
    Output("sem-results", "children"),
    Output("sem-interval", "disabled", allow_duplicate=True),
    Input("sem-interval", "n_intervals"),
    State("sem-run", "data"),
    prevent_initial_call=True,
)
def _poll_sem(_n, run_id):
    run = RUNS.get(run_id)
    if run is None:
        return "", "", "", True
    run.drain()
    trace = trace_panel(run)
    res = resource_cards(run.base_res)
    if not run.done.is_set():
        return trace, res, "", False

    k = run.result.get("k", 10)
    vec = run.result.get("vec")
    search_res = run.result.get("res")
    df = (search_res.to_pandas() if hasattr(search_res, "to_pandas")
          else pd.DataFrame(search_res if search_res is not None else []))
    blocks = [html.Div(f"Mode: {run.result.get('mode')}", className="caption")]
    if df is None or df.empty:
        blocks.append(html.Div("No results."))
    else:
        show = df[[c for c in ["id", "title", "rrf_score"] if c in df.columns]]
        blocks.append(_df_table(show.head(k)))
        ids = [int(i) for i in df["id"].tolist()] if "id" in df.columns else []
        if vec is not None:
            blocks.append(explain_block(run, "hybrid vector leg (HNSW-TQ)",
                                        {"column": "embedding", "query": vec, "k": k}))
        if ids:
            blocks.append(documents(ids))
    blocks.append(bottleneck(run))
    blocks.append(html.Div(f"End-to-end: {(time.time() - run.t0) * 1000:.1f} ms",
                           className="caption"))
    return trace, res, html.Div(blocks), True


# ── Graph RAG ───────────────────────────────────────────────────────────────
@callback(
    Output("gr-run", "data"),
    Output("gr-interval", "disabled"),
    Input("gr-go", "n_clicks"),
    State("gr-q", "value"),
    State("gr-topk", "value"),
    State("gr-hops", "value"),
    State("gr-rerank", "value"),
    prevent_initial_call=True,
)
def _start_gr(_n, q, top_k, hops, rerank_n):
    if not q:
        return None, True
    run = RunState("Graph RAG trace")
    run.result["top_k"] = int(top_k or 5)
    run.result["rerank_n"] = int(rerank_n or 8)
    run_id = f"gr-{time.time()}"
    RUNS[run_id] = run
    model = get_embedder()

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
                id_column="id", top_k=int(top_k or 5), hops=int(hops or 2),
            )
        except Exception as e:
            r.error = e
        r.stage("graph_rag_search (seeds+CSR+PPR)", (time.time() - t1) * 1000)

    run.start(work)
    return run_id, False


@callback(
    Output("gr-trace", "children"),
    Output("gr-res", "children"),
    Output("gr-results", "children"),
    Output("gr-interval", "disabled", allow_duplicate=True),
    Input("gr-interval", "n_intervals"),
    State("gr-run", "data"),
    prevent_initial_call=True,
)
def _poll_gr(_n, run_id):
    run = RUNS.get(run_id)
    if run is None:
        return "", "", "", True
    run.drain()
    trace = trace_panel(run)
    res = resource_cards(run.base_res)
    if not run.done.is_set():
        return trace, res, "", False

    top_k = run.result.get("top_k", 5)
    rerank_n = run.result.get("rerank_n", 8)
    vec = run.result.get("vec")
    result = run.result.get("result")
    blocks = []
    ids = []
    if run.error is not None:
        blocks.append(html.Div(f"{type(run.error).__name__}: {run.error}", className="error"))
    elif result is not None:
        seeds = [int(s) for s in (result.seeds or [])]
        blocks.append(html.Div(f"Seeds: {seeds}", className="caption"))
        ctx = result.nodes
        ctx_df = ctx.to_pandas() if hasattr(ctx, "to_pandas") else pd.DataFrame(ctx)
        if not ctx_df.empty:
            blocks.append(_df_table(ctx_df.head(30), height="260px"))
        ids = seeds + ([int(x) for x in ctx_df["id"]] if "id" in ctx_df.columns else [])
        if vec is not None and not ctx_df.empty and "id" in ctx_df.columns:
            pool = [int(x) for x in ctx_df["id"].tolist()]
            try:
                rr = nodes_t.vector_search(
                    "embedding", vec, k=min(int(rerank_n), len(pool)),
                    filter=f"id IN ({', '.join(map(str, pool))})",
                    columns=["id", "title"])
                rr = rr.to_pandas() if hasattr(rr, "to_pandas") else pd.DataFrame(rr)
                if not rr.empty:
                    blocks.append(html.Div("Bitmap-filtered rerank (topology ∩ semantics)",
                                           className="section-title"))
                    blocks.append(_df_table(rr, height="220px"))
            except Exception as e:
                blocks.append(html.Div(f"Rerank skipped ({type(e).__name__}: {e})",
                                       className="caption"))
    if ids:
        blocks.append(documents(ids, "Subgraph documents"))
    if vec is not None:
        blocks.append(explain_block(run, "graph RAG seed vector leg (HNSW-TQ)",
                                    {"column": "embedding", "query": vec, "k": int(top_k)}))
    blocks.append(bottleneck(run))
    blocks.append(html.Div(f"End-to-end: {(time.time() - run.t0) * 1000:.1f} ms",
                           className="caption"))
    return trace, res, html.Div(blocks), True


# ── Browse ──────────────────────────────────────────────────────────────────
@callback(
    Output("browse-cursor", "data"),
    Output("browse-stack", "data"),
    Input("browse-first", "n_clicks"),
    Input("browse-next", "n_clicks"),
    Input("browse-prev", "n_clicks"),
    Input("browse-jump", "value"),
    State("browse-cursor", "data"),
    State("browse-stack", "data"),
    prevent_initial_call=True,
)
def _browse_nav(_f, _n, _p, jump, cursor, stack):
    from dash import ctx
    trig = ctx.triggered_id
    stack = stack or []
    cursor = -1 if cursor is None else int(cursor)
    if trig == "browse-first":
        return -1, []
    if trig == "browse-next":
        page = nodes_t.execute_sql(
            f"SELECT id FROM t WHERE id > {cursor} ORDER BY id LIMIT 15")
        page = page.to_pandas() if hasattr(page, "to_pandas") else pd.DataFrame(page)
        new_cursor = int(page["id"].iloc[-1]) if not page.empty else cursor
        return new_cursor, stack + [cursor]
    if trig == "browse-prev":
        return (stack[-1], stack[:-1]) if stack else (cursor, stack)
    if trig == "browse-jump" and jump:
        hit = find_pages(jump, 1)
        if not hit.empty:
            return int(hit.iloc[0]["id"]) - 1, []
    return cursor, stack


@callback(Output("browse-table", "children"), Input("browse-cursor", "data"))
def _browse_table(cursor):
    cursor = -1 if cursor is None else int(cursor)
    page = nodes_t.execute_sql(
        f"SELECT id, title FROM t WHERE id > {cursor} ORDER BY id LIMIT 15")
    page = page.to_pandas() if hasattr(page, "to_pandas") else pd.DataFrame(page)
    if page.empty:
        return html.Div("End of corpus.")
    return _df_table(page, height="380px")


@callback(
    Output("browse-detail", "children"),
    Input("browse-table", "active_cell"),
    State("browse-table", "data"),
)
def _browse_detail(active_cell, data):
    if not active_cell or not data:
        return html.Div()
    pid = int(data[active_cell["row"]]["id"])
    detail = nodes_t.execute_sql(f"SELECT id, title, summary FROM t WHERE id = {pid}")
    detail = detail.to_pandas() if hasattr(detail, "to_pandas") else pd.DataFrame(detail)
    if detail.empty:
        return html.Div()
    r = detail.iloc[0]
    return html.Div([
        html.B(str(r["title"])),
        html.Div(f"id: {int(r['id'])}"),
        html.Div(str(r["summary"])[:1200], className="doc-body"),
    ])


# ── DRIFT ───────────────────────────────────────────────────────────────────
@callback(
    Output("drift-out", "children"),
    Input("drift-go", "n_clicks"),
    State("drift-q", "value"),
    State("drift-seeds", "value"),
    prevent_initial_call=True,
)
def _drift(_n, q, seeds_n):
    if not q:
        return html.Div()
    import shutil
    import pyarrow as pa
    tmp_edges = "/tmp/hdb_dash_region"
    tmp_comm = "file:///tmp/hdb_dash_communities"
    shutil.rmtree(tmp_edges, ignore_errors=True)
    shutil.rmtree("/tmp/hdb_dash_communities", ignore_errors=True)
    out = []
    try:
        vec = embed_text(q)
        hits = nodes_t.vector_search("embedding", vec, k=seeds_n) if vec else find_pages(q, seeds_n)
        hits = hits.to_pandas() if hasattr(hits, "to_pandas") else pd.DataFrame(hits)
        seed_ids = [int(i) for i in hits["id"]][:seeds_n]
        out.append(html.Div(f"Seeds: {[lookup_titles(seed_ids).get(i, i) for i in seed_ids]}"))
        pairs = [(int(s), int(n)) for s in seed_ids
                 for n in edges_t.graph_neighbors(int(s), 1, graph_column="source")]
        region = pd.DataFrame(pairs, columns=["source", "target"]).head(60_000)
        out.append(html.Div(f"Region: {len(region):,} edges"))
        if region.empty:
            out.append(html.Div("Empty region."))
            return html.Div(out)
        schema = pa.schema([("source", pa.int64()), ("target", pa.int64())])
        rt = benostreamdb.Table.create(f"file://{tmp_edges}", schema)
        rt.insert(pa.Table.from_pandas(region.astype("int64"), schema=schema, preserve_index=False))
        rt.commit()
        rt.wait_for_background_tasks()
        comm = rt.summarize_communities(doc_table=nodes_t, target_uri=tmp_comm,
                                        id_column="id", content_column="summary")
        res = rt.drift_search(query=q, community_table=comm, follow_up_llm=None,
                              n_depth=1, k_followups=2, top_k=5, hops=1)
        nodes_found = res.get("all_discovered_nodes", [])
        titles = lookup_titles([int(n) for n in nodes_found][:80])
        out.append(html.Div("Discovered nodes: " + ", ".join(
            str(titles.get(int(n), n)) for n in nodes_found[:40])))
        acts = res.get("actions", [])
        if acts:
            out.append(_df_table(pd.DataFrame(acts).head(20), height="260px"))
    except Exception as e:
        out.append(html.Div(f"{type(e).__name__}: {e}", className="error"))
    return html.Div(out)


# ── Graph traversals ────────────────────────────────────────────────────────
@callback(
    Output("trav-out", "children"),
    Input("trav-go", "n_clicks"),
    State("trav-a", "value"),
    State("trav-b", "value"),
    State("trav-op", "value"),
    prevent_initial_call=True,
)
def _trav(_n, a, b, op):
    ra, rb = find_pages(a, 1), find_pages(b, 1)
    if ra.empty:
        return html.Div(f"Could not resolve '{a}'.", className="error")
    ia = int(ra.iloc[0]["id"])
    ids = [ia]
    if not rb.empty:
        ids.append(int(rb.iloc[0]["id"]))
    titles = lookup_titles(ids)
    out = [html.Div(f"Resolved: {[titles.get(i, i) for i in ids]}")]
    try:
        if op == "shortest_path" and len(ids) == 2:
            path, note = _csr_or_sql(
                lambda: [int(n) for n in edges_t.shortest_path(ids[0], ids[1], graph_column="source")],
                lambda: [int(n) for n in edges_t.shortest_path(ids[0], ids[1]).to_pandas()["node"].tolist()],
                "shortest_path")
            if note:
                out.append(html.Div(note, className="caption"))
            pt = lookup_titles(path)
            out.append(html.Div(f"Path ({len(path)} hops): " +
                                " → ".join(str(pt.get(p, p)) for p in path)))
        elif op == "connecting_paths" and len(ids) == 2:
            eps, note = _csr_or_sql(
                lambda: [(int(u), int(v)) for u, v in edges_t.connecting_paths(ids, graph_column="source")],
                lambda: [(int(r["source"]), int(r["target"]))
                         for _, r in edges_t.connecting_paths(ids).to_pandas().iterrows()],
                "connecting_paths")
            if note:
                out.append(html.Div(note, className="caption"))
            out.append(html.Div(f"{len(eps)} connecting edges"))
        elif op == "graph_neighbors":
            ns, note = _csr_or_sql(
                lambda: [int(n) for n in edges_t.graph_neighbors(ia, graph_column="source")],
                lambda: [int(n) for n in edges_t.graph_neighbors(ia).to_pandas()["neighbor"].tolist()],
                "graph_neighbors")
            if note:
                out.append(html.Div(note, className="caption"))
            nt = lookup_titles(ns)
            out.append(html.Div("Neighbors: " + ", ".join(str(nt.get(n, n)) for n in ns[:50])))
        elif op == "subgraph":
            sg = edges_t.subgraph(seeds=ids, hops=2, directed=False)
            sg = sg.to_pandas() if hasattr(sg, "to_pandas") else pd.DataFrame(sg)
            out.append(html.Div(f"{len(sg)} subgraph edges"))
    except Exception as e:
        out.append(html.Div(f"{type(e).__name__}: {e}", className="error"))
    return html.Div(out)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
