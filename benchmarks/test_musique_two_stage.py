#!/usr/bin/env python3
"""Regression checks for the two-stage candidate merge in run_musique_graphrag.py.

These tests cover the pure-Python merge and fusion logic that sits between the
vector index and the cross-encoder. They stub the Arrow result surface so the
suite runs on a host without pyarrow or sentence-transformers:

    python3 benchmarks/test_musique_two_stage.py
"""

import json
import os
import sys
import types

sys.modules.setdefault("pyarrow", types.ModuleType("pyarrow"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import run_musique_graphrag as m  # noqa: E402


class FakeColumn:
    def __init__(self, values):
        self._values = values

    def to_pylist(self):
        return list(self._values)


class FakeTable:
    def __init__(self, columns):
        self.column_names = list(columns)
        self._columns = columns

    def column(self, name):
        return FakeColumn(self._columns[name])


class FakeCorpus:
    """Vector search returns a fixed ranked list; the bulk read returns three rows."""

    def __init__(self, ranked):
        self.ranked = ranked
        self.vector_calls = []

    def to_arrow(self, vector_filter=None, columns=None, filter=None):
        if vector_filter is not None:
            self.vector_calls.append(vector_filter)
            return FakeTable({"id": self.ranked[: vector_filter["k"]]})
        return FakeTable({
            "id": [1, 2, 3],
            "title": ["UHF", "Amblin", "Mike Medavoy"],
            "text": ["1989 film", "studio", "co-founded Orion Pictures"],
        })


class FakeEdges:
    def __init__(self, adjacency):
        self.adjacency = adjacency
        self.filters = []

    def to_arrow(self, filter=None, columns=None):
        self.filters.append(filter)
        predicate = filter or ""
        srcs = [int(x) for x in predicate.split("(")[1].rstrip(")").split(",")]
        targets = []
        for s in srcs:
            targets.extend(self.adjacency.get(s, []))
        return FakeTable({"target": targets})


class BrokenEdges:
    def to_arrow(self, filter=None, columns=None):
        raise RuntimeError("no csr index")


def check(name, cond, detail=None):
    print(f"{'PASS' if cond else 'FAIL'}  {name} {detail}")
    if not cond:
        raise SystemExit(1)


def test_stage1_merge():
    corpus = FakeCorpus([10, 11, 12, 13, 14, 15])
    edges = FakeEdges({10: [99, 20], 11: [99, 21]})
    pool, _ = m.stage1_candidate_pool(corpus, edges, np.zeros(4), top_k=5, seed_k=2, candidates=8)
    check("stage1 requests k equal to the pool cap", corpus.vector_calls[0]["k"] == 8, corpus.vector_calls[0])
    check("stage1 keeps seeds first", pool[:2] == [10, 11], pool)
    check("stage1 merges graph neighbours", {99, 20, 21}.issubset(set(pool)), pool)
    check("stage1 has no duplicates", len(pool) == len(set(pool)), pool)
    check("stage1 respects the pool cap", len(pool) == 8, pool)


def test_stage1_two_hop_expansion():
    """Two hops must reach the passage two entity links away from the seed."""
    corpus = FakeCorpus([10, 11, 12, 13, 14, 15])
    edges = FakeEdges({10: [20], 20: [30]})
    one_hop, _ = m.stage1_candidate_pool(corpus, edges, np.zeros(4), top_k=5, seed_k=1,
                                         candidates=4, hops=1)
    two_hop, _ = m.stage1_candidate_pool(corpus, edges, np.zeros(4), top_k=5, seed_k=1,
                                         candidates=4, hops=2)
    check("1 hop does not reach a depth-2 passage", 30 not in one_hop, one_hop)
    check("2 hops reach a depth-2 passage", 30 in two_hop, two_hop)
    # The pool is floored at top_k, so a candidates=4 request still yields 5 rows.
    check("2-hop pool respects max(top_k, candidates)", len(two_hop) <= 5, two_hop)


def test_stage1_graph_cannot_crowd_out_vectors():
    """One high-degree seed must not evict every non-seed vector candidate."""
    corpus = FakeCorpus(list(range(100, 120)))
    edges = FakeEdges({100: [200, 201, 202, 203, 204, 205, 206, 207, 208, 209]})
    pool, _ = m.stage1_candidate_pool(corpus, edges, np.zeros(4), top_k=5, seed_k=1, candidates=8)
    vector_hits = [pid for pid in pool if pid >= 100]
    graph_hits = [pid for pid in pool if pid >= 200]
    check("stage1 reserves vector slots", len(vector_hits) >= 3, pool)
    check("stage1 still admits graph hits", len(graph_hits) >= 1, pool)


def test_stage1_degrades_to_vector_ranking():
    pool, _ = m.stage1_candidate_pool(FakeCorpus([1, 2, 3]), FakeEdges({}), np.zeros(4),
                                      top_k=5, seed_k=2, candidates=6)
    check("stage1 falls back to vector order", pool == [1, 2, 3], pool)

    pool_broken, _ = m.stage1_candidate_pool(FakeCorpus([7, 8]), BrokenEdges(), np.zeros(4),
                                             top_k=5, seed_k=2, candidates=5)
    check("stage1 tolerates a missing edge index", pool_broken == [7, 8], pool_broken)


def test_stage2_ordering(monkeypatch_scores):
    original = m.reranker_scores
    m.reranker_scores = monkeypatch_scores
    try:
        scores = m.two_stage_rerank("two_stage_minilm", "q", ["a", "b", "c"])
        order = np.argsort(-scores, kind="stable")
        check("stage2 promotes the bridge passage", [["a", "b", "c"][i] for i in order] == ["b", "c", "a"], order.tolist())
    finally:
        m.reranker_scores = original


def test_ensemble_is_scale_invariant():
    original = m.reranker_scores

    def scale_mismatched(backend, query, texts):
        if backend == "minilm":
            return np.array([0.0, 50.0, 100.0])
        return np.array([-9.0, -8.5, -8.4])

    m.reranker_scores = scale_mismatched
    try:
        fused = m.ensemble_rerank_scores("q", ["a", "b", "c"])
        # minilm -> [0, 0.5, 1.0]; bge -> [0, 0.8333, 1.0]. Ordering is preserved even
        # though the raw BGE logits span an order of magnitude less than MiniLM's.
        check("ensemble fuses on a shared scale", np.allclose(fused, [0.0, 1.3333333, 2.0]), fused)
    finally:
        m.reranker_scores = original


def test_dedupe_by_title():
    ids = [5, 6, 7, 8]
    titles = {5: "Sound barrier", 6: "Sound barrier", 7: "Mexico City", 8: "Sound barrier"}
    out = m.dedupe_by_title(ids, titles)
    check("dedupe keeps the first paragraph of an article", out == [5, 7], out)
    check("dedupe preserves order", m.dedupe_by_title([8, 7], titles) == [8, 7])
    check("dedupe tolerates missing titles", m.dedupe_by_title([9, 9], {}) == [9])


def test_ppr_mode_variants():
    check("ppr_rerank maps to its reranker", m.reranker_variant("ppr_rerank_bge") == "bge")
    check("two_stage maps to its reranker", m.reranker_variant("two_stage_ensemble") == "ensemble")
    check("baselines have no reranker", m.reranker_variant("vector_only") == "")
    check("ppr ensemble uses both encoders",
          m.active_reranker_backends("ppr_rerank_ensemble") == ["minilm", "bge"])
    check("ppr minilm uses one encoder", m.active_reranker_backends("ppr_rerank_minilm") == ["minilm"])


def test_graph_metadata(tmp_dir="/tmp/musique_graph_meta"):
    os.makedirs(tmp_dir, exist_ok=True)
    path = m.graph_metadata_path(tmp_dir)
    if os.path.exists(path):
        os.remove(path)
    check("absent graph metadata is not a mismatch", m.graph_mismatch(tmp_dir, "idf") == "")
    m.write_graph_metadata(tmp_dir, "idf", 1234)
    meta = m.read_graph_metadata(tmp_dir)
    check("graph metadata records weighting", meta["edge_weighting"] == "idf", meta)
    check("same weighting is not a mismatch", m.graph_mismatch(tmp_dir, "idf") == "")
    # Requesting binary against an IDF graph must report what is on disk.
    reason = m.graph_mismatch(tmp_dir, "binary")
    check("weighting switch is detected", reason and "idf" in reason, reason)


def test_text_index():
    records = [{"paragraphs": [
        {"title": "UHF", "idx": 0, "paragraph_text": "1989 film", "is_supporting": True},
        {"title": "Amblin", "idx": 1, "paragraph_text": "studio", "is_supporting": False},
    ]}]
    passages = {
        m.passage_key("UHF", "1989 film"): {
            "id": 1, "key": m.passage_key("UHF", "1989 film"), "title": "UHF"},
        m.passage_key("Amblin", "studio"): {
            "id": 2, "key": m.passage_key("Amblin", "studio"), "title": "Amblin"},
        m.passage_key("Mike Medavoy", "co-founded Orion Pictures"): {
            "id": 3, "key": m.passage_key("Mike Medavoy", "co-founded Orion Pictures"),
            "title": "Mike Medavoy"},
    }
    text_index = m.build_passage_text_index(FakeCorpus([]), passages, records)
    check("text index fills bodies from records", text_index[1] == "UHF: 1989 film", text_index[1])
    check("text index fills gaps from the corpus table",
          text_index[3] == "Mike Medavoy: co-founded Orion Pictures", text_index[3])


def test_backend_labelling():
    original_minilm, original_bge = m.get_minilm_reranker, m.get_bge_reranker
    m.get_minilm_reranker = lambda: object()
    m.get_bge_reranker = lambda: None
    try:
        label = m.reranker_backend_label("two_stage_ensemble")
        check("label reports the fallback backend",
              label == "minilm:cross-encoder+bge:bi-encoder-fallback", label)
        check("baselines request no backends", m.active_reranker_backends("vector_only") == [])
    finally:
        m.get_minilm_reranker, m.get_bge_reranker = original_minilm, original_bge


class FakeBackend:
    """Minimal stand-in for EmbeddingBackend (no torch, no weights)."""

    def __init__(self, key, device="cpu"):
        self.key = key
        self.profile = m.embedding_profile(key)
        self.hf_id = self.profile["hf_id"]
        self.family = self.profile["family"]
        self.dim = self.profile["dim"]
        self.dtype = self.profile["default_dtype"]
        self.device = device
        self.max_seq_length = self.profile["max_seq_length"]


def test_embedding_profiles():
    nv = m.embedding_profile("nvembed")
    check("nvembed is 4096-d", nv["dim"] == 4096, nv["dim"])
    check("nvembed carries a retrieval instruction", bool(nv["query_instruction"]))
    check("nvembed reserves its weights", nv["weights_gb"] > 10, nv["weights_gb"])
    check("bge-large is 1024-d", m.embedding_profile("bge-large")["dim"] == 1024)
    check("cache stems are filesystem-safe", m.embedding_cache_stem("nvembed") == "nvembed")
    try:
        m.embedding_profile("does-not-exist")
    except SystemExit as exc:
        check("unknown profile exits cleanly", "Unknown --embed-model" in str(exc), str(exc))
    else:
        check("unknown profile exits cleanly", False)


def test_embedding_dtype_defaults():
    check("minilm defaults to float32",
          m.EmbeddingBackend("minilm", device="cpu").dtype == "float32")
    check("nvembed defaults to bfloat16",
          m.EmbeddingBackend("nvembed", device="cpu").dtype == "bfloat16")
    check("float16 on cpu is promoted to bfloat16",
          m.EmbeddingBackend("nvembed", device="cpu", dtype="float16").dtype == "bfloat16")


def test_l2_normalize():
    rows = m.l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32))
    check("rows are unit length", np.allclose(np.linalg.norm(rows[0]), 1.0), rows[0])
    check("zero rows stay finite", np.all(np.isfinite(rows[1])) and np.allclose(rows[1], 0.0), rows[1])
    one_d = m.l2_normalize(np.array([3.0, 4.0], dtype=np.float32))
    check("1-D input is reshaped", one_d.shape == (1, 2) and np.allclose(one_d[0], [0.6, 0.8]), one_d)


def test_embedding_metadata(tmp_dir="/tmp/musique_embed_meta"):
    os.makedirs(tmp_dir, exist_ok=True)
    meta_path = m.embedding_metadata_path(tmp_dir)
    if os.path.exists(meta_path):
        os.remove(meta_path)

    minilm, nv = FakeBackend("minilm"), FakeBackend("nvembed")
    check("empty scratch has no mismatch", m.corpus_embedding_mismatch(tmp_dir, nv) == "")

    m.write_embedding_metadata(tmp_dir, minilm, rows=42, encode_seconds=1.5)
    meta = m.read_embedding_metadata(tmp_dir)
    check("metadata records rows", meta["rows"] == 42, meta)
    check("metadata records the encode device", meta["device"] == "cpu", meta["device"])
    check("same model has no mismatch", m.corpus_embedding_mismatch(tmp_dir, minilm) == "")

    # A GPU-built corpus is reusable on CPU: the frozen vectors do not carry a device.
    m.write_embedding_metadata(tmp_dir, FakeBackend("minilm", device="cuda"), rows=42)
    check("encode device does not invalidate the corpus",
          m.corpus_embedding_mismatch(tmp_dir, minilm) == "")
    m.write_embedding_metadata(tmp_dir, minilm, rows=42, encode_seconds=1.5)
    reason = m.corpus_embedding_mismatch(tmp_dir, nv)
    check("embedder switch detected", "all-MiniLM" in reason, reason)

    # Same model id, different width: the Iceberg fixed-size list would be wrong.
    with open(meta_path, "w") as fh:
        json.dump({"profile": "nvembed", "hf_id": nv.hf_id, "dim": 1024}, fh)
    reason = m.corpus_embedding_mismatch(tmp_dir, nv)
    check("dimension switch detected", "1024-d" in reason and "4096-d" in reason, reason)


class StubEmbedder:
    """Deterministic stand-in for the encode path: one unit vector per passage."""

    key = "stub"
    hf_id = "stub/embedder"
    device = "cpu"
    dim = 4

    def __init__(self):
        self.calls = 0

    def encode_documents(self, texts, batch_size=8, show_progress_bar=False):
        self.calls += 1
        rows = len(list(texts))
        out = np.zeros((rows, self.dim), dtype=np.float32)
        out[:, 0] = 1.0
        return out


class BatchRecordingEmbedder(StubEmbedder):
    def __init__(self):
        super().__init__()
        self.batches = []

    def encode_documents(self, texts, batch_size=8, show_progress_bar=False):
        self.batches.append((len(list(texts)), batch_size))
        return super().encode_documents(texts, batch_size, show_progress_bar)


def test_encode_projection_uses_run_batch():
    """The ETA must be measured at the batch size the encode actually uses."""
    embedder = BatchRecordingEmbedder()
    m.estimate_encode_seconds(embedder, [f"p{i}" for i in range(100)], 100, batch_size=32)
    check("projection warms up at batch 1", embedder.batches[0] == (1, 1), embedder.batches[0])
    check("projection measures at the run batch size", embedder.batches[1] == (32, 32), embedder.batches[1])


def test_encode_resumable(tmp_dir="/tmp/musique_encode"):
    os.makedirs(tmp_dir, exist_ok=True)
    cache_path = os.path.join(tmp_dir, "vectors_stub_4.npy")
    partial_path = os.path.join(tmp_dir, "vectors_stub_4.partial.npy")
    for path in (cache_path, partial_path):
        if os.path.exists(path):
            os.remove(path)

    texts = [f"passage {i}" for i in range(10)]
    matrix, _ = m.encode_corpus_resumable(StubEmbedder(), texts, 4, cache_path, chunk_size=4)
    check("resumable encode covers every passage", matrix.shape == (10, 4), matrix.shape)
    check("checkpoint is removed on success", not os.path.exists(partial_path))

    # Simulate a kill after 6 rows: the resumed run must keep those rows and only
    # encode the remainder.
    np.save(partial_path, np.full((6, 4), 2.0, dtype=np.float32))
    matrix, _ = m.encode_corpus_resumable(StubEmbedder(), texts, 4, cache_path, chunk_size=4)
    check("resume keeps checkpointed rows", np.allclose(matrix[:6], 2.0), matrix[:2])
    check("resume encodes only the remainder", np.allclose(matrix[6:], [[1, 0, 0, 0]] * 4), matrix[6:])

    # A checkpoint from another embedder must be discarded, not trusted.
    np.save(partial_path, np.full((3, 2), 9.0, dtype=np.float32))
    matrix, _ = m.encode_corpus_resumable(StubEmbedder(), texts, 4, cache_path, chunk_size=4)
    check("wrong-width checkpoint is discarded",
          matrix.shape == (10, 4) and np.allclose(matrix[0], [1, 0, 0, 0]), matrix[0])

    try:
        m.encode_corpus_resumable(StubEmbedder(), texts, 4, cache_path, chunk_size=4,
                                  eta_limit_s=0.0, allow_slow=False)
    except SystemExit as exc:
        check("slow encode is refused without opt-in", "--allow-slow-encode" in str(exc), str(exc))
    else:
        check("slow encode is refused without opt-in", False)

    matrix, _ = m.encode_corpus_resumable(StubEmbedder(), texts, 4, cache_path, chunk_size=4,
                                          eta_limit_s=0.0, allow_slow=True)
    check("slow encode proceeds when allowed", matrix.shape == (10, 4))
    check("duration formatter is readable",
          m.format_duration(45) == "45s" and "min" in m.format_duration(600)
          and "h" in m.format_duration(9000))


def test_report(tmp_dir="/tmp/musique_two_stage_report"):
    results = {
        "vector_only": {"evidence_recall_at_k": 0.487, "precision_at_k": 0.21, "qps": 100.0,
                        "p50_ms": 9.1, "p99_ms": 20.0, "queries_evaluated": 10},
        "two_stage_minilm": {"evidence_recall_at_k": 0.58, "precision_at_k": 0.3, "qps": 40.0,
                             "p50_ms": 25.0, "p99_ms": 41.0, "queries_evaluated": 10,
                             "reranker_backend": "minilm:cross-encoder", "seed_k": 2,
                             "candidate_pool_size": 10, "avg_pool_size": 8.5,
                             "stage1_p50_ms": 11.0, "stage2_p50_ms": 9.4, "stage2_p99_ms": 15.0},
    }
    m.save_report(results, 5, tmp_dir)
    with open(os.path.join(tmp_dir, "MUSIQUE_BENCHMARK_REPORT.md")) as fh:
        report = fh.read()
    check("report includes the stage table", "Two-Stage Latency Breakdown" in report)
    check("report documents the wider HNSW request", "k = candidate pool cap" in report)
    with open(os.path.join(tmp_dir, "musique_graphrag_results.json")) as fh:
        check("json keeps stage metrics", json.load(fh)["two_stage_minilm"]["stage2_p50_ms"] == 9.4)


if __name__ == "__main__":
    test_stage1_merge()
    test_stage1_two_hop_expansion()
    test_stage1_graph_cannot_crowd_out_vectors()
    test_stage1_degrades_to_vector_ranking()
    test_stage2_ordering(lambda backend, q, texts: np.array([0.1, 9.0, 4.0])[: len(texts)])
    test_ensemble_is_scale_invariant()
    test_dedupe_by_title()
    test_ppr_mode_variants()
    test_graph_metadata()
    test_text_index()
    test_backend_labelling()
    test_embedding_profiles()
    test_embedding_dtype_defaults()
    test_l2_normalize()
    test_embedding_metadata()
    test_encode_projection_uses_run_batch()
    test_encode_resumable()
    test_report()
    print("\nALL TWO-STAGE CHECKS PASSED")
