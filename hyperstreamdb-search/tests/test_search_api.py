#!/usr/bin/env python3
"""ES 7.10 conformance suite for the ``hypersearch`` binary (plan Step 4.1).

Exercises the live REST API end to end: document ingestion via HTTP POST
(auto- and explicit-id writes, duplicate rejection), dynamic index creation
with schema evolution, and hybrid lexical + vector search (BM25 ``match``,
``knn``, fused hybrid, filters, pagination) against a spawned ``hypersearch``
process.

Run from the repo root:

    ./venv/bin/pytest hyperstreamdb-search/tests/test_search_api.py
"""

import os
import socket
import subprocess
import time
import uuid
from pathlib import Path

import pytest
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
BINARY = REPO_ROOT / "target" / "debug" / "hypersearch"

STARTUP_DEADLINE_S = 60.0
POLL_INTERVAL_S = 0.25


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Api:
    """HTTP helpers bound to one hypersearch process."""

    def __init__(self, base: str, session: requests.Session):
        self.base = base
        self.session = session

    def index(self, tag: str = "") -> str:
        suffix = f"-{tag}" if tag else ""
        return f"api-{uuid.uuid4().hex[:12]}{suffix}"

    def index_doc(self, index: str, doc, doc_id: str | None = None):
        path = f"/{index}/_doc" + (f"/{doc_id}" if doc_id else "")
        return self.session.post(self.base + path, json=doc)

    def delete_doc(self, index: str, doc_id: str):
        return self.session.delete(self.base + f"/{index}/_doc/{doc_id}")

    def refresh(self, index: str):
        return self.session.post(self.base + f"/{index}/_refresh")

    def refresh_all(self):
        return self.session.post(self.base + "/_refresh")

    def search(self, index: str, body: dict):
        return self.session.post(self.base + f"/{index}/_search", json=body)

    def search_get(self, index: str, params: dict):
        return self.session.get(self.base + f"/{index}/_search", params=params)

    def count(self, index: str, body: dict | None = None):
        if body is None:
            return self.session.get(self.base + f"/{index}/_count")
        return self.session.get(self.base + f"/{index}/_count", json=body)

    def bulk(self, ndjson: str, index: str | None = None):
        path = self.base + (f"/{index}/_bulk" if index else "/_bulk")
        return self.session.post(
            path, data=ndjson, headers={"Content-Type": "application/x-ndjson"}
        )

    def create_index(self, index: str, body: dict | None = None):
        return self.session.put(self.base + f"/{index}", json=body or {})

    def get_index(self, index: str):
        return self.session.get(self.base + f"/{index}")

    def delete_index(self, index: str):
        return self.session.delete(self.base + f"/{index}")

    def get_mapping(self, index: str):
        return self.session.get(self.base + f"/{index}/_mapping")

    def put_mapping(self, index: str, body: dict):
        return self.session.put(self.base + f"/{index}/_mapping", json=body)

    def cat_indices(self):
        return self.session.get(self.base + "/_cat/indices")

    def cluster_stats(self):
        return self.session.get(self.base + "/_cluster/stats")


@pytest.fixture(scope="session")
def api(tmp_path_factory):
    if not BINARY.exists():
        pytest.skip(f"{BINARY} not built; run `cargo build --bin hypersearch` first")

    port = _free_port()
    storage = tmp_path_factory.mktemp("hypersearch")
    env = {
        **os.environ,
        "HYPERSEARCH_BIND": "127.0.0.1",
        "HYPERSEARCH_PORT": str(port),
        "HYPERSEARCH_STORAGE_URI": f"file://{storage}",
    }
    proc = subprocess.Popen(
        [str(BINARY)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"

    deadline = time.time() + STARTUP_DEADLINE_S
    ready = False
    while time.time() < deadline:
        try:
            if requests.get(base + "/", timeout=2).status_code == 200:
                ready = True
                break
        except requests.RequestException:
            pass
        time.sleep(POLL_INTERVAL_S)
    if not ready:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        pytest.fail("hypersearch did not become ready within 60s")

    yield Api(base, requests.Session())

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def test_cluster_root(api):
    r = api.session.get(api.base + "/")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "hypersearch-1"
    assert body["cluster_name"] == "hypersearch"
    uuid.UUID(body["cluster_uuid"])  # random v4 per process
    version = body["version"]
    assert version["number"] == "7.10.2"
    assert version["lucene_version"] == "8.7.0"
    assert version["build_flavor"] == "default"
    assert version["build_type"] == "tar"
    assert version["build_snapshot"] is False
    assert body["tagline"] == "You know, you search"


def test_health_endpoints(api):
    for path in ("/_health", "/_cluster/health"):
        r = api.session.get(api.base + path)
        assert r.status_code == 200, path
        h = r.json()
        assert h["cluster_name"] == "hypersearch"
        assert h["status"] == "green"
        assert h["timed_out"] is False
        assert h["number_of_nodes"] == 1
        assert h["number_of_data_nodes"] == 1
        assert h["active_shards"] == h["active_primary_shards"] >= 0
        assert h["unassigned_shards"] == 0
        assert h["relocating_shards"] == 0
        assert h["active_shards_percent_as_number"] == 100.0


def test_metrics(api):
    r = api.session.get(api.base + "/metrics")
    assert r.status_code == 200
    assert r.headers["Content-Type"].startswith("text/plain; version=0.0.4")
    assert len(r.text.strip()) > 0


def test_doc_write_semantics(api):
    idx = api.index("write")

    # First write into a new index: 201 / "created", server-generated id.
    r = api.index_doc(idx, {"name": "alice", "age": 30})
    assert r.status_code == 201
    body = r.json()
    assert body["_index"] == idx
    assert body["result"] == "created"
    assert body["_version"] == 1
    assert body["_shards"] == {"total": 1, "successful": 1, "failed": 0}
    auto_id = body["_id"]
    assert len(auto_id) == 36 and auto_id.count("-") == 4
    uuid.UUID(auto_id)  # uuid4-shaped

    # Second write into the existing index: 200 / "updated".
    r = api.index_doc(idx, {"name": "bob"})
    assert r.status_code == 200
    assert r.json()["result"] == "updated"

    # Explicit id into a fresh index: 201 / "created", id echoed.
    idx2 = api.index("explicit")
    r = api.index_doc(idx2, {"name": "carol"}, doc_id="explicit-1")
    assert r.status_code == 201
    assert r.json()["_id"] == "explicit-1"
    assert r.json()["result"] == "created"


def test_duplicate_id_rejected(api):
    idx = api.index("dup")
    assert api.index_doc(idx, {"v": 1}, doc_id="doc-42").status_code == 201

    r = api.index_doc(idx, {"v": 2}, doc_id="doc-42")
    assert r.status_code == 400
    body = r.json()
    assert body["status"] == 400
    assert body["error"]["type"] == "resource_already_exists_exception"
    assert "doc-42" in body["error"]["reason"]


def test_non_object_body_rejected(api):
    idx = api.index("badbody")
    r = api.index_doc(idx, [1, 2, 3])
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "illegal_argument_exception"


def test_schema_evolution_and_match_all(api):
    idx = api.index("evo")
    docs = [
        ("evo-1", {"title": "one", "count": 1}),
        ("evo-2", {"title": "two", "count": 2.5, "active": True}),
        ("evo-3", {"title": "three", "active": False, "vec": [0.1, 0.2]}),
        ("evo-4", {"title": "four"}),
    ]
    for doc_id, doc in docs:
        assert api.index_doc(idx, doc, doc_id=doc_id).status_code in (200, 201)

    r = api.refresh(idx)
    assert r.status_code == 200
    assert r.json()["_shards"] == {"total": 1, "successful": 1, "failed": 0}

    r = api.search(idx, {"query": {"match_all": {}}, "size": 10})
    assert r.status_code == 200
    body = r.json()
    assert body["timed_out"] is False
    assert body["hits"]["total"] == {"value": 4, "relation": "eq"}

    hits = body["hits"]["hits"]
    # No relevance signal: id-ascending order, uniform 1.0 scores.
    assert [h["_id"] for h in hits] == ["evo-1", "evo-2", "evo-3", "evo-4"]
    for h in hits:
        assert h["_index"] == idx
        assert h["_score"] == 1.0
        assert "_id" not in h["_source"]
        assert "distance" not in h["_source"]

    by_id = {h["_id"]: h["_source"] for h in hits}
    # Evolved columns: absent fields come back as JSON null.
    assert by_id["evo-1"]["title"] == "one"
    assert by_id["evo-1"]["count"] == 1  # Int64+Float64 merged to f64
    assert by_id["evo-1"]["active"] is None
    assert by_id["evo-1"]["vec"] is None
    assert by_id["evo-2"]["count"] == 2.5
    assert by_id["evo-2"]["active"] is True
    # Vectors round-trip as f32; compare approximately.
    assert by_id["evo-3"]["vec"] == pytest.approx([0.1, 0.2], abs=1e-6)
    assert by_id["evo-3"]["count"] is None
    assert by_id["evo-4"]["count"] is None
    assert by_id["evo-4"]["active"] is None


def test_refresh_unknown_index_404(api):
    r = api.refresh(api.index("ghost"))
    assert r.status_code == 404
    body = r.json()
    assert body["status"] == 404
    assert body["error"]["type"] == "index_not_found_exception"


def test_match_bm25(api):
    idx = api.index("cat")
    docs = [
        {"title": "alpha", "body": "quick brown fox", "category": "animal", "age": 10},
        {"title": "beta", "body": "lazy dog sleeps", "category": "animal", "age": 20},
        {"title": "gamma", "body": "the cat purred", "category": "animal", "age": 30},
        {"title": "delta", "body": "a fish swims", "category": "seafood", "age": 40},
    ]
    for i, doc in enumerate(docs):
        api.index_doc(idx, doc, doc_id=f"cat-doc-{i}")
    assert api.refresh(idx).status_code == 200

    r = api.search(idx, {"query": {"match": {"body": "cat"}}})
    assert r.status_code == 200
    body = r.json()
    assert body["hits"]["total"]["value"] == 1
    assert body["hits"]["max_score"] > 0
    hit = body["hits"]["hits"][0]
    assert hit["_id"] == "cat-doc-2"
    assert hit["_score"] > 0
    assert hit["_source"]["title"] == "gamma"
    assert hit["_source"]["category"] == "animal"
    assert hit["_source"]["age"] == 30
    assert "_id" not in hit["_source"]
    assert "distance" not in hit["_source"]

    # Wrapped value form: {"field": {"query": "text"}}.
    r = api.search(idx, {"query": {"match": {"body": {"query": "cat"}}}})
    body = r.json()
    assert body["hits"]["total"]["value"] == 1
    assert body["hits"]["hits"][0]["_id"] == "cat-doc-2"


def test_knn(api):
    idx = api.index("knn")
    docs = [
        {"name": "a", "vec": [1.0, 0.0]},
        {"name": "b", "vec": [0.0, 1.0]},
        {"name": "c", "vec": [0.1, 0.1]},
        {"name": "d", "vec": [0.9, 0.1]},
    ]
    for i, doc in enumerate(docs):
        api.index_doc(idx, doc, doc_id=f"knn-doc-{i}")
    assert api.refresh(idx).status_code == 200

    # Top-level (ES8-style) knn spec.
    r = api.search(idx, {"knn": {"field": "vec", "vector": [1.0, 0.0], "k": 2}})
    assert r.status_code == 200
    body = r.json()
    assert body["hits"]["total"]["value"] == 2
    assert len(body["hits"]["hits"]) == 2
    first = body["hits"]["hits"][0]
    # Exact match has distance 0 -> ES-style score 1/(1+0) == 1.0.
    assert first["_id"] == "knn-doc-0"
    assert first["_score"] == 1.0
    assert "distance" not in first["_source"]
    scores = [h["_score"] for h in body["hits"]["hits"]]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    # query.knn form (wins over a top-level knn when both are present).
    r = api.search(
        idx,
        {
            "query": {"knn": {"field": "vec", "vector": [1.0, 0.0], "k": 2}},
            "knn": {"field": "vec", "vector": [0.0, 1.0], "k": 2},
        },
    )
    body = r.json()
    assert body["hits"]["hits"][0]["_id"] == "knn-doc-0"

    # k larger than the dataset returns every document.
    r = api.search(idx, {"knn": {"field": "vec", "vector": [1.0, 0.0], "k": 10}})
    assert r.json()["hits"]["total"]["value"] == 4


def test_hybrid_rrf(api):
    idx = api.index("hyb")
    docs = [
        {"body": "hello world", "vec": [1.0, 0.0]},
        {"body": "goodbye moon", "vec": [0.0, 1.0]},
        {"body": "hello moon", "vec": [0.5, 0.5]},
        {"body": "world moon", "vec": [1.0, 1.0]},
    ]
    for i, doc in enumerate(docs):
        api.index_doc(idx, doc, doc_id=f"hyb-doc-{i}")
    assert api.refresh(idx).status_code == 200

    r = api.search(
        idx,
        {
            "query": {
                "match": {"body": "hello"},
                "knn": {"field": "vec", "vector": [1.0, 0.0], "k": 2},
            }
        },
    )
    assert r.status_code == 200
    body = r.json()
    hits = body["hits"]["hits"]
    assert len(hits) > 0
    for h in hits:
        # RRF fusion scores lie strictly in (0, 1).
        assert 0.0 < h["_score"] < 1.0


def test_filters(api):
    idx = api.index("filt")
    docs = [
        {"title": "t1", "body": "quick brown fox", "category": "animal", "age": 10},
        {
            "title": "t2",
            "body": "lazy dog sleeps",
            "category": "animal",
            "age": 45,
            "extra": "x",
        },
        {"title": "t3", "body": "the cat purred", "category": "animal", "age": 30},
        {
            "title": "t4",
            "body": "a fish swims",
            "category": "seafood",
            "age": 40,
            "extra": "y",
        },
    ]
    for i, doc in enumerate(docs):
        api.index_doc(idx, doc, doc_id=f"fdoc-{i}")
    assert api.refresh(idx).status_code == 200

    # Disjoint match + term: empty page, no max_score key at all.
    r = api.search(
        idx, {"query": {"match": {"body": "quick"}}, "filter": {"term": {"category": "seafood"}}}
    )
    body = r.json()
    assert body["hits"]["total"]["value"] == 0
    assert body["hits"]["hits"] == []
    assert "max_score" not in body["hits"]

    # Intersecting match + term.
    r = api.search(
        idx, {"query": {"match": {"body": "quick"}}, "filter": {"term": {"category": "animal"}}}
    )
    body = r.json()
    assert body["hits"]["total"]["value"] == 1
    assert body["hits"]["hits"][0]["_id"] == "fdoc-0"

    # Single-bound range.
    r = api.search(idx, {"query": {"match_all": {}}, "filter": {"range": {"age": {"gte": 30}}}})
    body = r.json()
    assert body["hits"]["total"]["value"] == 3
    assert [h["_id"] for h in body["hits"]["hits"]] == ["fdoc-1", "fdoc-2", "fdoc-3"]
    assert all(h["_score"] == 1.0 for h in body["hits"]["hits"])

    # Multi-bound range.
    r = api.search(
        idx, {"query": {"match_all": {}}, "filter": {"range": {"age": {"gte": 30, "lt": 45}}}}
    )
    body = r.json()
    assert [h["_id"] for h in body["hits"]["hits"]] == ["fdoc-2", "fdoc-3"]

    # exists on a nullable evolved column.
    r = api.search(idx, {"query": {"match_all": {}}, "filter": {"exists": {"field": "extra"}}})
    body = r.json()
    assert [h["_id"] for h in body["hits"]["hits"]] == ["fdoc-1", "fdoc-3"]

    # Array of clauses is AND-joined.
    r = api.search(
        idx,
        {
            "query": {"match_all": {}},
            "filter": [
                {"term": {"category": "animal"}},
                {"range": {"age": {"gte": 30}}},
            ],
        },
    )
    body = r.json()
    assert [h["_id"] for h in body["hits"]["hits"]] == ["fdoc-1", "fdoc-2"]


def test_pagination(api):
    idx = api.index("pag")
    for i in range(5):
        api.index_doc(idx, {"n": i + 1}, doc_id=f"pdoc-{i}")
    assert api.refresh(idx).status_code == 200

    r = api.search(idx, {"query": {"match_all": {}}, "size": 2, "from": 1})
    assert r.status_code == 200
    body = r.json()
    assert body["hits"]["total"]["value"] == 5
    assert [h["_id"] for h in body["hits"]["hits"]] == ["pdoc-1", "pdoc-2"]
    assert all(h["_score"] == 1.0 for h in body["hits"]["hits"])

    # from beyond the end: empty page, no error, no max_score.
    r = api.search(idx, {"query": {"match_all": {}}, "size": 2, "from": 100})
    assert r.status_code == 200
    body = r.json()
    assert body["hits"]["total"]["value"] == 5
    assert body["hits"]["hits"] == []
    assert "max_score" not in body["hits"]


def test_search_unknown_index_404(api):
    r = api.search(api.index("ghost"), {"query": {"match_all": {}}})
    assert r.status_code == 404
    body = r.json()
    assert body["status"] == 404
    assert body["error"]["type"] == "index_not_found_exception"


def test_search_request_errors_400(api):
    idx = api.index("err")
    assert api.index_doc(idx, {"body": "hello"}).status_code == 201
    assert api.refresh(idx).status_code == 200

    cases = [
        # match value must be a string or wrapped object
        {"query": {"match": {"body": ["a", "b"]}}},
        # unsupported query clause
        {"query": {"match_phrase": {"body": "a"}}},
        # query must be an object
        {"query": "hello"},
        # unsupported filter clause
        {"filter": {"match_phrase": {"body": "a"}}},
        # knn without a vector
        {"knn": {"field": "body", "k": 3}},
        # SQL-injection guard on field names
        {"filter": {"term": {"bad;drop table": "x"}}},
    ]
    for case in cases:
        r = api.search(idx, case)
        assert r.status_code == 400, f"{case}: {r.status_code} {r.text}"
        body = r.json()
        assert body["status"] == 400
        assert body["error"]["type"] == "illegal_argument_exception", f"{case}: {body}"


# ---------------------------------------------------------------------------
# M3: index CRUD, mapping, bulk, count, cat/stats, delete 501, DSL extensions
# ---------------------------------------------------------------------------


def test_index_crud(api):
    idx = api.index("crud")

    # Create with an explicit mapping.
    r = api.create_index(
        idx,
        {
            "mappings": {
                "properties": {
                    "title": {"type": "text"},
                    "age": {"type": "long"},
                    "vec": {"type": "dense_vector", "dims": 2},
                }
            }
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["acknowledged"] is True
    assert body["shards_acknowledged"] is True
    assert body["index"] == idx

    # Creating again is a 400 resource_already_exists.
    r = api.create_index(idx, {})
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "resource_already_exists_exception"

    # GET returns the mapping.
    r = api.get_index(idx)
    assert r.status_code == 200
    meta = r.json()[idx]
    assert meta["mappings"]["properties"]["title"]["type"] == "text"
    assert meta["mappings"]["properties"]["age"]["type"] == "long"
    assert meta["mappings"]["properties"]["vec"]["type"] == "dense_vector"
    assert meta["mappings"]["properties"]["vec"]["dims"] == 2
    assert meta["aliases"] == {}
    assert meta["settings"]["index"]["number_of_shards"] == "1"

    # DELETE removes it; subsequent GET is 404.
    r = api.delete_index(idx)
    assert r.status_code == 200
    assert r.json()["acknowledged"] is True
    assert api.get_index(idx).status_code == 404


def test_mapping_get_put(api):
    idx = api.index("map")
    assert api.index_doc(idx, {"title": "one", "count": 1}, doc_id="m1").status_code == 201
    assert api.refresh(idx).status_code == 200

    # GET reflects the inferred schema.
    r = api.get_mapping(idx)
    assert r.status_code == 200
    props = r.json()[idx]["mappings"]["properties"]
    assert props["title"]["type"] == "text"
    assert props["count"]["type"] == "long"

    # PUT adds a new column; it shows up in a subsequent GET.
    r = api.put_mapping(idx, {"properties": {"score": {"type": "double"}}})
    assert r.status_code == 200, r.text
    assert r.json()["acknowledged"] is True
    props = api.get_mapping(idx).json()[idx]["mappings"]["properties"]
    assert props["score"]["type"] == "double"


def test_bulk_mixed(api):
    idx = api.index("bulk")
    # Pre-index b1 so a bulk duplicate of it is a real primary-key violation.
    assert api.index_doc(idx, {"title": "alpha", "age": 10}, doc_id="b1").status_code == 201

    ndjson = "\n".join(
        [
            # Duplicate of the already-buffered b1 -> per-item 400.
            '{"index": {"_index": "%s", "_id": "b1"}}' % idx,
            '{"title": "dup", "age": 30}',
            # A new doc via the `create` op -> 200 (index already exists).
            '{"create": {"_index": "%s", "_id": "b2"}}' % idx,
            '{"title": "beta", "age": 20}',
        ]
    )
    r = api.bulk(ndjson)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["errors"] is True
    assert len(body["items"]) == 2
    by_id = {}
    for it in body["items"]:
        for op_key in ("index", "create"):
            if op_key in it:
                by_id[it[op_key]["_id"]] = it[op_key]["status"]
                break
    assert by_id["b1"] == 400
    assert by_id["b2"] == 200

    assert api.refresh(idx).status_code == 200
    # b1 (original) + b2 landed; the duplicate was rejected.
    assert api.count(idx).json()["count"] == 2


def test_bulk_delete_is_501(api):
    idx = api.index("bulkdel")
    ndjson = "\n".join(
        [
            '{"index": {"_index": "%s", "_id": "d1"}}' % idx,
            '{"title": "x"}',
            '{"delete": {"_index": "%s", "_id": "d1"}}' % idx,
        ]
    )
    r = api.bulk(ndjson)
    assert r.status_code == 200
    body = r.json()
    assert body["errors"] is True
    delete_item = body["items"][1]["delete"]
    assert delete_item["status"] == 501


def test_count_filtered(api):
    idx = api.index("cnt")
    docs = [
        ("c1", {"category": "animal", "age": 10}),
        ("c2", {"category": "animal", "age": 45}),
        ("c3", {"category": "seafood", "age": 30}),
    ]
    for doc_id, doc in docs:
        api.index_doc(idx, doc, doc_id=doc_id)
    assert api.refresh(idx).status_code == 200

    assert api.count(idx).json()["count"] == 3
    # Filtered count.
    r = api.count(idx, {"filter": {"term": {"category": "animal"}}})
    assert r.status_code == 200
    assert r.json()["count"] == 2
    # Range filter.
    r = api.count(idx, {"filter": {"range": {"age": {"gte": 30}}}})
    assert r.json()["count"] == 2


def test_cat_indices_and_cluster_stats(api):
    idx = api.index("cat")
    for i in range(3):
        api.index_doc(idx, {"n": i}, doc_id=f"cat-{i}")
    assert api.refresh(idx).status_code == 200

    r = api.cat_indices()
    assert r.status_code == 200
    lines = r.text.strip().splitlines()
    assert lines[0].startswith("health")
    assert any(idx in line for line in lines[1:])

    r = api.cluster_stats()
    assert r.status_code == 200
    body = r.json()
    assert body["cluster_name"] == "hypersearch"
    assert body["status"] == "green"
    assert body["indices"]["count"] >= 1
    assert body["nodes"]["count"]["total"] == 1


def test_delete_doc_is_501(api):
    idx = api.index("del501")
    assert api.index_doc(idx, {"v": 1}, doc_id="x1").status_code == 201
    r = api.delete_doc(idx, "x1")
    assert r.status_code == 501
    body = r.json()
    assert body["status"] == 501


def test_global_refresh(api):
    idx = api.index("gref")
    api.index_doc(idx, {"v": 1}, doc_id="g1")
    r = api.refresh_all()
    assert r.status_code == 200
    assert r.json()["_shards"] == {"total": 1, "successful": 1, "failed": 0}


def test_terms_and_must_not(api):
    idx = api.index("tm")
    docs = [
        ("t1", {"category": "animal", "age": 10}),
        ("t2", {"category": "animal", "age": 45}),
        ("t3", {"category": "seafood", "age": 30}),
    ]
    for doc_id, doc in docs:
        api.index_doc(idx, doc, doc_id=doc_id)
    assert api.refresh(idx).status_code == 200

    # terms (IN).
    r = api.search(
        idx,
        {"query": {"match_all": {}}, "filter": {"terms": {"category": ["animal", "seafood"]}}},
    )
    assert r.json()["hits"]["total"]["value"] == 3

    r = api.search(idx, {"query": {"match_all": {}}, "filter": {"terms": {"category": ["animal"]}}})
    assert r.json()["hits"]["total"]["value"] == 2

    # bool must_not.
    r = api.search(
        idx,
        {
            "query": {"match_all": {}},
            "filter": {"bool": {"must_not": [{"term": {"category": "seafood"}}]}},
        },
    )
    assert r.json()["hits"]["total"]["value"] == 2


def test_source_includes_excludes(api):
    idx = api.index("src")
    api.index_doc(idx, {"title": "one", "body": "hello world", "age": 5}, doc_id="s1")
    assert api.refresh(idx).status_code == 200

    # includes.
    r = api.search(idx, {"query": {"match_all": {}}, "_source": {"includes": ["title"]}})
    src = r.json()["hits"]["hits"][0]["_source"]
    assert set(src.keys()) == {"title"}

    # excludes.
    r = api.search(idx, {"query": {"match_all": {}}, "_source": {"excludes": ["body"]}})
    src = r.json()["hits"]["hits"][0]["_source"]
    assert "body" not in src
    assert "title" in src and "age" in src


def test_search_get_q_param(api):
    idx = api.index("qget")
    docs = [
        ("q1", {"title": "the quick brown fox"}),
        ("q2", {"title": "lazy dog sleeps"}),
        ("q3", {"title": "a fish swims"}),
    ]
    for doc_id, doc in docs:
        api.index_doc(idx, doc, doc_id=doc_id)
    assert api.refresh(idx).status_code == 200

    # q= matches the string column(s).
    r = api.search_get(idx, {"q": "fox"})
    assert r.status_code == 200
    body = r.json()
    assert body["hits"]["total"]["value"] >= 1
    assert body["hits"]["hits"][0]["_id"] == "q1"

    # No q -> match_all.
    r = api.search_get(idx, {})
    assert r.json()["hits"]["total"]["value"] == 3
