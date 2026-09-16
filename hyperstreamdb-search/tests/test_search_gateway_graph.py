#!/usr/bin/env python3
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
    def __init__(self, base: str, session: requests.Session):
        self.base = base
        self.session = session

    def index(self, tag: str = "") -> str:
        suffix = f"-{tag}" if tag else ""
        return f"api-{uuid.uuid4().hex[:12]}{suffix}"

    def index_doc(self, index: str, doc, doc_id: str | None = None):
        path = f"/{index}/_doc" + (f"/{doc_id}" if doc_id else "")
        return self.session.post(self.base + path, json=doc)

    def refresh(self, index: str):
        return self.session.post(self.base + f"/{index}/_refresh")

    def graph_search(self, index: str, query: dict):
        return self.session.post(self.base + f"/{index}/_graph_search", json=query)

@pytest.fixture(scope="module")
def api(tmp_path_factory):
    if not BINARY.exists():
        pytest.skip(f"{BINARY} not built; run `cargo build --bin hypersearch` first")

    port = _free_port()
    storage = tmp_path_factory.mktemp("hypersearch_graph")
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

def test_graph_search_endpoint(api: Api):
    doc_index = api.index("docs")
    edge_index = api.index("edges")

    # Index doc table
    # Nodes 1..5
    api.index_doc(doc_index, {"id": 1, "content": "A", "embedding": [0.1, 0.9]}, doc_id="1")
    api.index_doc(doc_index, {"id": 2, "content": "B", "embedding": [0.2, 0.8]}, doc_id="2")
    api.index_doc(doc_index, {"id": 3, "content": "C", "embedding": [0.3, 0.7]}, doc_id="3")
    api.index_doc(doc_index, {"id": 4, "content": "D", "embedding": [0.4, 0.6]}, doc_id="4")
    api.index_doc(doc_index, {"id": 5, "content": "E", "embedding": [0.5, 0.5]}, doc_id="5")
    api.refresh(doc_index)

    # Index edge table
    # Graph: 1 -> 2, 2 -> 3, 4 -> 5, 1 -> 4
    api.index_doc(edge_index, {"source": 1, "target": 2, "relation": "rel1"}, doc_id="e1")
    api.index_doc(edge_index, {"source": 2, "target": 3, "relation": "rel2"}, doc_id="e2")
    api.index_doc(edge_index, {"source": 4, "target": 5, "relation": "rel1"}, doc_id="e3")
    api.index_doc(edge_index, {"source": 1, "target": 4, "relation": "rel2"}, doc_id="e4")
    api.refresh(edge_index)

    # Test graph search without relation filtering
    resp = api.graph_search(doc_index, {
        "edge_index": edge_index,
        "seed_ids": [1],
        "hops": 1,
        "directed": True,
        "id_field": "id",
        "query": {
            "match_all": {}
        }
    })
    
    assert resp.status_code == 200, resp.text
    hits = resp.json()["hits"]["hits"]
    # Hops 1 from seed 1 (directed) -> 2, 4. Plus seed 1.
    ids = {int(hit["_id"]) for hit in hits}
    assert ids == {1, 2, 4}

    # Test graph search with relation filtering
    resp = api.graph_search(doc_index, {
        "edge_index": edge_index,
        "seed_ids": [1],
        "hops": 1,
        "directed": True,
        "allowed_relations": ["rel1"],
        "id_field": "id",
        "query": {
            "match_all": {}
        }
    })
    
    assert resp.status_code == 200, resp.text
    hits = resp.json()["hits"]["hits"]
    # Only rel1: 1 -> 2
    ids = {int(hit["_id"]) for hit in hits}
    assert ids == {1, 2}
    
    # Test graph search hops = 2
    resp = api.graph_search(doc_index, {
        "edge_index": edge_index,
        "seed_ids": [1],
        "hops": 2,
        "directed": True,
        "id_field": "id",
        "query": {
            "match_all": {}
        }
    })
    
    assert resp.status_code == 200, resp.text
    hits = resp.json()["hits"]["hits"]
    # 1 -> 2 -> 3, 1 -> 4 -> 5
    ids = {int(hit["_id"]) for hit in hits}
    assert ids == {1, 2, 3, 4, 5}
