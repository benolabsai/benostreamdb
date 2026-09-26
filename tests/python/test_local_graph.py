import pyarrow as pa
import pytest
from benostreamdb import Table
import os

@pytest.fixture
def graph_table(tmpdir):
    schema = pa.schema([
        ("source", pa.uint64()),
        ("target", pa.uint64()),
        ("weight", pa.float64())
    ])
    table_uri = str(tmpdir.join("test_local_graph"))
    t = Table.create(table_uri, schema)
    
    # 1 -> 2
    # 2 -> 3
    # 3 -> 1
    # 1 -> 4
    # 4 -> 5
    edges = pa.Table.from_arrays([
        pa.array([1, 2, 3, 1, 4], type=pa.uint64()),
        pa.array([2, 3, 1, 4, 5], type=pa.uint64()),
        pa.array([1.0, 2.0, 1.5, 0.5, 1.0], type=pa.float64())
    ], names=["source", "target", "weight"])
    
    t.add_index("target", {"type": "graph", "src_column": "source", "dst_column": "target"})
    t.insert(edges)
    t.commit()
    t.wait_for_background_tasks()
    return t

def test_shortest_path(graph_table):
    # Shortest path from 1 to 5
    path = graph_table.shortest_path(1, 5, graph_column="source")
    assert path == [1, 4, 5]
    
    # Shortest path from 3 to 2
    path = graph_table.shortest_path(3, 2, graph_column="source")
    assert path == [3, 1, 2]

def test_neighbors(graph_table):
    # The CSR fast path returns a pyarrow Table with a 'neighbor' column,
    # consistent with the SQL path (it also carries `hops`/`max_degree`).
    res = graph_table.graph_neighbors(1, graph_column="source")
    df = res.to_pandas()
    # 1 has edges to 2 and 4
    assert set(df["neighbor"].astype(int)) == {2, 4}

def test_subgraph(graph_table):
    # With `graph_column`, subgraph now runs the multi-hop BFS (here 1 hop,
    # undirected) and returns the induced edges with payload columns. From
    # {1, 2, 3} the undirected 1-hop expansion reaches {1, 2, 3, 4}; the
    # induced subgraph keeps every edge whose endpoints are both visited.
    res = graph_table.subgraph([1, 2, 3], graph_column="source")
    df = res.to_pandas()
    edges = set(zip(df["source"].astype(int), df["target"].astype(int)))
    assert edges == {(1, 2), (2, 3), (3, 1), (1, 4)}

def test_connecting_paths(graph_table):
    edges = graph_table.connecting_paths([1, 5], graph_column="source")
    # path between 1 and 5 is 1->4->5, so edges should be (1, 4) and (4, 5)
    assert set(edges) == {(1, 4), (4, 5)}


def _is_connected(community, adj):
    """True when `community` is connected in the undirected graph `adj`."""
    if not community:
        return True
    start = next(iter(community))
    seen = {start}
    stack = [start]
    while stack:
        x = stack.pop()
        for y in adj.get(x, ()):
            if y in community and y not in seen:
                seen.add(y)
                stack.append(y)
    return seen == community


def test_leiden_communities_are_connected(tmpdir):
    """Leiden's refinement guarantees every community is internally connected."""
    # Two triangles joined by a single bridge 2-3.
    edges = pa.Table.from_arrays(
        [
            pa.array([0, 1, 2, 3, 4, 5, 2], type=pa.uint64()),
            pa.array([1, 2, 0, 4, 5, 3, 3], type=pa.uint64()),
            pa.array([1.0] * 7, type=pa.float64()),
        ],
        names=["source", "target", "weight"],
    )
    t = Table.create(
        str(tmpdir.join("leiden")),
        pa.schema(
            [
                ("source", pa.uint64()),
                ("target", pa.uint64()),
                ("weight", pa.float64()),
            ]
        ),
    )
    t.insert(edges)
    t.commit()
    t.wait_for_background_tasks()

    df = t.leiden_communities(resolution=1.0).to_pandas()
    # The UDF path mirrors the CSR path's output shape.
    assert {"community_id", "community"}.issubset(df.columns)
    communities = [set(int(x) for x in c) for c in df["community"]]

    # Valid partition: every node appears exactly once.
    assert set().union(*communities) == {0, 1, 2, 3, 4, 5}
    assert sum(len(c) for c in communities) == 6

    adj = {0: {1, 2}, 1: {0, 2}, 2: {0, 1, 3}, 3: {2, 4, 5}, 4: {3, 5}, 5: {3, 4}}
    for c in communities:
        assert _is_connected(c, adj)


def test_communities_without_index_uses_temp_csr(tmpdir):
    """Community detection works without a persisted graph index (temp CSR)."""
    schema = pa.schema(
        [
            ("source", pa.uint64()),
            ("target", pa.uint64()),
            ("weight", pa.float64()),
        ]
    )
    # Two disconnected triangles; no graph index is added.
    edges = pa.Table.from_arrays(
        [
            pa.array([0, 1, 2, 3, 4, 5], type=pa.uint64()),
            pa.array([1, 2, 0, 4, 5, 3], type=pa.uint64()),
            pa.array([1.0] * 6, type=pa.float64()),
        ],
        names=["source", "target", "weight"],
    )
    t = Table.create(str(tmpdir.join("no_csr")), schema)
    t.insert(edges)
    t.commit()
    t.wait_for_background_tasks()
    assert t.has_graph_index("source") is False

    df = t.communities(resolution=1.0).to_pandas()
    parts = {frozenset(int(x) for x in c) for c in df["community"]}
    assert parts == {frozenset({0, 1, 2}), frozenset({3, 4, 5})}

    # Warm-started updates work without a persisted index too.
    first = t.update_communities().to_pandas()
    second = t.update_communities(previous=first).to_pandas()
    m1 = {
        frozenset(int(x) for x in c): int(cid)
        for cid, c in zip(first["community_id"], first["community"])
    }
    m2 = {
        frozenset(int(x) for x in c): int(cid)
        for cid, c in zip(second["community_id"], second["community"])
    }
    assert m1 == m2


def test_to_polars_and_arrow_stream(graph_table):
    """Arrow is the boundary: to_arrow_stream and to_polars both work."""
    pl = pytest.importorskip("polars")

    df = graph_table.to_polars()
    assert isinstance(df, pl.DataFrame)
    assert {"source", "target", "weight"}.issubset(set(df.columns))
    assert df.height == len(graph_table)

    reader = graph_table.to_arrow_stream()
    table = reader.read_all()
    assert table.num_rows == len(graph_table)


def test_from_to_arrow_round_trip(tmpdir):
    """from_arrow / from_polars / from_pandas complement the to_* adapters."""
    pl = pytest.importorskip("polars")
    import pandas as pd

    src = pa.Table.from_arrays(
        [
            pa.array([1, 2, 3], type=pa.uint64()),
            pa.array(["a", "b", "c"], type=pa.string()),
        ],
        names=["id", "label"],
    )

    t = Table.from_arrow(str(tmpdir.join("from_arrow")), src)
    out = t.to_arrow()
    assert out.num_rows == 3
    assert set(out.column_names) == {"id", "label"}

    t2 = Table.from_polars(
        str(tmpdir.join("from_polars")), pl.DataFrame({"id": [1, 2], "label": ["x", "y"]})
    )
    assert t2.to_polars().height == 2

    t3 = Table.from_pandas(
        str(tmpdir.join("from_pandas")),
        pd.DataFrame({"id": [1, 2, 3], "label": ["p", "q", "r"]}),
    )
    assert len(t3) == 3


def test_communities_algorithm_selector(graph_table):
    """`communities()` dispatches to Louvain or Leiden and rejects unknown names."""
    louvain = graph_table.communities(algorithm="louvain").to_pandas()
    leiden = graph_table.communities(algorithm="leiden").to_pandas()
    assert len(louvain) > 0
    assert len(leiden) > 0
    with pytest.raises(ValueError):
        graph_table.communities(algorithm="bogus")


def test_summarize_communities_with_llm_and_embed(tmpdir):
    """LLM reports and embeddings are materialized and the embedding is indexed."""
    edge_schema = pa.schema(
        [
            ("source", pa.uint64()),
            ("target", pa.uint64()),
            ("weight", pa.float64()),
        ]
    )
    edges = pa.Table.from_arrays(
        [
            pa.array([0, 1, 2, 3, 4, 5, 2], type=pa.uint64()),
            pa.array([1, 2, 0, 4, 5, 3, 3], type=pa.uint64()),
            pa.array([1.0] * 7, type=pa.float64()),
        ],
        names=["source", "target", "weight"],
    )
    et = Table.create(str(tmpdir.join("edges")), edge_schema)
    et.insert(edges)
    et.commit()
    et.wait_for_background_tasks()

    doc_schema = pa.schema([("id", pa.uint64()), ("content", pa.string())])
    docs = pa.Table.from_arrays(
        [
            pa.array([0, 1, 2, 3, 4, 5], type=pa.uint64()),
            pa.array(["alpha", "beta", "gamma", "delta", "epsilon", "zeta"], type=pa.string()),
        ],
        names=["id", "content"],
    )
    dt = Table.create(str(tmpdir.join("docs")), doc_schema)
    dt.insert(docs)
    dt.commit()
    dt.wait_for_background_tasks()

    calls = {"llm": 0, "embed": 0}

    def fake_llm(prompt):
        calls["llm"] += 1
        return "REPORT: " + prompt[:20]

    def fake_embed(texts):
        calls["embed"] += 1
        return [[float(len(t)), 1.0, 0.0, 0.0] for t in texts]

    comm = et.summarize_communities(
        dt, str(tmpdir.join("comms")), llm=fake_llm, embed=fake_embed
    )
    df = comm.to_pandas()

    assert "report" in df.columns
    assert "embedding" in df.columns
    assert calls["llm"] == len(df)
    assert calls["embed"] == 1
    assert all(str(r).startswith("REPORT:") for r in df["report"])
    # The embedding column is indexed for vector retrieval.
    assert "embedding" in comm.index_columns


def _two_triangle_tables(tmpdir):
    edge_schema = pa.schema(
        [
            ("source", pa.uint64()),
            ("target", pa.uint64()),
            ("weight", pa.float64()),
        ]
    )
    edges = pa.Table.from_arrays(
        [
            pa.array([0, 1, 2, 3, 4, 5, 2], type=pa.uint64()),
            pa.array([1, 2, 0, 4, 5, 3, 3], type=pa.uint64()),
            pa.array([1.0] * 7, type=pa.float64()),
        ],
        names=["source", "target", "weight"],
    )
    et = Table.create(str(tmpdir.join("edges")), edge_schema)
    et.insert(edges)
    et.commit()
    et.wait_for_background_tasks()

    doc_schema = pa.schema(
        [
            ("id", pa.uint64()),
            ("content", pa.string()),
            ("embedding", pa.list_(pa.float32(), 4)),
        ]
    )
    docs = pa.Table.from_arrays(
        [
            pa.array([0, 1, 2, 3, 4, 5], type=pa.uint64()),
            pa.array(["a", "b", "c", "d", "e", "f"], type=pa.string()),
            pa.array([[1.0, 0.0, 0.0, 0.0]] * 6, type=pa.list_(pa.float32(), 4)),
        ],
        names=["id", "content", "embedding"],
    )
    dt = Table.create(str(tmpdir.join("docs")), doc_schema)
    dt.add_index("embedding", "hnsw")
    dt.insert(docs)
    dt.commit()
    dt.wait_for_background_tasks()
    return et, dt


def test_graph_rag_global_llm_router_prunes(tmpdir):
    """The router rates communities and prunes those below the threshold."""
    et, dt = _two_triangle_tables(tmpdir)
    calls = {"n": 0}

    def router(query, summary):
        calls["n"] += 1
        # Keep only the first community; prune the rest.
        return 1.0 if "Community 0 " in summary else 0.0

    res = et.graph_rag_search(
        query=[1.0, 0.0, 0.0, 0.0],
        doc_table=dt,
        mode="global",
        llm_router=router,
        relevance_threshold=0.5,
        top_k=5,
    )
    assert calls["n"] >= 1
    assert len(res.communities) == 1
    assert "relevance" in res.communities.columns


def test_graph_rag_global_without_router_keeps_all(tmpdir):
    """Without a router, the heuristic keeps every community."""
    et, dt = _two_triangle_tables(tmpdir)
    res = et.graph_rag_search(
        query=[1.0, 0.0, 0.0, 0.0], doc_table=dt, mode="global", top_k=5
    )
    assert len(res.communities) == 2


def test_extract_graph(tmpdir):
    """extract_graph materializes edges and claims from LLM JSON output."""
    import json

    doc_schema = pa.schema([("id", pa.uint64()), ("content", pa.string())])
    docs = pa.Table.from_arrays(
        [
            pa.array([1, 2], type=pa.uint64()),
            pa.array(["Alice works at Acme.", "Bob founded Beta."], type=pa.string()),
        ],
        names=["id", "content"],
    )
    dt = Table.create(str(tmpdir.join("docs")), doc_schema)
    dt.insert(docs)
    dt.commit()
    dt.wait_for_background_tasks()

    def fake_llm(prompt):
        return json.dumps(
            {
                "entities": ["Alice", "Acme"],
                "relationships": [
                    {"source": "Alice", "target": "Acme", "relation": "works_at", "weight": 1.0}
                ],
                "claims": [
                    {
                        "subject": "Alice",
                        "object": "Acme",
                        "claim": "Alice works at Acme",
                        "confidence": 0.9,
                    }
                ],
            }
        )

    et = dt.extract_graph(
        dt,
        str(tmpdir.join("kg")),
        llm=fake_llm,
        claims_uri=str(tmpdir.join("claims")),
    )
    df = et.to_pandas()
    assert len(df) == 2  # one edge per document
    assert set(df["relation"]) == {"works_at"}
    assert et.has_graph_index("source") is True
