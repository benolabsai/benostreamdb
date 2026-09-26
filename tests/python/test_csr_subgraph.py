"""Correctness tests for the CSR-backed graph fast path.

These tests build a small synthetic graph with a *known* topology, build both a
forward CSR (on ``source``) and a reverse CSR (on ``target``), and assert that
the memory-mapped CSR path returns exactly the same induced edge set as the
SQL ``bfs_visited`` path for hops=1 and hops=2, both directed and undirected.
"""

import pyarrow as pa
import pytest

from benostreamdb import Table

NUM_NODES = 100


def _build_edges():
    """Deterministic 100-node graph: a ring plus two chord families."""
    sources = []
    targets = []
    weights = []
    for i in range(NUM_NODES):
        # ring: i -> i+1
        sources.append(i)
        targets.append((i + 1) % NUM_NODES)
        weights.append(1.0)
        # chord: i -> (i*7 + 3) % N
        sources.append(i)
        targets.append((i * 7 + 3) % NUM_NODES)
        weights.append(0.5)
        # chord: i -> (i*13 + 11) % N
        sources.append(i)
        targets.append((i * 13 + 11) % NUM_NODES)
        weights.append(0.25)
    return pa.Table.from_arrays(
        [
            pa.array(sources, type=pa.uint64()),
            pa.array(targets, type=pa.uint64()),
            pa.array(weights, type=pa.float64()),
        ],
        names=["source", "target", "weight"],
    )


@pytest.fixture
def csr_graph_table(tmpdir):
    schema = pa.schema(
        [
            ("source", pa.uint64()),
            ("target", pa.uint64()),
            ("weight", pa.float64()),
        ]
    )
    table_uri = str(tmpdir.join("test_csr_subgraph"))
    t = Table.create(table_uri, schema)

    # Forward CSR: index column_name == "source" (the index's src_column).
    t.add_index(
        "target", {"type": "graph", "src_column": "source", "dst_column": "target"}
    )
    # Reverse CSR: index column_name == "target" (src_column == "target").
    t.add_index(
        "source", {"type": "graph", "src_column": "target", "dst_column": "source"}
    )

    t.insert(_build_edges())
    t.commit()
    t.wait_for_background_tasks()
    return t


def _edge_set(res):
    df = res.to_pandas()
    return set(zip(df["source"].astype(int), df["target"].astype(int)))


def _neighbor_set(res):
    df = res.to_pandas()
    return set(df["neighbor"].astype(int))


def test_has_graph_index(csr_graph_table):
    assert csr_graph_table.has_graph_index("source") is True
    assert csr_graph_table.has_graph_index("target") is True
    assert csr_graph_table.has_graph_index("weight") is False


@pytest.mark.parametrize("hops", [1, 2])
@pytest.mark.parametrize("directed", [True, False])
def test_csr_subgraph_matches_sql(csr_graph_table, hops, directed):
    """CSR path must return exactly the same induced edges as the SQL path."""
    seeds = [0, 5, 17, 42, 99]

    sql_res = csr_graph_table._inner.subgraph(seeds, hops, directed)
    csr_res = csr_graph_table._inner.subgraph(seeds, hops, directed, "source")

    assert _edge_set(csr_res) == _edge_set(sql_res)
    # Sanity: the subgraph is non-trivial.
    assert len(_edge_set(sql_res)) > 0


@pytest.mark.parametrize("hops", [1, 2])
def test_csr_graph_neighbors_matches_sql(csr_graph_table, hops):
    """CSR-backed graph_neighbors must match the SQL BFS (directed)."""
    for node in (0, 5, 42):
        sql_res = csr_graph_table._inner.graph_neighbors(node, hops)
        csr_res = csr_graph_table._inner.graph_neighbors(node, hops, "source")
        assert _neighbor_set(csr_res) == _neighbor_set(sql_res)


def test_python_wrapper_subgraph_uses_csr(csr_graph_table):
    """The Python wrapper forwards graph_column to the CSR path and returns a
    table with payload columns preserved."""
    res = csr_graph_table.subgraph([0, 1, 2], hops=1, directed=True, graph_column="source")
    df = res.to_pandas()
    assert {"source", "target", "weight"}.issubset(set(df.columns))
    assert len(df) > 0


def test_undirected_without_reverse_falls_back(tmpdir):
    """With only a forward CSR, undirected subgraph must fall back to SQL and
    still return the correct (undirected) induced edges."""
    schema = pa.schema(
        [
            ("source", pa.uint64()),
            ("target", pa.uint64()),
            ("weight", pa.float64()),
        ]
    )
    t = Table.create(str(tmpdir.join("fwd_only")), schema)
    t.add_index(
        "target", {"type": "graph", "src_column": "source", "dst_column": "target"}
    )
    t.insert(_build_edges())
    t.commit()
    t.wait_for_background_tasks()

    seeds = [0, 5, 17]
    sql_res = t._inner.subgraph(seeds, 2, False)
    csr_res = t._inner.subgraph(seeds, 2, False, "source")
    assert _edge_set(csr_res) == _edge_set(sql_res)


# ── Max-degree truncation ───────────────────────────────────────────────
#
# A "super-node" (node 0) fans out to 10 leaves, each of which fans out to a
# unique grandchild. This mirrors the Wikipedia link network, where a handful
# of hub pages link to millions of others and dominate an unconstrained BFS.

SUPER_NODE = 0
SUPER_DEGREE = 10


def _build_star_edges():
    sources = []
    targets = []
    weights = []
    for leaf in range(1, SUPER_DEGREE + 1):
        sources.append(SUPER_NODE)
        targets.append(leaf)
        weights.append(1.0)
        sources.append(leaf)
        targets.append(100 + leaf)
        weights.append(1.0)
    return pa.Table.from_arrays(
        [
            pa.array(sources, type=pa.uint64()),
            pa.array(targets, type=pa.uint64()),
            pa.array(weights, type=pa.float64()),
        ],
        names=["source", "target", "weight"],
    )


@pytest.fixture
def star_graph_table(tmpdir):
    schema = pa.schema(
        [
            ("source", pa.uint64()),
            ("target", pa.uint64()),
            ("weight", pa.float64()),
        ]
    )
    t = Table.create(str(tmpdir.join("star_csr")), schema)
    t.add_index(
        "target", {"type": "graph", "src_column": "source", "dst_column": "target"}
    )
    t.insert(_build_star_edges())
    t.commit()
    t.wait_for_background_tasks()
    return t


def test_max_degree_truncates_super_node(star_graph_table):
    """A node above the degree cap is reported but not expanded."""
    # Untruncated: 1 hop reaches all 10 leaves.
    full = star_graph_table.subgraph(
        [SUPER_NODE], hops=1, directed=True, graph_column="source"
    )
    assert len(_edge_set(full)) == SUPER_DEGREE

    # Truncated below the super-node's degree: no expansion at all.
    truncated = star_graph_table.subgraph(
        [SUPER_NODE], hops=1, directed=True, graph_column="source", max_degree=5
    )
    assert _edge_set(truncated) == set()


def test_max_degree_allows_ordinary_nodes(star_graph_table):
    """Nodes at or below the cap are expanded normally."""
    # Cap == super-node degree: node 0 is expanded (degree is not > cap), and
    # the 2-hop traversal reaches the grandchildren too.
    res = star_graph_table.subgraph(
        [SUPER_NODE], hops=2, directed=True, graph_column="source", max_degree=SUPER_DEGREE
    )
    edges = _edge_set(res)
    assert len(edges) == 2 * SUPER_DEGREE
    assert (SUPER_NODE, 1) in edges
    assert (1, 101) in edges


def test_max_degree_blocks_multi_hop_expansion(star_graph_table):
    """Truncating the super-node also prevents reaching its grandchildren."""
    res = star_graph_table.subgraph(
        [SUPER_NODE], hops=2, directed=True, graph_column="source", max_degree=5
    )
    assert _edge_set(res) == set()


def test_graph_neighbors_max_degree(star_graph_table):
    """graph_neighbors honours max_degree on the CSR fast path."""
    full = star_graph_table.graph_neighbors(
        SUPER_NODE, hops=1, graph_column="source"
    )
    assert len(_neighbor_set(full)) == SUPER_DEGREE

    truncated = star_graph_table.graph_neighbors(
        SUPER_NODE, hops=1, graph_column="source", max_degree=5
    )
    assert _neighbor_set(truncated) == set()


def test_subgraph_nodes_matches_subgraph(csr_graph_table):
    """subgraph_nodes returns the same visited set without materializing edges."""
    seeds = [0, 5, 17]
    nodes = set(
        csr_graph_table.subgraph_nodes(
            seeds, hops=2, directed=True, graph_column="source"
        )
    )
    res = csr_graph_table.subgraph(
        seeds, hops=2, directed=True, graph_column="source"
    )
    df = res.to_pandas()
    expected = (
        set(seeds)
        | set(df["source"].astype(int))
        | set(df["target"].astype(int))
    )
    assert nodes == expected


def test_build_profile_binding():
    """The extension exposes a reliable build-profile indicator."""
    import benostreamdb

    assert benostreamdb.build_profile() in ("debug", "release")
    assert isinstance(benostreamdb.is_debug_build(), bool)
    assert benostreamdb.is_debug_build() == (benostreamdb.build_profile() == "debug")


# ── Graph-index direction / column mapping ──────────────────────────────
#
# Regression tests for the bug where `add_index` registered a graph index
# under BOTH its `src_column` and the original `column` argument. Two graph
# indexes (forward + reverse) then collided in `index_configs`, clobbering
# each other and mislabeling the physical CSR files, so the CSR fast path
# followed the wrong direction.


def test_graph_index_direction_mapping(csr_graph_table):
    """Forward CSR is keyed by its src_column ("source"); reverse by "target"."""
    assert csr_graph_table.has_graph_index("source") is True
    assert csr_graph_table.has_graph_index("target") is True

    # Forward CSR (graph_column="source") follows out-edges: 0 -> 1, 3, 11.
    fwd = _neighbor_set(
        csr_graph_table.graph_neighbors(0, hops=1, graph_column="source")
    )
    assert fwd == {1, 3, 11}

    # Reverse CSR (graph_column="target") follows in-edges: 99, 71, 53 -> 0.
    rev = _neighbor_set(
        csr_graph_table.graph_neighbors(0, hops=1, graph_column="target")
    )
    assert rev == {99, 71, 53}
    # 0 -> 1 is an out-edge, so it must not appear in the reverse neighbourhood.
    assert 1 not in rev


def test_forward_only_index_column_mapping(tmpdir):
    """A single forward graph index is keyed by its src_column only."""
    schema = pa.schema(
        [
            ("source", pa.uint64()),
            ("target", pa.uint64()),
            ("weight", pa.float64()),
        ]
    )
    t = Table.create(str(tmpdir.join("fwd_mapping")), schema)
    t.add_index(
        "target", {"type": "graph", "src_column": "source", "dst_column": "target"}
    )
    t.insert(_build_edges())
    t.commit()
    t.wait_for_background_tasks()

    assert t.has_graph_index("source") is True
    assert t.has_graph_index("target") is False
    # The ignored `column` argument must not register a spurious index column.
    assert "source" in t.index_columns
    assert "target" not in t.index_columns


def test_both_graph_indexes_register_both_src_columns(csr_graph_table):
    """Forward and reverse indexes coexist without clobbering each other."""
    cols = set(csr_graph_table.index_columns)
    assert {"source", "target"}.issubset(cols)


# ── Token-budget (max_nodes) cap ────────────────────────────────────────
#
# `max_nodes` is a hard cap on the total visited set (seeds included). The
# exact neighbours kept depend on CSR ordering, so these tests assert on the
# size and shape of the result rather than a specific node set.


def test_max_nodes_caps_visited_set(star_graph_table):
    """max_nodes bounds the visited set, so only 4 of the 10 leaves are kept."""
    res = star_graph_table.subgraph(
        [SUPER_NODE], hops=2, directed=True, graph_column="source", max_nodes=5
    )
    edges = _edge_set(res)
    # visited = {0} + 4 leaves; the induced edges are all 0 -> leaf.
    assert len(edges) == 4
    assert all(s == SUPER_NODE for s, _ in edges)


def test_max_nodes_seed_only(star_graph_table):
    """A cap equal to the seed count prevents any expansion."""
    res = star_graph_table.subgraph(
        [SUPER_NODE], hops=2, directed=True, graph_column="source", max_nodes=1
    )
    assert _edge_set(res) == set()


def test_max_nodes_larger_than_reachable_is_noop(star_graph_table):
    """A cap above the reachable set does not change the result."""
    full = _edge_set(
        star_graph_table.subgraph(
            [SUPER_NODE], hops=2, directed=True, graph_column="source"
        )
    )
    capped = _edge_set(
        star_graph_table.subgraph(
            [SUPER_NODE], hops=2, directed=True, graph_column="source", max_nodes=1000
        )
    )
    assert capped == full


def test_graph_neighbors_max_nodes(star_graph_table):
    """graph_neighbors honours max_nodes on the CSR fast path."""
    res = star_graph_table.graph_neighbors(
        SUPER_NODE, hops=2, graph_column="source", max_nodes=5
    )
    assert len(_neighbor_set(res)) == 4


def test_communities_csr_matches_udf(tmpdir):
    """CSR-backed community detection matches the SQL UDAF on a clear graph."""
    # Two disconnected triangles: the partition is unambiguous (2 communities).
    schema = pa.schema(
        [
            ("source", pa.uint64()),
            ("target", pa.uint64()),
            ("weight", pa.float64()),
        ]
    )
    edges = pa.Table.from_arrays(
        [
            pa.array([0, 1, 2, 3, 4, 5], type=pa.uint64()),
            pa.array([1, 2, 0, 4, 5, 3], type=pa.uint64()),
            pa.array([1.0] * 6, type=pa.float64()),
        ],
        names=["source", "target", "weight"],
    )
    t = Table.create(str(tmpdir.join("comm_csr")), schema)
    t.add_index(
        "target", {"type": "graph", "src_column": "source", "dst_column": "target"}
    )
    t.add_index(
        "source", {"type": "graph", "src_column": "target", "dst_column": "source"}
    )
    t.insert(edges)
    t.commit()
    t.wait_for_background_tasks()

    expected = {frozenset({0, 1, 2}), frozenset({3, 4, 5})}

    csr = {
        frozenset(int(x) for x in c)
        for c in t.communities(resolution=1.0).to_pandas()["community"]
    }
    udf = {
        frozenset(int(x) for x in c)
        for c in t.louvain_communities(resolution=1.0).to_pandas()["community"]
    }
    assert csr == expected
    assert udf == expected


def test_leiden_csr_communities_connected(tmpdir):
    """Leiden over the CSR returns only connected communities."""
    schema = pa.schema(
        [
            ("source", pa.uint64()),
            ("target", pa.uint64()),
            ("weight", pa.float64()),
        ]
    )
    # Two triangles joined by a bridge; plus a pendant leaf.
    edges = pa.Table.from_arrays(
        [
            pa.array([0, 1, 2, 3, 4, 5, 2, 5], type=pa.uint64()),
            pa.array([1, 2, 0, 4, 5, 3, 3, 6], type=pa.uint64()),
            pa.array([1.0] * 8, type=pa.float64()),
        ],
        names=["source", "target", "weight"],
    )
    t = Table.create(str(tmpdir.join("leiden_csr")), schema)
    t.add_index(
        "target", {"type": "graph", "src_column": "source", "dst_column": "target"}
    )
    t.add_index(
        "source", {"type": "graph", "src_column": "target", "dst_column": "source"}
    )
    t.insert(edges)
    t.commit()
    t.wait_for_background_tasks()

    df = t.communities(resolution=1.0, algorithm="leiden").to_pandas()
    communities = [set(int(x) for x in c) for c in df["community"]]

    # Valid partition covering every node exactly once.
    assert set().union(*communities) == {0, 1, 2, 3, 4, 5, 6}
    assert sum(len(c) for c in communities) == 7

    adj = {
        0: {1, 2},
        1: {0, 2},
        2: {0, 1, 3},
        3: {2, 4, 5},
        4: {3, 5},
        5: {3, 4, 6},
        6: {5},
    }
    for c in communities:
        # Every community must be connected.
        start = next(iter(c))
        seen = {start}
        stack = [start]
        while stack:
            x = stack.pop()
            for y in adj[x]:
                if y in c and y not in seen:
                    seen.add(y)
                    stack.append(y)
        assert seen == c


def test_graph_index_uses_v2_format(tmpdir):
    """CSR sidecars carry the v2 format suffix, so legacy v1 files are ignored."""
    import os

    schema = pa.schema(
        [
            ("source", pa.uint64()),
            ("target", pa.uint64()),
            ("weight", pa.float64()),
        ]
    )
    uri = str(tmpdir.join("v2_format"))
    t = Table.create(uri, schema)
    t.add_index(
        "target", {"type": "graph", "src_column": "source", "dst_column": "target"}
    )
    t.insert(_build_edges())
    t.commit()
    t.wait_for_background_tasks()

    files = []
    for _root, _dirs, names in os.walk(uri):
        files.extend(names)

    assert any(f.endswith(".graph_v2.csr.offsets") for f in files)
    assert any(f.endswith(".graph_v2.csr.edges") for f in files)
    assert any(f.endswith(".graph_v2.csr.dict") for f in files)
    # No legacy v1 sidecars must be produced.
    assert not any(f.endswith(".graph.csr.offsets") for f in files)


def test_update_communities_preserves_ids(tmpdir):
    """Warm-started updates keep stable community IDs across recomputes."""
    schema = pa.schema(
        [
            ("source", pa.uint64()),
            ("target", pa.uint64()),
            ("weight", pa.float64()),
        ]
    )
    # Two disconnected triangles.
    edges = pa.Table.from_arrays(
        [
            pa.array([0, 1, 2, 3, 4, 5], type=pa.uint64()),
            pa.array([1, 2, 0, 4, 5, 3], type=pa.uint64()),
            pa.array([1.0] * 6, type=pa.float64()),
        ],
        names=["source", "target", "weight"],
    )
    t = Table.create(str(tmpdir.join("update_comm")), schema)
    t.add_index(
        "target", {"type": "graph", "src_column": "source", "dst_column": "target"}
    )
    t.add_index(
        "source", {"type": "graph", "src_column": "target", "dst_column": "source"}
    )
    t.insert(edges)
    t.commit()
    t.wait_for_background_tasks()

    first = t.update_communities()
    df1 = first.to_pandas()
    assert {"community_id", "community"}.issubset(df1.columns)
    mapping1 = {
        frozenset(int(x) for x in c): int(cid)
        for cid, c in zip(df1["community_id"], df1["community"])
    }
    assert set(mapping1.keys()) == {frozenset({0, 1, 2}), frozenset({3, 4, 5})}

    # A warm-started recompute of the unchanged graph must preserve the IDs.
    second = t.update_communities(previous=first)
    df2 = second.to_pandas()
    mapping2 = {
        frozenset(int(x) for x in c): int(cid)
        for cid, c in zip(df2["community_id"], df2["community"])
    }
    assert mapping2 == mapping1
