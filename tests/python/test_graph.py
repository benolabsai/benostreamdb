import pyarrow as pa
import pytest
from hyperstreamdb import Table
import os

@pytest.fixture
def graph_table(tmpdir):
    schema = pa.schema([
        ("source", pa.uint64()),
        ("target", pa.uint64()),
        ("weight", pa.float64())
    ])
    table_uri = str(tmpdir.join("test_graph"))
    t = Table.create(table_uri, schema)
    
    # Create a small sample graph
    # 1 -> 2 (weight 1.0)
    # 2 -> 3 (weight 2.0)
    # 3 -> 1 (weight 1.5)
    # 1 -> 4 (weight 0.5)
    # 4 -> 5 (weight 1.0)
    edges = pa.Table.from_arrays([
        pa.array([1, 2, 3, 1, 4], type=pa.uint64()),
        pa.array([2, 3, 1, 4, 5], type=pa.uint64()),
        pa.array([1.0, 2.0, 1.5, 0.5, 1.0], type=pa.float64())
    ], names=["source", "target", "weight"])
    
    t.insert(edges)
    return t

def test_pagerank(graph_table):
    pr_df = graph_table.pagerank(damping=0.85, iterations=30).to_pandas()
    assert len(pr_df) == 5
    pr_dict = dict(zip(pr_df['node'], pr_df['score']))
    assert 1 in pr_dict
    assert pr_dict[1] > 0

def test_shortest_path(graph_table):
    # Shortest path from 1 to 5
    sp_df = graph_table.shortest_path(1, 5).to_pandas()
    assert len(sp_df) >= 2
    nodes = set(sp_df['node'])
    assert 1 in nodes
    assert 5 in nodes

def test_connected_components(graph_table):
    cc_df = graph_table.connected_components().to_pandas()
    assert len(cc_df) > 0
    assert 'component' in cc_df.columns

def test_strongly_connected_components(graph_table):
    scc_df = graph_table.strongly_connected_components().to_pandas()
    assert len(scc_df) == 5
    assert 'scc_id' in scc_df.columns

def test_topological_sort(tmpdir):
    # Topo sort needs a DAG
    schema = pa.schema([("source", pa.uint64()), ("target", pa.uint64())])
    t = Table.create(str(tmpdir.join("dag")), schema)
    edges = pa.Table.from_arrays([
        pa.array([1, 2], type=pa.uint64()),
        pa.array([2, 3], type=pa.uint64())
    ], names=["source", "target"])
    t.insert(edges)
    
    ts_df = t.topological_sort().to_pandas()
    assert len(ts_df) == 3
    nodes = list(ts_df['node'])
    assert nodes.index(1) < nodes.index(2)
    assert nodes.index(2) < nodes.index(3)

def test_graph_neighbors(graph_table):
    neighbors_df = graph_table.graph_neighbors(1, 1).to_pandas()
    neighbors = set(neighbors_df['neighbor'])
    assert 2 in neighbors
    assert 4 in neighbors
    assert 3 not in neighbors

def test_degree_centrality(graph_table):
    dc_df = graph_table.degree_centrality().to_pandas()
    assert len(dc_df) == 5
    dc_dict = dict(zip(dc_df['node'], dc_df['degree']))
    # node 1 has 1 in (3->1), 2 out (1->2, 1->4) -> degree 3
    assert dc_dict[1] == 3

def test_louvain_communities(graph_table):
    lc_df = graph_table.louvain_communities().to_pandas()
    assert len(lc_df) > 0
    assert 'community' in lc_df.columns

def test_label_propagation(graph_table):
    lp_df = graph_table.label_propagation_communities().to_pandas()
    assert len(lp_df) > 0
    assert 'community' in lp_df.columns

def test_modularity(tmpdir):
    # modularity requires source_community and target_community
    schema = pa.schema([
        ("source", pa.uint64()),
        ("target", pa.uint64()),
        ("weight", pa.float64()),
        ("source_community", pa.uint64()),
        ("target_community", pa.uint64())
    ])
    t = Table.create(str(tmpdir.join("mod_graph")), schema)
    edges = pa.Table.from_arrays([
        pa.array([1, 2], type=pa.uint64()),
        pa.array([2, 3], type=pa.uint64()),
        pa.array([1.0, 1.0], type=pa.float64()),
        pa.array([1, 1], type=pa.uint64()),
        pa.array([1, 2], type=pa.uint64()),
    ], names=["source", "target", "weight", "source_community", "target_community"])
    t.insert(edges)
    mod_df = t.modularity().to_pandas()
    assert len(mod_df) == 1
    assert 'modularity' in mod_df.columns

def test_adamic_adar(graph_table):
    aa_df = graph_table.adamic_adar(1, 5).to_pandas()
    assert len(aa_df) == 1
    assert 'score' in aa_df.columns

def test_preferential_attachment(graph_table):
    pa_df = graph_table.preferential_attachment(1, 5).to_pandas()
    assert len(pa_df) == 1
    assert 'score' in pa_df.columns

def test_resource_allocation(graph_table):
    ra_df = graph_table.resource_allocation_index(1, 5).to_pandas()
    assert len(ra_df) == 1
    assert 'score' in ra_df.columns

def test_jaccard_coefficient(graph_table):
    jc_df = graph_table.jaccard_coefficient(1, 5).to_pandas()
    assert len(jc_df) == 1
    assert 'score' in jc_df.columns

def test_clustering_coefficient(graph_table):
    cc_df = graph_table.clustering_coefficient().to_pandas()
    assert len(cc_df) == 5
    cc_dict = dict(zip(cc_df['node'], cc_df['clustering_coefficient']))
    assert 1 in cc_dict

def test_to_graphviz(graph_table):
    dot_df = graph_table.to_graphviz().to_pandas()
    assert len(dot_df) == 1
    dot_str = dot_df['dot'][0]
    assert "digraph" in dot_str
