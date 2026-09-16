import os
import shutil
import pytest
import pyarrow as pa
import networkx as nx
import numpy as np
from hyperstreamdb import Table

@pytest.fixture
def temp_dir(tmpdir):
    d = str(tmpdir.join("edge_tables"))
    os.makedirs(d, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)

def test_create_edge_table_standard(temp_dir):
    uri = f"file://{temp_dir}/t_standard"
    table = Table.create_edge_table(uri, node_id_type="uint64", with_relation=True, with_weight=True)
    
    assert table.is_edge_table()
    endpoints = table.edge_endpoints()
    assert endpoints == ("source", "target")
    
    cols = table.columns
    assert "source" in cols
    assert "target" in cols
    assert "relation" in cols
    assert "weight" in cols

def test_networkx_roundtrip_digraph(temp_dir):
    # Create a directed NetworkX graph with attributes
    G = nx.DiGraph()
    G.add_edge(1, 2, weight=2.5, relation="cites", label="docA_docB")
    G.add_edge(2, 3, weight=1.0, relation="mentions", label="docB_docC")
    G.add_edge(3, 1, weight=0.8, relation="related_to", label="docC_docA")
    
    uri = f"file://{temp_dir}/t_digraph"
    table = Table.from_networkx(uri, G, index_endpoints=True)
    table.commit()
    
    assert table.is_edge_table()
    assert len(table) == 3
    
    # Round-trip back to NetworkX
    G_out = table.to_networkx(directed=True)
    
    assert isinstance(G_out, nx.DiGraph)
    assert set(G_out.nodes()) == {1, 2, 3}
    assert G_out.has_edge(1, 2)
    assert G_out.has_edge(2, 3)
    assert G_out.has_edge(3, 1)
    
    # Check attributes preserved
    e12 = G_out[1][2]
    assert e12["weight"] == 2.5
    assert e12["relation"] == "cites"
    assert e12["label"] == "docA_docB"

def test_networkx_roundtrip_undirected_graph(temp_dir):
    G = nx.Graph()
    G.add_edge(10, 20, weight=3.0, relation="co_authors")
    G.add_edge(20, 30, weight=1.5, relation="co_authors")
    
    uri = f"file://{temp_dir}/t_graph"
    table = Table.from_networkx(uri, G)
    table.commit()
    
    G_out = table.to_networkx(directed=False)
    assert isinstance(G_out, nx.Graph)
    assert not G_out.is_directed()
    assert G_out.has_edge(10, 20)
    assert G_out.has_edge(20, 10)
    assert G_out[10][20]["weight"] == 3.0

def test_networkx_multidigraph(temp_dir):
    G = nx.MultiDiGraph()
    G.add_edge(1, 2, key="cites", weight=1.0, timestamp=2023)
    G.add_edge(1, 2, key="disputes", weight=2.0, timestamp=2024)
    
    uri = f"file://{temp_dir}/t_multidigraph"
    table = Table.from_networkx(uri, G)
    table.commit()
    
    G_out = table.to_networkx(directed=True, multigraph=True)
    assert isinstance(G_out, nx.MultiDiGraph)
    assert G_out.number_of_edges(1, 2) == 2

def test_edge_table_with_embedding_vector_search(temp_dir):
    # Edge table with 4-dimensional embeddings
    uri = f"file://{temp_dir}/t_edge_vectors"
    dim = 4
    table = Table.create_edge_table(
        uri,
        node_id_type="uint64",
        embedding_dim=dim,
        index_endpoints=True,
        index_embedding=True
    )
    
    # Create 3 edges with embeddings
    # Edge (1 -> 2): semantic relationship "acquisition"
    # Edge (2 -> 3): semantic relationship "partnership"
    # Edge (3 -> 4): semantic relationship "lawsuit"
    data = pa.Table.from_arrays([
        pa.array([1, 2, 3], type=pa.uint64()),
        pa.array([2, 3, 4], type=pa.uint64()),
        pa.array(["acquired", "partnered", "sued"], type=pa.string()),
        pa.array([1.0, 1.0, 1.0], type=pa.float64()),
        pa.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ], type=pa.list_(pa.float32(), dim))
    ], names=["source", "target", "relation", "weight", "embedding"])
    
    table.write(data)
    table.commit()
    
    assert len(table) == 3
    
    # Vector search on edge embedding: search for query close to [1.0, 0.0, 0.0, 0.0]
    query_vec = [0.95, 0.05, 0.0, 0.0]
    res = table.search("embedding", query_vec, k=1)
    import pandas as pd
    df_res = res if isinstance(res, pd.DataFrame) else res.to_pandas()
    
    assert len(df_res) == 1
    assert df_res.iloc[0]["relation"] == "acquired"
    assert df_res.iloc[0]["source"] == 1
    assert df_res.iloc[0]["target"] == 2
    
    # Hybrid Graph RAG query: use retrieved endpoints as seeds for subgraph expansion!
    seed_nodes = [int(df_res.iloc[0]["source"]), int(df_res.iloc[0]["target"])]
    subgraph_df = table.subgraph(seed_nodes, hops=1).to_pandas()
    assert len(subgraph_df) >= 1
    assert 1 in set(subgraph_df["source"]).union(set(subgraph_df["target"]))

def test_graph_rag_on_networkx_table(temp_dir):
    # Ingest a graph from NetworkX and run Graph RAG functions
    G = nx.path_graph(5, create_using=nx.DiGraph)  # 0 -> 1 -> 2 -> 3 -> 4
    
    uri = f"file://{temp_dir}/t_path_graph"
    table = Table.from_networkx(uri, G)
    table.commit()
    
    # Run Personalized PageRank biased toward node 0
    ppr_df = table.personalized_pagerank([0], alpha=0.85).to_pandas()
    assert len(ppr_df) == 5
    ppr_dict = dict(zip(ppr_df['node'], ppr_df['score']))
    assert ppr_dict[0] > ppr_dict[4]
    
    # Run connecting paths between 0 and 4
    paths_df = table.connecting_paths([0, 4], max_depth=5).to_pandas()
    assert len(paths_df) >= 4
    edges_found = set(zip(paths_df['source'], paths_df['target']))
    assert (0, 1) in edges_found
    assert (3, 4) in edges_found
