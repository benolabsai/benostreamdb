import pyarrow as pa
import pytest
from hyperstreamdb import Table

@pytest.fixture
def rag_knowledge_graph(tmpdir):
    """
    Sample Knowledge Graph representing entities and relationships:
    1: OpenAI
    2: Sam Altman
    3: Y Combinator
    4: Microsoft
    5: Satya Nadella
    6: TSMC
    7: Nvidia
    8: Jensen Huang
    
    Edges (directed relations):
    2 -> 1 (Sam Altman founded/leads OpenAI)
    2 -> 3 (Sam Altman led Y Combinator)
    4 -> 1 (Microsoft invested in OpenAI)
    5 -> 4 (Satya Nadella leads Microsoft)
    7 -> 1 (Nvidia supplies OpenAI)
    8 -> 7 (Jensen Huang leads Nvidia)
    7 -> 6 (Nvidia manufactures at TSMC)
    """
    schema = pa.schema([
        ("source", pa.uint64()),
        ("target", pa.uint64()),
        ("weight", pa.float64()),
    ])
    table_uri = str(tmpdir.join("rag_kg"))
    t = Table.create(table_uri, schema)

    edges = pa.Table.from_arrays([
        pa.array([2, 2, 4, 5, 7, 8, 7], type=pa.uint64()),
        pa.array([1, 3, 1, 4, 1, 7, 6], type=pa.uint64()),
        pa.array([1.0, 0.8, 1.0, 1.0, 0.9, 1.0, 0.95], type=pa.float64()),
    ], names=["source", "target", "weight"])

    t.insert(edges)
    return t

def test_personalized_pagerank_seed_activation(rag_knowledge_graph):
    """
    Personalized PageRank (PPR) should concentrate probability mass around the seed entities.
    Seeds: [8] (Jensen Huang) -> Should highly rank 8 (seed), 7 (Nvidia), and 1 (OpenAI) / 6 (TSMC).
    """
    ppr_df = rag_knowledge_graph.personalized_pagerank(
        seeds=[8],
        damping=0.85,
        iterations=30,
        directed=False
    ).to_pandas()

    assert len(ppr_df) > 0
    scores = dict(zip(ppr_df['node'], ppr_df['score']))

    # Seed 8 should have high score
    assert 8 in scores
    assert 7 in scores
    # Jensen Huang (8) -> Nvidia (7) should be strongly activated
    assert scores[8] > 0
    assert scores[7] > 0
    # Closer entities (7) should have higher score than distant entities (3)
    if 3 in scores:
        assert scores[7] > scores[3]

def test_induced_subgraph_extraction(rag_knowledge_graph):
    """
    Extract induced subgraph around seeds [2, 4] (Sam Altman, Microsoft) within 1 hop.
    Should extract edges connecting 2, 4, and 1 (OpenAI).
    """
    sub_df = rag_knowledge_graph.subgraph(seeds=[2, 4], hops=1, directed=False).to_pandas()
    assert len(sub_df) > 0
    assert 'source' in sub_df.columns
    assert 'target' in sub_df.columns
    assert 'weight' in sub_df.columns

    extracted_nodes = set(sub_df['source']).union(set(sub_df['target']))
    # Both seeds and common target 1 (OpenAI) must be present
    assert 2 in extracted_nodes
    assert 4 in extracted_nodes
    assert 1 in extracted_nodes
    # Entity 6 (TSMC) is 3 hops away from seeds [2, 4], must not be in 1-hop subgraph
    assert 6 not in extracted_nodes

def test_multi_seed_connecting_paths(rag_knowledge_graph):
    """
    Find connecting paths between distant query entities:
    Seeds: [8, 5] (Jensen Huang, Satya Nadella).
    Path: 8 -> 7 -> 1 <- 4 <- 5
    """
    paths_df = rag_knowledge_graph.connecting_paths(seeds=[8, 5], directed=False).to_pandas()
    assert len(paths_df) > 0
    assert 'source' in paths_df.columns
    assert 'target' in paths_df.columns

    path_nodes = set(paths_df['source']).union(set(paths_df['target']))
    assert 8 in path_nodes
    assert 5 in path_nodes
    assert 1 in path_nodes # Central bridge (OpenAI)

def test_louvain_resolution_macro_vs_micro(tmpdir):
    """
    Test Louvain multi-resolution clustering:
    Two dense cliques connected by a single weak bridge:
    Clique A: 1, 2, 3
    Clique B: 4, 5, 6
    Bridge: 3 -- 4
    """
    schema = pa.schema([
        ("source", pa.uint64()),
        ("target", pa.uint64()),
        ("weight", pa.float64()),
    ])
    t = Table.create(str(tmpdir.join("clique_graph")), schema)
    edges = pa.Table.from_arrays([
        pa.array([1, 2, 1, 4, 5, 4, 3], type=pa.uint64()),
        pa.array([2, 3, 3, 5, 6, 6, 4], type=pa.uint64()),
        pa.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1], type=pa.float64()),
    ], names=["source", "target", "weight"])
    t.insert(edges)

    # Standard resolution (1.0) should easily separate the two cliques
    comm_df = t.louvain_communities(resolution=1.0).to_pandas()
    assert len(comm_df) >= 2
