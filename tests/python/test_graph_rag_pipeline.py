import os
import tempfile
import pyarrow as pa
import pytest
from hyperstreamdb import Table, GraphRagResult

@pytest.fixture
def rag_doc_table(tmpdir):
    """
    Document/entity table for Graph RAG tests.
    8 entities across AI, enterprise tech, and semiconductor hardware.
    """
    schema = pa.schema([
        ("id", pa.uint64()),
        ("title", pa.string()),
        ("content", pa.string()),
        ("embedding", pa.list_(pa.float32(), 4)),
    ])
    uri = str(tmpdir.join("rag_docs"))
    table = Table.create(uri, schema)
    
    # 4D embeddings: AI domain ~ [1, 0, 0, 0], Hardware ~ [0, 1, 0, 0]
    data = pa.Table.from_arrays([
        pa.array([1, 2, 3, 4, 5, 6, 7, 8], type=pa.uint64()),
        pa.array([
            "OpenAI", "Sam Altman", "Y Combinator", "Microsoft", 
            "Satya Nadella", "TSMC", "Nvidia", "Jensen Huang"
        ], type=pa.string()),
        pa.array([
            "OpenAI develops frontier artificial intelligence and large language models like GPT-4.",
            "Sam Altman is the co-founder and CEO of OpenAI leading frontier AI research.",
            "Y Combinator is a prominent startup accelerator founded by Paul Graham.",
            "Microsoft is a global tech company investing heavily in OpenAI infrastructure.",
            "Satya Nadella is Chairman and CEO of Microsoft overseeing cloud AI partnerships.",
            "TSMC is a dedicated semiconductor foundry manufacturing advanced GPU silicon.",
            "Nvidia designs GPU hardware architectures and AI acceleration systems.",
            "Jensen Huang is founder and CEO of Nvidia pioneering accelerated computing.",
        ], type=pa.string()),
        pa.array([
            [1.0, 0.0, 0.0, 0.1],  # 1: OpenAI
            [0.9, 0.1, 0.0, 0.0],  # 2: Sam Altman
            [0.6, 0.2, 0.1, 0.1],  # 3: Y Combinator
            [0.8, 0.3, 0.0, 0.0],  # 4: Microsoft
            [0.7, 0.4, 0.0, 0.0],  # 5: Satya Nadella
            [0.0, 0.9, 0.2, 0.1],  # 6: TSMC
            [0.1, 1.0, 0.0, 0.0],  # 7: Nvidia
            [0.0, 0.95, 0.05, 0.0],# 8: Jensen Huang
        ], type=pa.list_(pa.float32(), 4)),
    ], names=["id", "title", "content", "embedding"])
    
    table.insert(data)
    table.commit()
    return table

@pytest.fixture
def rag_edge_table(tmpdir):
    """
    Edge table representing the relational knowledge graph connecting entities:
    2 -> 1 (founded)
    2 -> 3 (led)
    4 -> 1 (invested_in)
    5 -> 4 (leads)
    7 -> 1 (supplies)
    8 -> 7 (leads)
    7 -> 6 (manufactures_at)
    """
    uri = str(tmpdir.join("rag_edges"))
    schema = pa.schema([
        ("source", pa.uint64()),
        ("target", pa.uint64()),
        ("relation", pa.string()),
        ("weight", pa.float64()),
        ("timestamp", pa.string()),
    ])
    table = Table.create_edge_table(uri, schema)
    
    edges = pa.Table.from_arrays([
        pa.array([2, 2, 4, 5, 7, 8, 7], type=pa.uint64()),
        pa.array([1, 3, 1, 4, 1, 7, 6], type=pa.uint64()),
        pa.array([
            "founded", "led", "invested_in", "leads", 
            "supplies", "leads", "manufactures_at"
        ], type=pa.string()),
        pa.array([1.0, 0.8, 1.0, 1.0, 0.9, 1.0, 0.95], type=pa.float64()),
        pa.array([
            "2015-12-11T00:00:00Z", # founded
            "2014-02-21T00:00:00Z", # led
            "2019-07-22T00:00:00Z", # invested_in
            "2014-02-04T00:00:00Z", # leads (Microsoft)
            "2020-01-01T00:00:00Z", # supplies
            "1993-04-05T00:00:00Z", # leads (Nvidia)
            "2000-01-01T00:00:00Z", # manufactures_at
        ], type=pa.string()),
    ], names=["source", "target", "relation", "weight", "timestamp"])
    
    table.insert(edges)
    table.commit()
    return table

def test_graph_rag_local_search(rag_doc_table, rag_edge_table):
    """
    Test Local Search:
    Seed discovery via vector search -> Multi-hop induced subgraph ->
    Personalized PageRank grounding -> Context-enriched document ranking.
    """
    # Query vector close to Jensen Huang (8) and Nvidia (7)
    query_vec = [0.0, 0.98, 0.02, 0.0]
    
    result = rag_doc_table.graph_rag_search(
        query=query_vec,
        edge_table=rag_edge_table,
        mode="local",
        vector_column="embedding",
        id_column="id",
        top_k=1,
        hops=2,
        alpha=0.85,
        directed=False,
    )
    
    assert isinstance(result, GraphRagResult)
    assert result.mode == "local"
    # Seed should be entity 8 (Jensen Huang) or 7 (Nvidia)
    assert len(result.seeds) == 1
    assert result.seeds[0] in [7, 8]
    
    # Subgraph should be extracted around the seed
    assert not result.edges.empty
    subgraph_nodes = set(result.edges["source"]).union(set(result.edges["target"]))
    assert 7 in subgraph_nodes
    
    # Nodes DataFrame should be retrieved and ranked with PageRank
    assert not result.nodes.empty
    assert "id" in result.nodes.columns
    assert "title" in result.nodes.columns
    assert "content" in result.nodes.columns
    assert "pagerank" in result.nodes.columns
    
    # The seed and its immediate neighbor (Nvidia 7) should have highest PageRank
    top_node_id = result.nodes["id"].iloc[0]
    assert top_node_id in [7, 8]
    
    # Verify prompt-ready context formatting
    context = result.format_context(max_tokens=1000)
    assert "### Graph RAG Context (Mode: local)" in context
    assert "#### Discovered Entities (Ranked by Relevance)" in context
    assert "#### Relational Graph Context" in context
    assert "Nvidia" in context
    
    # Verify serialization
    d = result.to_dict()
    assert d["mode"] == "local"
    assert len(d["nodes"]) > 0
    assert len(d["edges"]) > 0

def test_graph_rag_reciprocal_invocation(rag_doc_table, rag_edge_table):
    """
    Test calling graph_rag_search on edge_table instead of doc_table.
    Both Table roles should be handled seamlessly.
    """
    query_vec = [0.95, 0.05, 0.0, 0.0]  # OpenAI query
    
    result = rag_edge_table.graph_rag_search(
        query=query_vec,
        doc_table=rag_doc_table,
        mode="local",
        top_k=1,
        hops=1,
    )
    
    assert isinstance(result, GraphRagResult)
    assert result.mode == "local"
    assert 1 in result.seeds or 2 in result.seeds
    assert not result.nodes.empty

def test_community_summarization_workflow(rag_doc_table, rag_edge_table, tmpdir):
    """
    Test Community Summarization Workflow:
    Multi-resolution Louvain community detection -> degree centrality hub ranking ->
    Document context aggregation -> Materialized community Iceberg table.
    """
    comm_uri = str(tmpdir.join("materialized_communities"))
    
    comm_table = rag_edge_table.summarize_communities(
        doc_table=rag_doc_table,
        target_uri=comm_uri,
        id_column="id",
        content_column="content",
        resolution=1.0,
        top_entities_per_comm=3,
    )
    
    assert isinstance(comm_table, Table)
    df = comm_table.execute_sql("SELECT * FROM t").to_pandas()
    
    assert len(df) > 0
    assert "community_id" in df.columns
    assert "title" in df.columns
    assert "member_count" in df.columns
    assert "members" in df.columns
    assert "top_entities" in df.columns
    assert "summary" in df.columns
    
    # Each community should have members and non-empty summary
    for _, row in df.iterrows():
        assert row["member_count"] > 0
        assert len(row["members"]) == row["member_count"]
        assert len(row["top_entities"]) <= 3
        assert len(row["summary"]) > 0

def test_graph_rag_global_search(rag_doc_table, rag_edge_table):
    """
    Test Global Search:
    Community clustering -> Corpus-wide synthesis and key entity extraction.
    """
    query_vec = [0.0, 1.0, 0.0, 0.0]  # Hardware query
    
    result = rag_doc_table.graph_rag_search(
        query=query_vec,
        edge_table=rag_edge_table,
        mode="global",
        top_k=2,
    )
    
    assert isinstance(result, GraphRagResult)
    assert result.mode == "global"
    assert result.communities is not None
    assert not result.communities.empty
    
    # Verify global context formatting
    context = result.format_context()
    assert "### Graph RAG Context (Mode: global)" in context
    assert "#### Discovered Communities" in context
    assert "#### Discovered Entities" in context

def test_graph_rag_temporal_filtering(rag_doc_table, rag_edge_table):
    """
    Test Local Search with Temporal Filter:
    Should only find edges within the timestamp window.
    """
    query_vec = [0.9, 0.1, 0.0, 0.1] # Near OpenAI
    
    # Filter for edges >= 2018 (should exclude 'founded' and 'led' but include 'invested_in' and 'supplies')
    result = rag_doc_table.graph_rag_search(
        query=query_vec,
        edge_table=rag_edge_table,
        mode="local",
        vector_column="embedding",
        top_k=2,
        hops=2,
        time_column="timestamp",
        time_start="2018-01-01T00:00:00Z"
    )
    
    edges_df = result.edges
    assert not edges_df.empty
    
    # 'founded' edge was 2015 (2 -> 1), should not be present
    # 'led' edge was 2014 (2 -> 3), should not be present
    # 'invested_in' edge was 2019 (4 -> 1), should be present
    # 'supplies' edge was 2020 (7 -> 1), should be present
    
    sources = edges_df["source"].tolist()
    targets = edges_df["target"].tolist()
    nodes = set(sources + targets)
    
    # Node 2 (Sam Altman) was only connected via pre-2018 edges
    assert 2 not in nodes
    
    # Nodes 4 (Microsoft) and 7 (Nvidia) connected post-2018
    assert 4 in nodes or 7 in nodes

def test_hybrid_search_fusion(rag_doc_table):
    """
    Test hybrid search combining BM25 keyword search and dense vector search via RRF.
    """
    # Keyword: "semiconductor", Vector: [0.0, 1.0, 0.0, 0.0]
    results = rag_doc_table.hybrid_search(
        text_column="content",
        query_text="semiconductor",
        vector_column="embedding",
        query_vector=[0.0, 1.0, 0.0, 0.0],
        k=3,
        rrf_k=60,
    )
    
    assert len(results) > 0
    assert "rrf_score" in results.columns
    assert "id" in results.columns
    
    # Results should be sorted descending by RRF score
    scores = results["rrf_score"].tolist()
    assert scores == sorted(scores, reverse=True)
    
    # TSMC (6) and Nvidia (7) should be at or near the top
    top_ids = results["id"].tolist()
    assert 6 in top_ids or 7 in top_ids


def test_hippo_rag_weighted_ppr(rag_edge_table):
    """
    Test HippoRAG-style continuous seed weighting in Personalized PageRank.
    Shifting weight to node 7 (Nvidia) vs node 1 (OpenAI) shifts probability distribution.
    """
    seeds = [1, 7]
    
    # Run with heavy weight on 7
    res_heavy_7 = rag_edge_table.personalized_pagerank(seeds, seed_weights=[0.05, 0.95])
    df_heavy_7 = res_heavy_7.to_pandas()
    score_7_heavy = df_heavy_7[df_heavy_7["node"] == 7]["score"].iloc[0]
    score_1_light = df_heavy_7[df_heavy_7["node"] == 1]["score"].iloc[0]
    assert score_7_heavy > score_1_light

    # Run with heavy weight on 1
    res_heavy_1 = rag_edge_table.personalized_pagerank(seeds, seed_weights=[0.95, 0.05])
    df_heavy_1 = res_heavy_1.to_pandas()
    score_1_heavy = df_heavy_1[df_heavy_1["node"] == 1]["score"].iloc[0]
    score_7_light = df_heavy_1[df_heavy_1["node"] == 7]["score"].iloc[0]
    assert score_1_heavy > score_7_light


def test_relation_filtered_subgraph(rag_edge_table):
    """
    Test relation / predicate pushdown into multi-hop induced subgraph extraction.
    Only allowed relations should be traversed during subgraph expansion.
    """
    # From seed 4 (Microsoft), edges in dataset: 4->1 (invested_in), 5->4 (leads)
    res_invested = rag_edge_table.subgraph(seeds=[4], hops=1, allowed_relations=["invested_in"])
    df_invested = res_invested.to_pandas()
    assert not df_invested.empty
    invested_nodes = set(df_invested["source"]).union(set(df_invested["target"]))
    assert 1 in invested_nodes
    assert 5 not in invested_nodes  # 5->4 (leads) was pruned

    res_leads = rag_edge_table.subgraph(seeds=[4], hops=1, allowed_relations=["leads"])
    df_leads = res_leads.to_pandas()
    assert not df_leads.empty
    leads_nodes = set(df_leads["source"]).union(set(df_leads["target"]))
    assert 5 in leads_nodes
    assert 1 not in leads_nodes  # 4->1 (invested_in) was pruned


def test_dual_vector_graph_rag(rag_doc_table, tmpdir):
    """
    Test Dual Vector-Graph RAG (Semantic Edge Search).
    Edges with vector embeddings are searched in parallel with document entities.
    """
    uri = str(tmpdir.join("semantic_edges"))
    schema = pa.schema([
        ("source", pa.uint64()),
        ("target", pa.uint64()),
        ("relation", pa.string()),
        ("weight", pa.float64()),
        ("embedding", pa.list_(pa.float32(), 4)),
    ])
    edge_tbl = Table.create_edge_table(uri, schema, embedding_dim=4)
    
    # Edge embeddings: edge 7->6 has [0.0, 1.0, 0.0, 0.0]
    edges = pa.Table.from_arrays([
        pa.array([2, 4, 7], type=pa.uint64()),
        pa.array([1, 1, 6], type=pa.uint64()),
        pa.array(["founded", "invested_in", "manufactures_at"], type=pa.string()),
        pa.array([1.0, 1.0, 1.0], type=pa.float64()),
        pa.array([
            [0.9, 0.1, 0.0, 0.0],
            [0.8, 0.2, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],  # 7 -> 6
        ], type=pa.list_(pa.float32(), 4)),
    ], names=["source", "target", "relation", "weight", "embedding"])
    
    edge_tbl.insert(edges)
    edge_tbl.commit()

    # Query with hardware vector [0.0, 0.99, 0.01, 0.0] and search_edges=True
    result = rag_doc_table.graph_rag_search(
        query=[0.0, 0.99, 0.01, 0.0],
        edge_table=edge_tbl,
        mode="local",
        search_edges=True,
        top_k=2,
    )

    assert isinstance(result, GraphRagResult)
    assert not result.nodes.empty
    assert 6 in result.seeds or 7 in result.seeds


def test_hierarchical_community_summarization(rag_doc_table, rag_edge_table, tmpdir):
    """
    Test Hierarchical Community Summarization producing multi-level Louvain pyramids.
    """
    comm_uri = str(tmpdir.join("hierarchical_communities"))
    
    comm_table = rag_edge_table.summarize_communities(
        doc_table=rag_doc_table,
        target_uri=comm_uri,
        id_column="id",
        content_column="content",
        hierarchical=True,
        resolutions=[0.5, 1.5],
        top_entities_per_comm=3,
    )
    
    df = comm_table.execute_sql("SELECT * FROM t").to_pandas()
    assert not df.empty
    assert "level" in df.columns
    assert "parent_community_id" in df.columns
    
    levels = set(df["level"])
    assert 0 in levels
    assert 1 in levels

    # Global search using hierarchical community table
    result = rag_doc_table.graph_rag_search(
        query=[0.0, 1.0, 0.0, 0.0],
        edge_table=rag_edge_table,
        community_table=comm_table,
        mode="global",
        top_k=5,
    )
    assert result.mode == "global"
    context = result.format_context()
    assert "#### Level 0 Communities" in context or "#### Level 1 Communities" in context


def test_entity_equivalence_resolution(tmpdir):
    """
    Test Entity Equivalence Resolution (transitive alias closure over same_as edges).
    """
    uri = str(tmpdir.join("alias_edges"))
    schema = pa.schema([
        ("source", pa.uint64()),
        ("target", pa.uint64()),
        ("relation", pa.string()),
    ])
    table = Table.create_edge_table(uri, schema, with_weight=False)
    
    data = pa.Table.from_arrays([
        pa.array([10, 20, 40], type=pa.uint64()),
        pa.array([20, 30, 50], type=pa.uint64()),
        pa.array(["same_as", "same_as", "same_as"], type=pa.string()),
    ], names=["source", "target", "relation"])
    table.insert(data)
    table.commit()

    mapping = table.resolve_entities(relation="same_as")
    assert mapping[10] == 10
    assert mapping[20] == 10
    assert mapping[30] == 10
    assert mapping[40] == 40
    assert mapping[50] == 40
