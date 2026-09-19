import pytest
import pandas as pd
import pyarrow as pa
from hyperstreamdb import Table
import os

def setup_test_tables(tmp_path):
    # Create simple edge table
    edge_schema = pa.schema([
        pa.field("source", pa.uint64()),
        pa.field("target", pa.uint64())
    ])
    
    edge_uri = f"file://{tmp_path}/edges"
    edge_table = Table.create(edge_uri, edge_schema)
    
    # Graph: (1->2), (2->3), (3->1) [Community 0] and (4->5), (5->6), (6->4) [Community 1], (3->4) bridging
    edges_df = pd.DataFrame({
        "source": [1, 2, 3, 4, 5, 6, 3],
        "target": [2, 3, 1, 5, 6, 4, 4]
    })
    edge_table.insert(pa.Table.from_pandas(edges_df))
    edge_table.commit()

    # Create community table manually for test
    comm_schema = pa.schema([
        pa.field("community_id", pa.uint64()),
        pa.field("member_count", pa.uint32()),
        pa.field("members", pa.list_(pa.uint64())),
        pa.field("seed_overlap", pa.uint32())
    ])
    
    comm_uri = f"file://{tmp_path}/communities"
    comm_table = Table.create(comm_uri, comm_schema)
    
    comm_df = pd.DataFrame({
        "community_id": [0, 1],
        "member_count": [3, 3],
        "members": [[1, 2, 3], [4, 5, 6]],
        "seed_overlap": [1, 0]
    })
    comm_table.insert(pa.Table.from_pandas(comm_df))
    comm_table.commit()

    return edge_table, comm_table

def test_drift_search_heuristic(tmp_path):
    edge_table, comm_table = setup_test_tables(tmp_path)
    
    # Run drift search without LLM (uses heuristic)
    res = edge_table.drift_search(
        query="test query",
        community_table=comm_table,
        n_depth=2,
        k_followups=2,
        top_k=2,
        hops=1
    )
    
    assert "all_discovered_nodes" in res
    assert "actions" in res
    assert len(res["actions"]) > 0

def test_drift_search_with_llm_callback(tmp_path):
    edge_table, comm_table = setup_test_tables(tmp_path)
    
    # Mock LLM callback
    def mock_llm(**kwargs):
        phase = kwargs.get("phase")
        if phase == "primer":
            return [("Follow up from primer", 0.9, [1, 2])]
        elif phase == "follow_up":
            round_num = kwargs.get("round_num", 0)
            if round_num < 2:
                return [("Follow up from deeper", 0.8, [3, 4])]
            return []
            
    res = edge_table.drift_search(
        query="test query",
        community_table=comm_table,
        follow_up_llm=mock_llm,
        n_depth=2,
        k_followups=2,
        top_k=2,
        hops=1
    )
    
    assert "all_discovered_nodes" in res
    assert "actions" in res
    
    actions = res["actions"]
    assert len(actions) > 0
    queries = [a["query"] for a in actions]
    assert any("Follow up from primer" in q for q in queries)
