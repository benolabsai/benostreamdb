import pytest
import pyarrow as pa
from hyperstreamdb import Table
import tempfile
import os

@pytest.fixture
def graph_table(tmpdir):
    uri = str(tmpdir.join("graph_edges"))
    schema = pa.schema([
        ("source", pa.uint64()),
        ("target", pa.uint64()),
        ("weight", pa.float32()),
        ("relation", pa.string()),
    ])
    table = Table.create(uri, schema)
    
    data = pa.Table.from_arrays([
        pa.array([1, 2, 3], type=pa.uint64()),
        pa.array([2, 3, 1], type=pa.uint64()),
        pa.array([0.5, 0.8, 1.2], type=pa.float32()),
        pa.array(["knows", "likes", "follows"], type=pa.string()),
    ], names=["source", "target", "weight", "relation"])
    
    table.insert(data)
    table.commit()
    return table

def test_export_graph_gml(graph_table):
    with tempfile.NamedTemporaryFile(suffix=".gml", delete=False) as f:
        filepath = f.name
    
    try:
        graph_table.export_graph_gml(filepath)
        assert os.path.exists(filepath)
        
        with open(filepath, "r") as f:
            content = f.read()
            assert "graph [" in content
            assert "node [" in content
            assert "edge [" in content
            assert "source 1" in content
            assert "target 2" in content
            assert 'relation "knows"' in content
            assert "weight 0.5" in content
    finally:
        os.remove(filepath)

def test_export_graph_graphml(graph_table):
    with tempfile.NamedTemporaryFile(suffix=".graphml", delete=False) as f:
        filepath = f.name
    
    try:
        graph_table.export_graph_graphml(filepath)
        assert os.path.exists(filepath)
        
        with open(filepath, "r") as f:
            content = f.read()
            assert "<graphml" in content
            assert "<graph" in content
            assert "<node id=\"1\"/>" in content
            assert "<edge source=\"1\" target=\"2\">" in content
            assert "<data key=\"relation\">knows</data>" in content
    finally:
        os.remove(filepath)
