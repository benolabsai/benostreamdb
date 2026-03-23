
import pytest
import importlib.util

def qdrant_not_installed():
    return importlib.util.find_spec("qdrant_client") is None

@pytest.mark.skipif(qdrant_not_installed(), reason="qdrant-client not installed")
def test_qdrant_install_and_query():
    """
    Test to ensure qdrant-client is installed and functional.
    Original logic from repro_qdrant.py.
    """
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance, PointStruct

    client = QdrantClient(":memory:")
    collection_name = "structure_test"
    
    # Create collection
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=4, distance=Distance.EUCLID)
    )
    
    # Upsert a point
    client.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(id=1, vector=[0.1, 0.1, 0.1, 0.1], payload={})
        ]
    )
    
    # Query the collection
    result = client.query_points(
        collection_name=collection_name,
        query=[0.1, 0.1, 0.1, 0.1],
        limit=1,
        with_vectors=True
    )
    
    # Assertions
    assert result is not None
    assert hasattr(result, 'points')
    assert len(result.points) > 0
    assert result.points[0].id == 1
    assert result.points[0].vector == pytest.approx([0.1, 0.1, 0.1, 0.1])
