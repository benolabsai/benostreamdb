"""Test hybrid search with pre-filtering on scalar columns."""
import os
import sys
import shutil
import tempfile
import numpy as np
import pandas as pd

sys.path.insert(0, './python')
from hyperstreamdb import Table
from sentence_transformers import SentenceTransformer

def test_hybrid_search_prefilter():
    """Test that hybrid search correctly applies scalar filters before vector search."""
    # Use a temporary directory for the test database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_news_db")

        # Create test data with category column
        texts = [
            "New breakthrough in quantum computing announced by researchers",
            "Stock market reaches all-time high amid economic recovery",
            "Olympic athlete wins gold medal in track and field",
            "Scientists discover new exoplanet in habitable zone",
            "Tech company releases revolutionary AI model",
            "International trade agreement signed between nations",
            "Basketball team wins championship in overtime",
            "Medical researchers develop new cancer treatment",
        ]
        categories = [
            "Sci/Tech", "Business", "Sports", "Sci/Tech",
            "Sci/Tech", "World", "Sports", "Sci/Tech"
        ]

        # Generate embeddings
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(texts)

        # Create DataFrame
        df = pd.DataFrame({
            'text': texts,
            'category': categories,
            'embedding': [list(e) for e in embeddings]
        })

        # Create table and ingest data
        table = Table(db_path)
        table.add_index_columns(["embedding", "category", "text"])
        table.write(df)
        table.commit()

        # Test hybrid search with pre-filter
        query = "Winning medals in international sports competitions"
        query_embedding = list(model.encode(query))

        hybrid_results = table.filter(
            filter="category = 'Sci/Tech'",
            vector_filter=query_embedding,
            k=5
        ).to_pandas()

        # Verify results
        assert len(hybrid_results) > 0, "Should return results"
        assert "category" in hybrid_results.columns, "Results should include category column"
        assert all(hybrid_results["category"] == "Sci/Tech"), "All results should be Sci/Tech"

        print(f"Test passed: {len(hybrid_results)} Sci/Tech results returned")
        print(hybrid_results[["category", "text", "distance"]])

if __name__ == "__main__":
    test_hybrid_search_prefilter()
