# Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.
# Regression test suite for Arrow C FFI string offset conformance and hybrid search.

import os
import shutil
import tempfile
import benostreamdb as bsdb


def test_vector_search_with_string_columns_no_offset_corruption():
    """Verify that vector search correctly projects string columns without offset corruption

    or C++ basic_string::_M_create length errors across the Arrow C FFI stream boundary.
    """
    tmpdir = tempfile.mkdtemp(prefix="benostream_test_str_")
    try:
        table_uri = "file://" + tmpdir
        table = bsdb.Table(table_uri)
        table.add_index("vec", "hnsw")
        table.add_index("title", "bm25")

        records = [
            {"id": 1, "title": "Lakehouse Vector Search", "category": "ai", "vec": [0.12, 0.54, 0.89]},
            {"id": 2, "title": "Graph RAG on Iceberg", "category": "ai", "vec": [0.18, 0.49, 0.82]},
            {"id": 3, "title": "Serverless Database Engine", "category": "database", "vec": [0.85, 0.12, 0.28]},
            {"id": 4, "title": "Hybrid Search with BM25", "category": "ai", "vec": [0.15, 0.62, 0.81]},
            {"id": 5, "title": "CSR Graph Algorithms", "category": "graph", "vec": [0.72, 0.35, 0.14]},
        ]
        table.write(records)
        table.commit()

        # 1. Vector Search with Top-3
        results = table.vector_search("vec", [0.10, 0.50, 0.85], k=3)
        assert len(results) == 3

        # Verify exact string values across rows (no offset shifting or empty strings)
        expected_titles = [
            "Lakehouse Vector Search",
            "Graph RAG on Iceberg",
            "Hybrid Search with BM25",
        ]
        assert list(results["title"]) == expected_titles
        assert list(results["id"]) == [1, 2, 4]
        assert list(results["category"]) == ["ai", "ai", "ai"]

        # 2. Verify explicit projection
        proj_results = table.vector_search("vec", [0.10, 0.50, 0.85], k=3, columns=["id", "title", "distance"])
        assert list(proj_results.columns) == ["id", "title", "distance"]
        assert list(proj_results["title"]) == expected_titles

        # 3. Hybrid Search (RRF)
        hybrid_res = table.hybrid_search(
            text_column="title",
            query_text="Lakehouse",
            vector_column="vec",
            query_vector=[0.10, 0.50, 0.85],
            k=3,
        )
        assert len(hybrid_res) == 3
        assert list(hybrid_res["id"]) == [1, 2, 4]
        assert list(hybrid_res["title"]) == expected_titles
        assert "rrf_score" in hybrid_res.columns
        assert hybrid_res["rrf_score"].iloc[0] > hybrid_res["rrf_score"].iloc[1]
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    test_vector_search_with_string_columns_no_offset_corruption()
    print("ALL TESTS PASSED")
