"""Correctness tests for the vector aggregate helpers and text-scoring UDFs.

Each helper is compared against an independent reference implementation
(numpy for the vector aggregates; hand-computed values for the corpus-free
BM25 / TF-IDF variants), so a regression in the math is caught even if the
Rust and Python implementations drift together.
"""

import math

import numpy as np
import pytest

import benostreamdb


# ---------------------------------------------------------------------------
# Vector aggregates vs numpy
# ---------------------------------------------------------------------------

VECTORS = [
    [1.0, 5.0, -2.0],
    [3.0, 2.0, 0.0],
    [2.0, 4.0, 7.0],
    [8.0, -1.0, 3.0],
]


def test_centroid_matches_numpy():
    got = benostreamdb.centroid(VECTORS)
    expected = np.mean(np.array(VECTORS), axis=0).tolist()
    assert got == pytest.approx(expected)


def test_vector_min_matches_numpy():
    got = benostreamdb.vector_min(VECTORS)
    expected = np.min(np.array(VECTORS), axis=0).tolist()
    assert got == pytest.approx(expected)


def test_vector_max_matches_numpy():
    got = benostreamdb.vector_max(VECTORS)
    expected = np.max(np.array(VECTORS), axis=0).tolist()
    assert got == pytest.approx(expected)


def test_vector_stddev_matches_numpy_population():
    got = benostreamdb.vector_stddev(VECTORS)
    # numpy's default ddof=0 is the population standard deviation.
    expected = np.std(np.array(VECTORS), axis=0).tolist()
    assert got == pytest.approx(expected)


def test_vector_median_matches_numpy():
    got = benostreamdb.vector_median(VECTORS)
    expected = np.median(np.array(VECTORS), axis=0).tolist()
    assert got == pytest.approx(expected)


def test_vector_median_even_count_matches_numpy():
    vectors = [[1.0], [2.0], [3.0], [4.0]]
    got = benostreamdb.vector_median(vectors)
    expected = np.median(np.array(vectors), axis=0).tolist()
    assert got == pytest.approx(expected)


def test_dimension_mismatch_raises():
    with pytest.raises(ValueError):
        benostreamdb.centroid([[1.0, 2.0], [3.0]])


def test_empty_input_returns_none():
    assert benostreamdb.centroid([]) is None
    assert benostreamdb.vector_min([]) is None
    assert benostreamdb.vector_max([]) is None
    assert benostreamdb.vector_stddev([]) is None
    assert benostreamdb.vector_median([]) is None


# ---------------------------------------------------------------------------
# BM25 / TF-IDF vs hand-computed references
# ---------------------------------------------------------------------------

K1 = 1.2


def _reference_bm25(text, query, k1=K1):
    """Corpus-free BM25: sum_t tf*(k1+1)/(tf+k1)."""
    import re
    from collections import Counter

    def tok(s):
        return [t for t in re.sub(r"[^0-9a-zA-Z]+", " ", s).lower().split() if t]

    tf = Counter(tok(text))
    return sum(
        tf[t] * (k1 + 1.0) / (tf[t] + k1) for t in tok(query) if tf.get(t)
    )


def test_bm25_score_matches_reference():
    text = "the quick brown fox jumps over the lazy dog"
    query = "quick fox"
    assert benostreamdb.bm25_score(text, query) == pytest.approx(
        _reference_bm25(text, query)
    )


def test_bm25_score_repeated_term_saturates():
    # A term appearing twice scores more than once, but less than double.
    once = benostreamdb.bm25_score("fox", "fox")
    twice = benostreamdb.bm25_score("fox fox", "fox")
    assert twice > once
    assert twice < 2 * once


def test_bm25_score_no_match_is_zero():
    assert benostreamdb.bm25_score("the quick brown fox", "zebra") == 0.0


def test_tf_idf_matches_reference():
    text = "a b a c"
    got = benostreamdb.tf_idf(text)
    # tokens: a, b, a, c -> counts a=2, b=1, c=1; total=4
    assert got == pytest.approx([2 / 4, 1 / 4, 1 / 4])


def test_tf_idf_empty_is_empty():
    assert benostreamdb.tf_idf("") == []


def test_tf_idf_weights_sum_to_one():
    got = benostreamdb.tf_idf("one two two three three three")
    assert math.isclose(sum(got), 1.0, rel_tol=1e-6)
