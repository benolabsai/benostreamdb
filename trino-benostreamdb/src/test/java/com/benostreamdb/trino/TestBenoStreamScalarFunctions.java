package com.benostreamdb.trino;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Arithmetic coverage for the shared helpers behind the scalar functions.
 *
 * <p>The SQL entry points take Trino {@code Block}s and are exercised end to end
 * in {@code TestBenoStreamDBQueries}, which asserts registration, resolution,
 * and results for {@code l2_distance}, {@code vector_distance},
 * {@code sparse_dot_product}, {@code hybrid_score} and
 * {@code reciprocal_rank_fusion}. Keeping the pure arithmetic here means a
 * formula change and a wiring change fail in different places.</p>
 */
public class TestBenoStreamScalarFunctions {

    @Test
    void cosineDistanceMatchesSpark() {
        assertEquals(0.0, BenoStreamScalarFunctions.cosineDistance(List.of(1.0, 0.0), List.of(1.0, 0.0)), 1e-12);
        assertEquals(1.0, BenoStreamScalarFunctions.cosineDistance(List.of(1.0, 0.0), List.of(0.0, 1.0)), 1e-12);
        // Spark's `.max(0.0)` only floors the result at zero; it does not cap it
        // at 1, so opposing vectors legitimately yield 2.0.
        assertEquals(2.0, BenoStreamScalarFunctions.cosineDistance(List.of(1.0, 0.0), List.of(-1.0, 0.0)), 1e-12);
        // Zero vectors return 0 rather than NaN.
        assertEquals(0.0, BenoStreamScalarFunctions.cosineDistance(List.of(0.0), List.of(0.0)), 1e-12);
    }

    @Test
    void l2AndDotMatchSpark() {
        assertEquals(5.0, BenoStreamScalarFunctions.l2Distance(List.of(0.0, 0.0), List.of(3.0, 4.0)), 1e-12);
        assertEquals(11.0, BenoStreamScalarFunctions.dotProduct(List.of(1.0, 2.0), List.of(3.0, 4.0)), 1e-12);
    }

    @Test
    void hybridScoreClampsAlpha() {
        assertEquals(0.25, BenoStreamScalarFunctions.hybridScore(1.0, 0.0, 0.25), 1e-12);
        assertEquals(1.0, BenoStreamScalarFunctions.hybridScore(1.0, 0.0, 5.0), 1e-12);
        assertEquals(0.0, BenoStreamScalarFunctions.hybridScore(1.0, 0.0, -2.0), 1e-12);
    }
}
