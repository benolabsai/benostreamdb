package com.benostreamdb.trino;

import org.junit.Test;

import java.util.Optional;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class VectorSearchSqlTest {

    @Test
    public void buildsKnnSqlWithFilter() {
        VectorSearchHandle handle = new VectorSearchHandle(
                "default.docs",
                "embedding",
                "[0.1, 0.2, 0.3]",
                5,
                Optional.of("category = 'science'"));

        String sql = VectorSearchSql.build(handle);

        assertTrue(sql.contains("dist_cosine(embedding, '[0.1, 0.2, 0.3]')"));
        assertTrue(sql.contains("WHERE (category = 'science')"));
        assertTrue(sql.contains("LIMIT 5"));
        assertTrue(sql.contains("ORDER BY dist_cosine("));
        assertFalse(sql.contains("ARRAY["));
    }

    @Test
    public void buildsKnnSqlWithoutFilter() {
        VectorSearchHandle handle = new VectorSearchHandle(
                "default.docs", "embedding", "[1.0, 0.0]", 3, Optional.empty());

        String sql = VectorSearchSql.build(handle);

        assertFalse(sql.contains("WHERE"));
        assertTrue(sql.contains("LIMIT 3"));
    }

    @Test
    public void parsesVector() {
        assertEquals(3, VectorSearchSql.parseVector("[0.1, 0.2, 0.3]").size());
    }

    @Test
    public void defaultsToCosineDistance() {
        VectorSearchHandle handle = new VectorSearchHandle(
                "default.docs", "embedding", "[1.0, 0.0]", 3, Optional.empty());

        assertEquals(VectorSearchHandle.DEFAULT_METRIC, handle.getMetric());
        assertTrue(VectorSearchSql.build(handle).contains("dist_cosine(embedding,"));
    }

    @Test
    public void honoursEachMetric() {
        assertTrue(sqlFor("l2").contains("dist_l2(embedding,"));
        assertTrue(sqlFor("euclidean").contains("dist_l2(embedding,"));
        assertTrue(sqlFor("dot").contains("dist_ip(embedding,"));
        assertTrue(sqlFor("inner_product").contains("dist_ip(embedding,"));
        assertTrue(sqlFor("l1").contains("dist_l1(embedding,"));
        assertTrue(sqlFor("manhattan").contains("dist_l1(embedding,"));
        assertTrue(sqlFor("hamming").contains("dist_hamming(embedding,"));
        assertTrue(sqlFor("jaccard").contains("dist_jaccard(embedding,"));
    }

    @Test
    public void metricDrivesBothSelectAndOrderBy() {
        String sql = sqlFor("l2");

        assertTrue(sql.contains("dist_l2(embedding, '[0.1, 0.2]') AS distance"));
        assertTrue(sql.contains("ORDER BY dist_l2(embedding, '[0.1, 0.2]')"));
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsUnknownMetric() {
        sqlFor("levenshtein");
    }

    private static String sqlFor(String metric) {
        return VectorSearchSql.build(new VectorSearchHandle(
                "default.docs",
                "embedding",
                "[0.1, 0.2]",
                5,
                Optional.empty(),
                metric));
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsNonNumericVector() {
        VectorSearchSql.parseVector("[0.1, nope]");
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsEmptyVector() {
        VectorSearchSql.parseVector("[]");
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsOutOfRangeK() {
        VectorSearchSql.build(new VectorSearchHandle(
                "default.docs", "embedding", "[0.1]", 0, Optional.empty()));
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsFilterWithSemicolon() {
        VectorSearchSql.build(new VectorSearchHandle(
                "default.docs",
                "embedding",
                "[0.1]",
                5,
                Optional.of("category = 'science'; DROP TABLE docs")));
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsInvalidColumnIdentifier() {
        VectorSearchSql.build(new VectorSearchHandle(
                "default.docs",
                "embedding) DROP",
                "[0.1]",
                5,
                Optional.empty()));
    }
}
