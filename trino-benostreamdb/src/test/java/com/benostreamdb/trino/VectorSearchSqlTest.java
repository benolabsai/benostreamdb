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
