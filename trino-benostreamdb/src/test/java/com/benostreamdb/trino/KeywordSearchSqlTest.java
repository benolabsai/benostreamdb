package com.benostreamdb.trino;

import org.junit.Test;

import java.util.Optional;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

public class KeywordSearchSqlTest {

    private static KeywordSearchHandle handle(
            String column, String query, long k, Optional<String> filter) {
        return new KeywordSearchHandle(
                "default.docs", "t", column, query, k, filter);
    }

    @Test
    public void buildsEqualityPredicateThatTriggersBm25Pushdown() {
        String sql = KeywordSearchSql.build(handle("body", "vector database", 7, Optional.empty()));

        // The engine only takes the BM25 index path for an equality predicate on
        // a BM25-indexed column, so the query must appear as `column = 'text'`.
        assertTrue(sql.contains("WHERE \"body\" = 'vector database'"));
        assertTrue(sql.contains("bm25_score(\"body\", 'vector database') AS score"));
        assertTrue(sql.contains("FROM t"));
        assertTrue(sql.contains("LIMIT 7"));
        // An ORDER BY would force a sort above the scan and lose the index path.
        assertFalse(sql.contains("ORDER BY"));
    }

    @Test
    public void appendsFilterInsideWhere() {
        String sql = KeywordSearchSql.build(
                handle("body", "graph", 10, Optional.of("lang = 'en'")));

        assertTrue(sql.contains("WHERE \"body\" = 'graph' AND (lang = 'en')"));
        assertTrue(sql.endsWith("LIMIT 10"));
    }

    @Test
    public void ignoresBlankFilter() {
        String sql = KeywordSearchSql.build(handle("body", "graph", 10, Optional.of("   ")));

        assertFalse(sql.contains("AND ("));
    }

    @Test
    public void escapesSingleQuotesInQuery() {
        String sql = KeywordSearchSql.build(handle("body", "it's here", 5, Optional.empty()));

        assertTrue(sql.contains("'it''s here'"));
    }

    @Test
    public void rejectsOutOfRangeK() {
        try {
            KeywordSearchSql.build(handle("body", "q", 0, Optional.empty()));
            fail("expected IllegalArgumentException for k=0");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("k must be between"));
        }
    }

    @Test
    public void rejectsBlankQuery() {
        try {
            KeywordSearchSql.build(handle("body", "   ", 5, Optional.empty()));
            fail("expected IllegalArgumentException for a blank query");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("query must not be blank"));
        }
    }

    @Test
    public void rejectsNonIdentifierColumn() {
        try {
            KeywordSearchSql.build(handle("body\"); DROP TABLE x; --", "q", 5, Optional.empty()));
            fail("expected IllegalArgumentException for an injected column name");
        } catch (IllegalArgumentException expected) {
            assertEquals("invalid column identifier: 'body\"); drop table x; --'",
                    expected.getMessage());
        }
    }
}
