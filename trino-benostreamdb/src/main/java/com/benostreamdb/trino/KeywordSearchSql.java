package com.benostreamdb.trino;

import java.util.Locale;
import java.util.regex.Pattern;

/**
 * Builds the engine SQL for a BM25 keyword search.
 *
 * <p>The engine triggers its BM25 index path on an <em>equality predicate</em>
 * against a column that carries a BM25 index (see
 * {@code src/core/sql/mod.rs}, "SQL BM25 Pushdown"): the right-hand literal
 * becomes the query string and the predicate is not applied as a row filter, so
 * ranking rows still match it. That is why the statement selects {@code *} and
 * repeats the equality in the {@code WHERE} clause rather than modelling the
 * query as a function call.</p>
 *
 * <p>The connector registers the target table as {@code t} inside
 * {@code openQuery}, so the statement always scans {@code t}.</p>
 */
public final class KeywordSearchSql {

    private static final Pattern IDENTIFIER = Pattern.compile("[A-Za-z_][A-Za-z0-9_]*");
    private static final long MAX_K = 10_000L;
    private static final int MAX_QUERY_LENGTH = 4_096;

    private KeywordSearchSql() {
    }

    public static String build(KeywordSearchHandle handle) {
        String[] schemaTable = BenoStreamDBTableUri.splitTable(handle.getTable());
        validateIdentifier(schemaTable[0], "table");
        validateIdentifier(schemaTable[1], "table");
        validateIdentifier(handle.getColumn(), "column");

        String query = handle.getQuery();
        if (query == null || query.isBlank()) {
            throw new IllegalArgumentException("query must not be blank");
        }
        if (query.length() > MAX_QUERY_LENGTH) {
            throw new IllegalArgumentException(
                    "query must be at most " + MAX_QUERY_LENGTH + " characters");
        }

        long k = handle.getK();
        if (k < 1 || k > MAX_K) {
            throw new IllegalArgumentException("k must be between 1 and " + MAX_K + ", got " + k);
        }

        String escaped = query.replace("'", "''");
        StringBuilder sql = new StringBuilder();
        sql.append("SELECT *, bm25_score(\"")
                .append(handle.getColumn())
                .append("\", '")
                .append(escaped)
                .append("') AS score FROM t WHERE \"")
                .append(handle.getColumn())
                .append("\" = '")
                .append(escaped)
                .append('\'');

        handle.getFilter()
                .map(String::trim)
                .filter(filter -> !filter.isBlank())
                .ifPresent(filter -> sql.append(" AND (").append(filter).append(')'));

        sql.append(" LIMIT ").append(k);
        return sql.toString();
    }

    static void validateIdentifier(String value, String what) {
        if (value == null || !IDENTIFIER.matcher(value).matches()) {
            throw new IllegalArgumentException(
                    "invalid " + what + " identifier: "
                            + (value == null ? "null" : "'" + value.toLowerCase(Locale.ROOT) + "'"));
        }
    }
}
