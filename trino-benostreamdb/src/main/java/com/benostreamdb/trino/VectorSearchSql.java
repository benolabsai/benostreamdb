package com.benostreamdb.trino;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

/**
 * Builds the engine SQL for a vector-search table function call.
 *
 * <p>The connector registers the target table as {@code t} in
 * {@code openQuery}, so the generated statement always scans {@code t}.</p>
 */
public final class VectorSearchSql {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final Pattern IDENTIFIER = Pattern.compile("[A-Za-z_][A-Za-z0-9_]*");
    private static final long MAX_K = 10_000L;
    private static final int MAX_DIMENSION = 65_536;

    private VectorSearchSql() {
    }

    public static String build(VectorSearchHandle handle) {
        String[] schemaTable = BenoStreamDBTableUri.splitTable(handle.getTable());
        validateIdentifier(schemaTable[0], "table");
        validateIdentifier(schemaTable[1], "table");
        validateIdentifier(handle.getColumn(), "column");

        long k = handle.getK();
        if (k < 1 || k > MAX_K) {
            throw new IllegalArgumentException("k must be between 1 and " + MAX_K + ", got " + k);
        }

        // A single-quoted vector literal is rewritten to a FixedSizeList by the
        // engine's PgVectorRewriter, which is what the KNN optimizer recognizes.
        String literal = "'[" + String.join(", ", parseVector(handle.getQuery())) + "]'";
        String distance = "dist_cosine(" + handle.getColumn() + ", " + literal + ")";

        StringBuilder sql = new StringBuilder();
        sql.append("SELECT *, ").append(distance).append(" AS distance FROM t");

        handle.getFilter()
                .filter(filter -> !filter.isBlank())
                .ifPresent(filter -> {
                    validateFilter(filter);
                    sql.append(" WHERE (").append(filter).append(")");
                });

        sql.append(" ORDER BY ").append(distance).append(" LIMIT ").append(k);
        return sql.toString();
    }

    static List<String> parseVector(String query) {
        if (query == null || query.isBlank()) {
            throw new IllegalArgumentException("query must be a non-empty JSON array");
        }
        final JsonNode root;
        try {
            root = MAPPER.readTree(query);
        } catch (Exception e) {
            throw new IllegalArgumentException("query must be a JSON array of numbers", e);
        }
        if (root == null || !root.isArray() || root.isEmpty()) {
            throw new IllegalArgumentException("query must be a non-empty JSON array");
        }
        if (root.size() > MAX_DIMENSION) {
            throw new IllegalArgumentException(
                    "query has " + root.size() + " dimensions; maximum is " + MAX_DIMENSION);
        }

        List<String> values = new ArrayList<>(root.size());
        for (JsonNode node : root) {
            if (!node.isNumber()) {
                throw new IllegalArgumentException("query must contain only numbers");
            }
            double value = node.asDouble();
            if (!Double.isFinite(value)) {
                throw new IllegalArgumentException("query contains a non-finite number");
            }
            values.add(Double.toString(value));
        }
        return values;
    }

    private static void validateIdentifier(String value, String label) {
        if (value == null || !IDENTIFIER.matcher(value).matches()) {
            throw new IllegalArgumentException(
                    "Invalid " + label + " identifier: " + value);
        }
    }

    private static void validateFilter(String filter) {
        if (filter.contains(";")
                || filter.contains("--")
                || filter.contains("/*")
                || filter.contains("*/")) {
            throw new IllegalArgumentException("filter contains a disallowed SQL token");
        }
    }
}
