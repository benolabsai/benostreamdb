package com.benostreamdb.trino;

import io.trino.spi.StandardErrorCode;
import io.trino.spi.TrinoException;
import io.trino.spi.connector.ColumnMetadata;
import io.trino.spi.function.table.Descriptor;
import io.trino.spi.type.Type;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * SQL and schema helpers for {@code graph_rag_search}.
 *
 * <p>The statement is evaluated in a session where the document table is
 * {@code t} and the edge table is {@code e}; both aliases are registered by
 * {@code openQueryMulti}. Graph traversal uses the literal {@code 'e'} because
 * the engine's graph table functions take the table name as a string.</p>
 */
public final class GraphRagSql {

    private static final Set<String> MODES = Set.of("auto", "in_memory", "out_of_core", "cached");

    private GraphRagSql() {
    }

    public static String build(GraphRagSearchFunction.Handle handle) {
        String mode = handle.getMode().trim().toLowerCase(java.util.Locale.ENGLISH);
        if (!MODES.contains(mode)) {
            throw userError("mode must be one of " + MODES + ", got: " + handle.getMode());
        }
        // The query vector is a numeric literal list, not an identifier; it is
        // validated as such and interpolated into the bracketed literal form the
        // KNN optimizer expects.
        String vector = validateVector(handle.getQueryVector());
        String idColumn = validateIdentifier(handle.getIdColumn(), "id_column");
        String vectorColumn = validateIdentifier(handle.getVectorColumn(), "vector_column");

        // `dist_cosine` over the literal is what the KNN optimizer recognises;
        // `ARRAY[...]` does not reach the index.
        return "SELECT d.*, g.hop FROM t d"
                + " JOIN graph_neighbors('e', '" + handle.getSeeds() + "', " + handle.getHops()
                + ", '" + mode + "') g ON d." + idColumn + " = g.node"
                + " ORDER BY dist_cosine(d." + vectorColumn + ", '[" + vector + "]')"
                + " LIMIT " + handle.getK();
    }

    public static Descriptor descriptor(List<ColumnMetadata> documentColumns) {
        List<String> names = new ArrayList<>();
        List<Type> types = new ArrayList<>();
        for (ColumnMetadata column : documentColumns) {
            names.add(column.getName());
            types.add(column.getType());
        }
        // `hop` comes from the traversal, not from the document.
        names.add("hop");
        types.add(io.trino.spi.type.IntegerType.INTEGER);
        return Descriptor.descriptor(names, types);
    }

    private static String validateVector(String value) {
        if (value == null || !value.matches("[-+0-9eE.,\\s]+")) {
            throw userError("query_vector must be a comma-separated list of numbers, got: " + value);
        }
        return value.trim();
    }

    private static String validateIdentifier(String value, String label) {
        if (value == null || !value.matches("[A-Za-z_][A-Za-z0-9_]*")) {
            throw userError(label + " must be a simple column identifier, got: " + value);
        }
        return value;
    }

    private static TrinoException userError(String message) {
        return new TrinoException(StandardErrorCode.GENERIC_USER_ERROR, message);
    }
}
