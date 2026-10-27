package com.benostreamdb.trino;

/**
 * Maps a Trino schema/table name to a BenoStreamDB table URI under a
 * configurable warehouse root (the {@code benostream.warehouse} catalog
 * property, default {@code s3://default}).
 */
public final class BenoStreamDBTableUri {

    /** Default warehouse root when {@code benostream.warehouse} is not set. */
    public static final String DEFAULT_WAREHOUSE = "s3://default";

    private BenoStreamDBTableUri() {
    }

    public static String of(String warehouse, String schemaName, String tableName) {
        String base = warehouse == null || warehouse.isEmpty() ? DEFAULT_WAREHOUSE : warehouse;
        while (base.endsWith("/")) {
            base = base.substring(0, base.length() - 1);
        }
        return base + "/" + schemaName + "/" + tableName;
    }

    /**
     * Resolve a user-supplied table reference to a warehouse URI.
     *
     * Accepts a full URI, {@code table}, {@code schema.table}, or
     * {@code catalog.schema.table}. A bare table name is placed in the
     * {@code default} schema, matching the connector's documented examples.
     */
    public static String resolve(String warehouse, String table) {
        if (table == null || table.trim().isEmpty()) {
            throw new IllegalArgumentException("BenoStreamDB table name is required");
        }
        String value = table.trim();
        if (value.contains("://")) {
            return value;
        }
        String[] parts = value.split("\\.");
        if (parts.length == 1) {
            return of(warehouse, "default", parts[0]);
        }
        if (parts.length == 2) {
            return of(warehouse, parts[0], parts[1]);
        }
        if (parts.length == 3) {
            return of(warehouse, parts[1], parts[2]);
        }
        throw new IllegalArgumentException(
                "Expected table, schema.table, or catalog.schema.table, got: " + table);
    }

    /**
     * Split a table-function argument into {@code [schema, table]}.
     *
     * <p>Table functions require a catalog-relative name; full URIs are
     * rejected because the function's split must be addressable by the
     * connector's schema/table handle.</p>
     */
    public static String[] splitTable(String table) {
        if (table == null || table.trim().isEmpty()) {
            throw new IllegalArgumentException("BenoStreamDB table name is required");
        }
        String value = table.trim();
        if (value.contains("://")) {
            throw new IllegalArgumentException(
                    "Table functions require table, schema.table, or catalog.schema.table, not a URI");
        }
        String[] parts = value.split("\\.");
        if (parts.length == 1) {
            return new String[] {"default", parts[0]};
        }
        if (parts.length == 2) {
            return new String[] {parts[0], parts[1]};
        }
        if (parts.length == 3) {
            return new String[] {parts[1], parts[2]};
        }
        throw new IllegalArgumentException(
                "Expected table, schema.table, or catalog.schema.table, got: " + table);
    }
}
