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
}
