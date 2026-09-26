package com.benostreamdb.trino;

public class BenoStreamDBJNIBridge {
    private static boolean loaded = false;

    static {
        try {
            System.loadLibrary("benostreamdb");
            loaded = true;
            System.out.println("Successfully loaded native BenoStreamDB library for Trino.");
        } catch (UnsatisfiedLinkError e) {
            System.err.println("Failed to load native BenoStreamDB library: " + e.getMessage() + ". Using fallback/mock implementation for testing.");
        }
    }

    public static boolean isLoaded() {
        return loaded;
    }

    // GPU context configuration
    public static native boolean setGpuContext(String deviceType);

    // Vector Search
    public static native int vectorSearch(String table, String segmentId, String column, int k, long queryVectorPtr, int queryVectorLen, long outArrayPtr, long outSchemaPtr);

    // Schema resolution: returns a JSON array of {name, type, nullable}.
    public static native String getTableSchema(String tableUri);

    // Primary key: returns a JSON array of column names.
    public static native String getPrimaryKey(String tableUri);

    // DDL: create a table from a JSON array of {name, type, nullable}.
    public static native boolean createTable(String tableUri, String schemaJson);

    // Write path: append an Arrow batch (C Data Interface) and commit.
    public static native boolean appendBatch(String tableUri, long inArrayPtr, long inSchemaPtr);

    // Merge path: upsert an Arrow batch on the given comma-separated key columns.
    public static native boolean mergeRows(String tableUri, String keyColumns, long inArrayPtr, long inSchemaPtr);

    // Delete path: delete rows matching a SQL filter.
    public static native boolean deleteRows(String tableUri, String filter);
}
