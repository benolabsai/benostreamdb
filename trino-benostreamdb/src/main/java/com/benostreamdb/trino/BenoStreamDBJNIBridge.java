package com.benostreamdb.trino;

public class BenoStreamDBJNIBridge {
    private static boolean loaded = false;

    static {
        try {
            System.loadLibrary("benostreamdb");
            loaded = true;
            System.out.println("Successfully loaded native BenoStreamDB library for Trino.");
        } catch (UnsatisfiedLinkError e) {
            System.err.println("Failed to load native BenoStreamDB library: " + e.getMessage()
                    + ". The connector will fail on use; add libbenostreamdb to java.library.path.");
        }
    }

    public static boolean isLoaded() {
        return loaded;
    }

    // GPU context configuration
    public static native boolean setGpuContext(String deviceType);

    // Multi-GPU: install a comma-separated device pool (e.g. "cuda:0,cuda:1").
    // Each engine worker thread is assigned one device round-robin. A single
    // device (or empty) is a no-op and `setGpuContext` still applies.
    public static native boolean setGpuDevicePool(String devices);

    // Vector Search
    public static native int vectorSearch(String table, String segmentId, String column, int k, long queryVectorPtr, int queryVectorLen, long outArrayPtr, long outSchemaPtr);

    // Schema resolution: returns a JSON array of {name, type, nullable}.
    public static native String getTableSchema(String tableUri);

    // Primary key: returns a JSON array of column names.
    public static native String getPrimaryKey(String tableUri);

    // Metadata listing: JSON arrays of schema / table names under the warehouse.
    public static native String listSchemas(String warehouse);

    public static native String listTables(String warehouse, String schema);

    // DDL: create a schema (a directory under the warehouse) / drop a table.
    public static native boolean createSchema(String warehouse, String schema);

    public static native boolean dropTable(String tableUri);

    // SQL query pushdown: run a query through the engine's session (full index
    // and vector-search pushdown) and stream the result batches.
    public static native long openQuery(String tableUri, String sql);

    public static native long readQueryBatch(long handle, long outArrayPtr, long outSchemaPtr);

    public static native void closeQuery(long handle);

    // DDL: create a table from a JSON array of {name, type, nullable}.
    public static native boolean createTable(String tableUri, String schemaJson);

    // Write path: append an Arrow batch (C Data Interface) and commit.
    public static native boolean appendBatch(String tableUri, long inArrayPtr, long inSchemaPtr);

    // Merge path: upsert an Arrow batch on the given comma-separated key columns.
    public static native boolean mergeRows(String tableUri, String keyColumns, long inArrayPtr, long inSchemaPtr);

    // Delete path: delete rows matching a SQL filter.
    public static native boolean deleteRows(String tableUri, String filter);

    // Observability: render the engine's Prometheus metrics as text.
    public static native String gatherMetrics();
}
