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
    
    // We can migrate other JNI methods here in the future if needed
}
