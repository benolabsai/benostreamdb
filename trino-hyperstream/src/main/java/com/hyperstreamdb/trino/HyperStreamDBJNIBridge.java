package com.hyperstreamdb.trino;

public class HyperStreamDBJNIBridge {
    private static boolean loaded = false;

    static {
        try {
            System.loadLibrary("hyperstreamdb");
            loaded = true;
            System.out.println("Successfully loaded native HyperStreamDB library for Trino.");
        } catch (UnsatisfiedLinkError e) {
            System.err.println("Failed to load native HyperStreamDB library: " + e.getMessage() + ". Using fallback/mock implementation for testing.");
        }
    }

    public static boolean isLoaded() {
        return loaded;
    }

    // GPU context configuration
    public static native boolean setGpuContext(String deviceType);
    
    // We can migrate other JNI methods here in the future if needed
}
