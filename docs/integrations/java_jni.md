# Java JNI Integration

To support big data ecosystems like **Trino** and **Spark**, HyperStreamDB exposes its core functionality to the JVM via the Java Native Interface (JNI).

## Overview

The JNI layer acts as a bridge between the JVM and the Rust core. It avoids the overhead of standard object serialization by leveraging the **Arrow C Data Interface**:

1.  **Rust Side**: Reads data from storage, filters it using indexes, and produces Apache Arrow batches in native memory.
2.  **JNI Bridge**: Passes pointers to these memory structures to Java.
3.  **Java Side**: Wraps these memory pointers using the Java Arrow library, allowing "Zero-Copy" access.

## Build Requirements

*   **Rust**: Stable toolchain (1.75+)
*   **Java**: JDK 11 or higher
*   **Maven**: For building the Java modules

## Shared Library

The core logic is compiled into a dynamic library:
*   `libhyperstreamdb.so` (Linux, WSL2)
*   `libhyperstreamdb.dylib` (macOS)

Note: Native Windows `.dll` builds are no longer supported. Windows users should use WSL2.

This library is packed into the JARs for Trino and Spark connectors.

## Key Methods

The JNI interface exposes methods for:
*   `openSession(path)`: Initialize a reader for a dataset.
*   `getSplits(filter)`: Retrieve a list of relevant segments based on the SQL filter (utilizing indexes).
*   `readBatch(split)`: Read a batch of data from a specific split into an Arrow format.
