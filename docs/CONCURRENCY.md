# Concurrency and Atomic Commits

HyperStreamDB is designed for high-concurrency environments where multiple clients may be reading from and writing to the same table simultaneously.

## Optimistic Concurrency Control (OCC)

HyperStreamDB employs **Optimistic Concurrency Control** to ensure ACID compliance without the need for heavyweight central locks in most cases.

### Snapshot Versioning
Every table state is represented by a specific version of the manifest file (e.g., `_manifest/v100.json`). These files are immutable once written.

### The Commit Protocol
When a client (writer) wants to commit changes:
1.  **Read Latest**: The client reads the current latest version (e.g., `v100`).
2.  **Prepare**: The client calculates the new state (`v101`) based on the changes (e.g., added or removed segments).
3.  **Atomic Swap**: The client attempts to write the new manifest file `v101.json` using an **atomic "create-if-not-exists"** primitive.

### Conflict Resolution
If another client successfully committed `v101.json` while the first client was preparing its changes:
- The first client's write operation will fail with an `AlreadyExists` or conflict error.
- HyperStreamDB automatically **retries** the commit (up to 100 times).
- In each retry, the client re-reads the *new* latest version, merges its changes again, and attempts to commit the *next* version (e.g., `v102`).
- A randomized **exponential backoff** is used between retries to reduce contention.

## Catalog-Level Locking

While OCC works perfectly on local file systems and some cloud storage providers (like Azure Blob or Google Cloud Storage with certain settings), some providers like **AWS S3** do not natively support atomic "create-if-not-exists" with strong consistency for all operations.

In these cases, HyperStreamDB leverages **Iceberg-compatible catalogs** to provide the necessary atomicity:

- **AWS Glue**: Uses the Glue Catalog's built-in versioning and optimistic locking.
- **Nessie**: Provides Git-like branching and merging with cross-table atomic commits.
- **Hive Metastore**: Uses a relational database backend (like PostgreSQL or MySQL) to provide transactionally safe updates to the `metadata_location` parameter.
- **REST Catalog**: Delegates atomicity to a centralized REST server (e.g., Tabular, Polaris).

## Read Isolation

Readers in HyperStreamDB always see a **consistent snapshot** of the table. Once a reader loads a particular version (e.g., `v100`), it will continue to see that state even if newer versions are committed by other clients. This provides **Snapshot Isolation**, which is ideal for long-running analytical queries.
